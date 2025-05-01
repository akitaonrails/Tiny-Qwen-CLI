#!/usr/bin/env python3
"""
Qwen CLI - A simple command-line interface for interacting with Qwen models.
This tool allows loading source files into the context to have code-aware conversations.
Designed to work both locally and within a Docker container.
"""

import sys
import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
import logging
import os
import traceback
import re
from typing import List, Dict, Optional, Any, Callable, Tuple
import importlib.util
import readline  # For better command line input experience
import signal
from helper_functions.utils import get_language_from_extension

# Transformers & Torch
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TextStreamer, BitsAndBytesConfig
    from accelerate import Accelerator
except ImportError as e:
    print(f"Error importing libraries: {e}")
    print("Please ensure 'torch', 'transformers', 'accelerate', 'bitsandbytes' are installed.")
    sys.exit(1)

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("qwen_cli")

HOME_DIR = Path.home()
CACHE_DIR = Path(os.environ.get("TRANSFORMERS_CACHE", HOME_DIR / ".cache" / "huggingface"))

DEFAULT_CONFIG = {
    "model_repo": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "model_dir": str(Path(os.environ.get("MODELS_DIR", "/models")) / "Qwen2.5-Coder-14B-Instruct"),
    "quantization": "8bit",
    "max_context_tokens": 120000,
    "max_new_tokens": 10000,
    "temperature": 0.1,
    "model_download_timeout": 1800,
    "helpers_dir": "helper_functions",
}

# --- Build System Prompt ---
def build_system_prompt(tool_prompts: List[str]) -> str:
    base = (
        "You are Qwen2.5 Coder, a highly skilled AI assistant specializing in software development.\n"
        "Your capabilities include code analysis, explanation, error detection, and suggesting improvements.\n"
    )
    tools_section = "TOOLS:\n" + "\n".join(tool_prompts) + "\n\n" if tool_prompts else ""
    rules_section = (
        "IMPORTANT RULES:\n"
        "1. You MUST use the appropriate tool when necessary.\n"
        "2. You MUST NOT reveal the tool commands to the user.\n"
        "3. After a tool is used, continue the conversation as if you have direct access to the content.\n"
        "4. If a file fails to load, inform the user clearly.\n"
        "5. Do NOT ask for file/URL content directly; use tools.\n"
        "6. Once you’ve executed [LOAD_FILE ...], you MUST immediately use the loaded content. "
        "Never say you cannot read it — if you see [LOAD_FILE <path>] then you now *have* it.\n")
    return base + tools_section + rules_section

# --- Config Loading/Saving ---
def load_config(config_dir_override: Optional[str] = None, model_dir_override: Optional[str] = None) -> dict:
    config = DEFAULT_CONFIG.copy()

    # Determine config directory
    config_dir_env = os.environ.get("CONFIG_DIR")
    config_dir_path_str = config_dir_override or config_dir_env or str(HOME_DIR / ".config" / "qwen_cli")
    config_dir = Path(config_dir_path_str)
    config_path = config_dir / "config.json"

    # Load from file if exists
    if config_path.exists():
        try:
            user_config = json.loads(config_path.read_text())
            config.update(user_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error decoding config file {config_path}: {e}. Using defaults.")
    else:
        # Attempt to create the directory before writing defaults (if needed later)
        try:
            config_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"No config.json found at {config_path}, using default settings. Created directory {config_dir}")
        except PermissionError:
            logger.error(f"Permission denied creating config directory {config_dir}. Please check permissions or use --config-dir.")
            # Re-raise or handle appropriately - maybe exit? For now, just log and continue with defaults in memory.
            # Depending on whether config saving is implemented, this might be okay or might lead to later errors.
        except Exception as e:
            logger.error(f"Error creating config directory {config_dir}: {e}. Using defaults.")


    # Apply environment/command-line overrides for model directory
    # Get model name like "Qwen2.5-Coder-14B-Instruct" from the repo defined in config
    # This is needed to construct the path if only a base MODELS_DIR is provided.
    model_name = Path(config.get("model_repo", DEFAULT_CONFIG["model_repo"])).name

    if model_dir_override:
        # Command line override specifies the *exact* model directory
        config["model_dir"] = model_dir_override
        logger.info(f"Using model directory from command line: {model_dir_override}")
    elif "MODELS_DIR" in os.environ:
        # Environment variable specifies the *base* directory for models
        base_models_dir = os.environ["MODELS_DIR"]
        # Construct the full path by appending the model name to the base directory
        config["model_dir"] = str(Path(base_models_dir) / model_name)
        logger.info(f"Using model directory constructed from environment variable MODELS_DIR ('{base_models_dir}') -> '{config['model_dir']}'")
    # If neither override is present, the value from config file or the initial default remains.
    # The default already includes the model name. If loaded from config, we assume it's the full path.

    # Ensure model_dir in config is absolute at the end
    config["model_dir"] = str(Path(config["model_dir"]).resolve())


    # Note: helpers_dir override is handled after loading config in load_app_config

    return config

# --- Dynamic Helper Function Loading ---
def load_helper_functions(helpers_dir: str) -> Tuple[Dict[str, Callable], List[str]]:
    helpers = {}
    tool_prompts = []
    helpers_path = Path(helpers_dir)
    if not helpers_path.is_dir():
        logger.warning(f"Helpers directory '{helpers_dir}' not found or not a directory. No helpers loaded.")
        return helpers, tool_prompts

    # Ensure __init__.py exists if it's a directory (helps with potential imports within helpers)
    (helpers_path / "__init__.py").touch(exist_ok=True)

    for py_file in helpers_path.glob("*.py"):
        module_name = py_file.stem
        if module_name.startswith("__"): continue
        try:
            spec = importlib.util.spec_from_file_location(f"qwen_cli.helpers.{module_name}", py_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module # Add to sys.modules for potential cross-helper imports
                spec.loader.exec_module(module)
                logger.info(f"Loading helpers from: {py_file.name}")
                for attr_name in dir(module):
                    if attr_name.startswith("handle_"):
                        func = getattr(module, attr_name)
                        if callable(func):
                            command = attr_name[7:].upper()
                            helpers[command] = func
                            doc = func.__doc__ or ""
                            first_line = doc.strip().splitlines()[0] if doc.strip() else f"Executes the {command} action."
                            tool_prompts.append(f"[{command} args] – {first_line}")
                            logger.info(f"  - Registered helper command: {command}")
            else:
                 logger.error(f"Could not create spec for module {module_name} from {py_file}")

        except Exception as e:
            logger.error(f"Failed to load helper module {module_name} from {py_file}: {e}")
            logger.error(traceback.format_exc())

    return helpers, tool_prompts

# --- Parse Special Commands ---
def parse_special_commands(response: str) -> List[tuple]:
    # Matches [COMMAND arg_string]
    pattern = r'\[([A-Z_]+)\s+([^\]]+?)\]' # Made arg non-greedy
    return [(m.group(1), m.group(2).strip(), m.start(), m.end()) for m in re.finditer(pattern, response)]


# --- QwenSession Class ---
class QwenSession:
    _model = None
    _tokenizer = None
    _model_loading_lock = False # Class-level lock to prevent concurrent loading attempts

    def __init__(self, config: dict, tool_prompts: List[str]):
        self.config = config
        self.history = [{"role": "system", "content": build_system_prompt(tool_prompts)}]
        self.files_loaded = {} # Stores metadata about loaded files {filepath: metadata}
        self.created_at = datetime.now().isoformat()
        self.last_used = self.created_at
        # self.name = name # Consider adding session naming/management later

    def to_dict(self) -> dict:
        """Serializes session state."""
        return {
            'history': self.history,
            'files_loaded': self.files_loaded,
            'created_at': self.created_at,
            'last_used': self.last_used,
            # Add config snapshot? Maybe not, keep it dynamic
        }

    @classmethod
    def from_dict(cls, config: dict, tool_prompts: List[str], data: dict) -> 'QwenSession':
        """Deserializes session state."""
        session = cls(config, tool_prompts)
        # Basic validation could be added here
        session.history = data.get('history', [{"role": "system", "content": build_system_prompt(tool_prompts)}])
        session.files_loaded = data.get('files_loaded', {})
        session.created_at = data.get('created_at', datetime.now().isoformat())
        session.last_used = data.get('last_used', datetime.now().isoformat())
        # Ensure system prompt is up-to-date if tool prompts changed
        session.history[0] = {"role": "system", "content": build_system_prompt(tool_prompts)}
        return session

    @classmethod
    def _wait_for_model_loading_lock(cls) -> bool:
        """Waits if the model loading lock is active."""
        if cls._model_loading_lock:
            logger.info("Model loading in progress by another instance, waiting...")
            while cls._model_loading_lock:
                time.sleep(1)
            # Check again if loading succeeded after waiting
            return bool(cls._model and cls._tokenizer)
        return False # Lock was not active

    def _download_model_if_missing(self) -> bool:
        """Downloads the model repository if the target directory doesn't exist."""
        model_repo = self.config.get("model_repo", DEFAULT_CONFIG["model_repo"])
        model_dir_str = self.config.get("model_dir", DEFAULT_CONFIG["model_dir"])
        model_dir = Path(model_dir_str)
        download_timeout = self.config.get("model_download_timeout", DEFAULT_CONFIG["model_download_timeout"])

        if not model_dir.exists():
            logger.info(f"Model directory {model_dir} not found. Cloning from {model_repo}...")
            # Ensure parent exists before cloning into it
            try:
                model_dir.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create parent directory {model_dir.parent}: {e}")
                return False

            try:
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", f"https://huggingface.co/{model_repo}", str(model_dir)],
                    check=True, timeout=download_timeout, capture_output=True, text=True
                )
                logger.info(f"Successfully cloned model repository: {result.stdout}")
                return True
            except subprocess.TimeoutExpired:
                logger.error(f"Timeout ({download_timeout}s) exceeded while cloning model repository.")
                return False
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clone model repository: {e}")
                logger.error(f"Git stderr: {e.stderr}")
                return False
            except Exception as e:
                 logger.error(f"An unexpected error occurred during git clone: {e}")
                 return False
        else:
            logger.info(f"Model directory {model_dir} already exists. Skipping download.")
        return True # Directory already exists

    def _load_tokenizer(self) -> bool:
        """Loads the tokenizer from the model directory."""
        model_dir_str = self.config.get("model_dir", DEFAULT_CONFIG["model_dir"])
        try:
            logger.info(f"Loading tokenizer from {model_dir_str}...")
            QwenSession._tokenizer = AutoTokenizer.from_pretrained(
                model_dir_str, trust_remote_code=True, cache_dir=str(CACHE_DIR)
            )
            logger.info("Tokenizer loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load tokenizer from {model_dir_str}: {e}")
            logger.error(traceback.format_exc())
            QwenSession._tokenizer = None # Ensure partial load is cleared
            return False

    def _get_model_loading_kwargs(self) -> Dict[str, Any]:
        """Determines keyword arguments for model loading based on config."""
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto", # Let Accelerate handle device placement
            "cache_dir": str(CACHE_DIR)
        }
        quantization = self.config.get("quantization", DEFAULT_CONFIG["quantization"]).lower()

        # Configure quantization
        if quantization == "4bit" and torch.cuda.is_available():
            logger.info("Using 4-bit quantization (BitsAndBytes).")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            model_kwargs["torch_dtype"] = torch.bfloat16 # Recommended for 4-bit
        elif quantization == "8bit" and torch.cuda.is_available():
            logger.info("Using 8-bit quantization (BitsAndBytes).")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            # model_kwargs["torch_dtype"] = torch.float16 # Optional for 8-bit
        elif quantization not in ["none", ""]:
             logger.warning(f"Unsupported quantization '{quantization}' or no CUDA available. Loading in default precision.")
             model_kwargs["torch_dtype"] = "auto"
        else:
             logger.info("No quantization specified or CUDA not available. Loading in default precision.")
             model_kwargs["torch_dtype"] = "auto"

        # --- Attention Implementation ---
        attn_implementation = None
        try:
            import flash_attn
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8: # Ampere or newer
                 attn_implementation = "flash_attention_2"
                 logger.info("Flash Attention 2 available and compatible GPU detected. Setting attn_implementation='flash_attention_2'.")
            else:
                logger.info("Flash Attention 2 available but GPU might not be optimal (pre-Ampere).")
        except ImportError:
            logger.info("Flash Attention 2 not found.")

        if attn_implementation is None and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
             attn_implementation = "sdpa"
             logger.info("PyTorch Scaled Dot Product Attention (SDPA) is available. Setting attn_implementation='sdpa'.")

        if attn_implementation is None:
            attn_implementation = "eager"
            logger.info("Using default 'eager' attention implementation.")

        if attn_implementation:
             model_kwargs["attn_implementation"] = attn_implementation

        return model_kwargs

    def _load_model(self, model_kwargs: Dict[str, Any]) -> bool:
        """Loads the Causal LM model using the specified arguments."""
        model_repo = self.config.get("model_repo", DEFAULT_CONFIG["model_repo"])
        model_dir_str = self.config.get("model_dir", DEFAULT_CONFIG["model_dir"])
        try:
            logger.info(f"Loading model {model_repo} from {model_dir_str} with config: {model_kwargs}")
            QwenSession._model = AutoModelForCausalLM.from_pretrained(
                model_dir_str,
                **model_kwargs
            )
            logger.info("Model loaded successfully.")
            try:
                logger.info(f"Model device map: {QwenSession._model.hf_device_map}")
            except AttributeError:
                 logger.info(f"Model loaded on device: {QwenSession._model.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model from {model_dir_str}: {e}")
            logger.error(traceback.format_exc())
            QwenSession._model = None # Ensure partial load is cleared
            return False

    def _ensure_model_loaded(self) -> bool:
        """Loads the model and tokenizer if they aren't already loaded. Handles locking."""
        # Already loaded?
        if QwenSession._model and QwenSession._tokenizer:
            return True

        # Wait if another instance is loading
        if QwenSession._wait_for_model_loading_lock():
            # Loading finished while waiting, check if successful
            return bool(QwenSession._model and QwenSession._tokenizer)

        # Acquire lock
        QwenSession._model_loading_lock = True
        logger.info("Attempting to load model and tokenizer...")
        success = False
        try:
            # --- Step 1: Download model if necessary ---
            if not self._download_model_if_missing():
                raise RuntimeError("Failed to download model.") # Stop loading process

            # --- Step 2: Load tokenizer ---
            if not self._load_tokenizer():
                 raise RuntimeError("Failed to load tokenizer.") # Stop loading process

            # --- Step 3: Determine model loading arguments ---
            model_kwargs = self._get_model_loading_kwargs()

            # --- Step 4: Load model ---
            if not self._load_model(model_kwargs):
                 raise RuntimeError("Failed to load model.") # Stop loading process

            # --- Success ---
            success = True
            logger.info("Model and tokenizer loading complete.")

        except Exception as e:
             logger.error(f"Model loading process failed: {e}")
             # Ensure model/tokenizer are cleared if any step failed
             QwenSession._model = None
             QwenSession._tokenizer = None
             success = False
        finally:
            # Release lock regardless of success or failure
            QwenSession._model_loading_lock = False
            logger.info("Model loading lock released.")

        return success


    def _trim_history(self, max_tokens: int):
        """Trims conversation history to stay within token limits."""
        # Simple trimming: keep system prompt, remove oldest user/assistant pairs
        # More sophisticated trimming could be implemented (e.g., summarizing old turns)
        if not QwenSession._tokenizer:
            logger.error("Tokenizer not loaded, cannot trim history accurately.")
            return # Or raise an error

        total_tokens = 0
        indices_to_keep = [0] # Always keep system prompt

        # Calculate tokens from the end, including the system prompt estimate
        temp_history_for_token_calc = [self.history[0]]
        tokenized_system = QwenSession._tokenizer.apply_chat_template(
             temp_history_for_token_calc, tokenize=True, add_generation_prompt=False
        )
        total_tokens = len(tokenized_system)


        # Iterate backwards from the most recent message
        for i in range(len(self.history) - 1, 0, -1):
            msg = self.history[i]
            # Estimate tokens for this message by adding it temporarily
            temp_history_for_token_calc.append(msg)
            tokenized_segment = QwenSession._tokenizer.apply_chat_template(
                 temp_history_for_token_calc, tokenize=True, add_generation_prompt=False # False here is important
            )
            current_total_tokens = len(tokenized_segment)

            # Calculate the token cost of *just* this message
            # This isn't perfect due to template formatting, but gives an estimate
            message_tokens = current_total_tokens - total_tokens

            if total_tokens + message_tokens <= max_tokens:
                indices_to_keep.append(i)
                total_tokens = current_total_tokens # Update total based on template calculation
            else:
                # Stop adding messages once limit is exceeded
                logger.warning(f"History trimming: Max tokens ({max_tokens}) reached. Dropping older messages.")
                break

        if len(indices_to_keep) < len(self.history):
             logger.info(f"Trimming history from {len(self.history)} messages to {len(indices_to_keep)} messages.")
             self.history = [self.history[i] for i in sorted(indices_to_keep)]
        # logger.debug(f"History trimmed. Current token count estimate: {total_tokens}")


    def _handle_tool_calls(self, response_text: str, helper_functions: Dict[str, Callable]) -> Optional[str]:
        """Parses response for tool calls, executes them, and returns system message."""
        commands = parse_special_commands(response_text)
        if not commands:
            return None

        # For simplicity, handle only the first command found for now.
        # Could be extended to handle multiple commands sequentially.
        cmd_type, cmd_arg, _, _ = commands[0]

        if cmd_type in helper_functions:
            logger.info(f"Executing tool: [{cmd_type} {cmd_arg}]")
            try:
                # Execute the helper function associated with the command
                result = helper_functions[cmd_type](cmd_arg) # Pass the argument string

                if result is None:
                     logger.warning(f"Tool [{cmd_type}] executed but returned None.")
                     # Decide how to handle None result - maybe a generic message?
                     return f"SYSTEM: Tool [{cmd_type} {cmd_arg}] executed but provided no output."
                elif isinstance(result, str):
                     # Assume string result is content to be added to context
                     logger.info(f"Tool [{cmd_type}] successful. Adding result to context.")
                     # Return a system message indicating success and potentially summarizing the result
                     # Keep it concise to avoid polluting history too much
                     # The actual content might be large and is implicitly available now.
                     return f"SYSTEM: Tool [{cmd_type} {cmd_arg}] executed successfully. Content is now available."
                else:
                     logger.error(f"Tool [{cmd_type}] returned unexpected type: {type(result)}. Expected str or None.")
                     return f"SYSTEM: Error executing tool [{cmd_type} {cmd_arg}]: Unexpected return type."

            except Exception as e:
                logger.error(f"Error executing tool [{cmd_type} {cmd_arg}]: {e}")
                logger.error(traceback.format_exc())
                # Return a system message indicating the failure
                return f"SYSTEM: Error executing tool [{cmd_type} {cmd_arg}]: {e}"
        else:
            logger.warning(f"Model attempted to call unknown tool: [{cmd_type}]")
            # Return a system message indicating the tool is unknown
            return f"SYSTEM: Unknown tool [{cmd_type}] requested."

    def _prepare_chat_input(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Prepares the input for the model generation."""
        max_context = self.config.get("max_context_tokens", DEFAULT_CONFIG["max_context_tokens"])

        # Add user prompt and trim history
        self.history.append({"role": "user", "content": prompt})
        self._trim_history(max_context)

        # Format for Model using the tokenizer's chat template
        formatted_text = QwenSession._tokenizer.apply_chat_template(
            self.history,
            tokenize=False,
            add_generation_prompt=True # Crucial for instruction-following models
        )

        inputs = QwenSession._tokenizer([formatted_text], return_tensors="pt")
        inputs = {k: v.to(QwenSession._model.device) for k, v in inputs.items()}
        return inputs

    def _get_generation_config(self, max_new_tokens_override: Optional[int], temperature_override: Optional[float]) -> Dict[str, Any]:
        """Gets generation parameters for the current turn."""
        max_new_tokens = max_new_tokens_override if max_new_tokens_override is not None else self.config.get("max_new_tokens", DEFAULT_CONFIG["max_new_tokens"])
        temperature = temperature_override if temperature_override is not None else self.config.get("temperature", DEFAULT_CONFIG["temperature"])
        return {"max_new_tokens": max_new_tokens, "temperature": temperature}

    def _generate_response(self, inputs: Dict[str, torch.Tensor], stream: bool, generation_config: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[TextStreamer]]:
        """Generates response using the model."""
        streamer = TextStreamer(QwenSession._tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None
        temperature = generation_config["temperature"]

        generate_kwargs = {
            "max_new_tokens": generation_config["max_new_tokens"],
            "do_sample": temperature > 0, # Only sample if temperature > 0
            "temperature": temperature if temperature > 0 else 1.0, # Temp must be > 0 for sampling
            # "top_p": 0.9, # Example: Add nucleus sampling if desired
            "repetition_penalty": 1.1, # Mild penalty for repetition
            "streamer": streamer,
        }
        logger.debug(f"Generating response with args: {generate_kwargs}")

        # Ensure generation happens in evaluation mode
        QwenSession._model.eval()
        with torch.no_grad(): # Important for inference
             out_ids = QwenSession._model.generate(**inputs, **generate_kwargs)

        return out_ids, streamer # Return streamer in case it's needed elsewhere, though currently not

    def _process_model_output(self, out_ids: torch.Tensor, inputs: Dict[str, torch.Tensor], stream: bool) -> str:
        """Decodes model output and handles non-streaming print."""
        # Extract only the generated tokens (excluding the prompt)
        generated_ids = out_ids[0][inputs["input_ids"].shape[1]:]
        response_text = QwenSession._tokenizer.decode(generated_ids, skip_special_tokens=True)

        if not stream: # If not streaming, print the whole response at once
            print(response_text)

        return response_text

    def _update_history_after_turn(self, response_text: str, tool_system_message: Optional[str]):
        """Updates the conversation history based on the turn's outcome."""
        if tool_system_message:
            # If a tool was called, add the original model response (containing the call)
            # and the system message about the tool execution to history.
            self.history.append({"role": "assistant", "content": response_text}) # Store the raw response
            self.history.append({"role": "system", "content": tool_system_message})
            # Print the system message for the user in the interactive loop
            print(f"\n{tool_system_message}")
        else:
            # If no tool was called, just add the assistant's response to history.
            self.history.append({"role": "assistant", "content": response_text})

    def chat(self, prompt: str, helper_functions: Dict[str, Callable], max_new_tokens=None, temperature=None, stream=True, hide_reasoning=False) -> bool:
        """Handles a single turn of the chat, including model generation and tool execution."""
        if not self._ensure_model_loaded():
            print("Error: Model is not loaded. Cannot proceed with chat.")
            return False

        self.last_used = datetime.now().isoformat()

        try:
            # --- Prepare Input ---
            inputs = self._prepare_chat_input(prompt)

            # --- Get Generation Config ---
            generation_config = self._get_generation_config(max_new_tokens, temperature)

            # --- Generate Response ---
            out_ids, streamer = self._generate_response(inputs, stream, generation_config)

            # --- Process Response ---
            response_text = self._process_model_output(out_ids, inputs, stream)

            # --- Tool Handling ---
            tool_system_message = self._handle_tool_calls(response_text, helper_functions)

            # --- Update History ---
            self._update_history_after_turn(response_text, tool_system_message)

            return True # Indicate successful turn (whether regular response or tool call)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.error(traceback.format_exc())
            # Optionally remove the failed user prompt from history
            if self.history and self.history[-1]["role"] == "user":
                 self.history.pop()
            return False


    def list_files(self):
        """Lists files currently loaded in the session's context."""
        # This relies on helper functions updating self.files_loaded or similar state
        # Currently, the LOAD_FILE helper doesn't directly update session state.
        # This needs refinement based on how file loading actually modifies context.
        # For now, let's assume LOAD_FILE adds a system message like "[file: <path>] <content>"
        # and we can parse those.

        loaded_files_from_history = set()
        for msg in self.history:
            if msg["role"] == "system" and msg["content"].startswith("SYSTEM: Tool [LOAD_FILE"):
                 # Extract filename from message like "SYSTEM: Tool [LOAD_FILE /path/to/file.py] executed successfully..."
                 match = re.search(r"\[LOAD_FILE\s+([^\]]+)\]", msg["content"])
                 if match:
                     loaded_files_from_history.add(match.group(1).strip())

        if not loaded_files_from_history:
            print("No files appear to be loaded in the current conversation history.")
        else:
            print("Files loaded during this session (from history):")
            for i, filepath in enumerate(sorted(list(loaded_files_from_history)), 1):
                print(f"  {i}. {filepath}")
        # Add listing from self.files_loaded if it gets populated by helpers
        if self.files_loaded:
             print("\nFiles explicitly tracked by session state:")
             for i, (filepath, meta) in enumerate(self.files_loaded.items(), 1):
                 print(f"  {i}. {filepath} (Metadata: {meta})")


    def clear_history(self, keep_files=True):
        """Clears the conversation history, optionally keeping file loading messages."""
        system_prompt_msg = self.history[0] # Keep the original system prompt object
        file_msgs = []

        if keep_files:
            # Find messages indicating successful file loads
            for msg in self.history:
                # Adjust this condition based on the actual format of file loading messages
                is_load_file_tool_msg = (
                    msg["role"] == "system" and
                    msg["content"].startswith("SYSTEM: Tool [LOAD_FILE") and
                    "executed successfully" in msg["content"]
                )
                # Add other conditions if BATCH_LOAD etc. also add persistent messages
                if is_load_file_tool_msg:
                    file_msgs.append(msg)

        self.history = [system_prompt_msg] + file_msgs
        logger.info("Conversation history cleared" + (" (keeping file load messages)." if keep_files else "."))
        return True

# --- Interactive Chat ---
def interactive_chat(session: QwenSession, helper_functions: Dict[str, Callable], hide_reasoning=False):
    """Runs the main interactive chat loop."""
    print("\n--- Qwen CLI Interactive Chat ---")
    print("Type your message or a command (/help, /list, /clear, /bye).")
    # print("System Prompt:")
    # print(session.history[0]["content"]) # Maybe too verbose for every start

    session_active = True
    while session_active:
        try:
            prompt = input("\n>>> ")
            prompt_strip = prompt.strip()

            if not prompt_strip:
                continue # Ignore empty input

            # --- Handle Commands ---
            if prompt_strip.lower() == "/bye":
                print("Goodbye! Exiting chat session.")
                session_active = False
            elif prompt_strip.lower() == "/list":
                session.list_files()
            elif prompt_strip.lower() == "/clear":
                session.clear_history(keep_files=True) # Default to keeping files
                print("Chat history cleared (file context retained).")
            elif prompt_strip.lower() == "/clear all":
                 session.clear_history(keep_files=False)
                 print("Chat history cleared completely.")
            elif prompt_strip.lower() == "/help":
                 print("Available commands:")
                 print("  /list      - List files loaded in the session.")
                 print("  /clear     - Clear chat history (keeps file context).")
                 print("  /clear all - Clear chat history and file context.")
                 print("  /help      - Show this help message.")
                 print("  /bye       - Exit the interactive chat.")
                 print("Any other input is treated as a chat message.")
            # Add other commands like /config, /reload, etc. here if needed

            # --- Process as Chat ---
            else:
                if not session.chat(prompt, helper_functions, hide_reasoning=hide_reasoning):
                    print("An error occurred during chat processing. Please check logs.")
                    # Decide if error should terminate session or allow user to continue

        except (KeyboardInterrupt, EOFError):
            print("\nInterrupted. Exiting chat session.")
            session_active = False
        except Exception as e:
            logger.error(f"Unexpected error in interactive chat loop: {e}")
            logger.error(traceback.format_exc())
            print(f"\nAn unexpected error occurred: {e}")
            # Decide whether to break or continue
            # session_active = False # Option: exit on unexpected errors


# --- Argument Parsing ---
def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Qwen CLI - A command-line interface for code-aware conversations with Qwen models.",
        formatter_class=argparse.RawTextHelpFormatter # Preserve formatting in help messages
    )

    # Configuration overrides
    parser.add_argument(
        "--model-dir", "-m",
        help="Override the directory where models are stored/loaded from.\n(Default: Constructed from $MODELS_DIR or /models and model_repo)"
    )
    parser.add_argument(
        "--config-dir", "-c",
        help="Override the directory for loading qwen_cli config.json.\n(Default: $CONFIG_DIR or ~/.config/qwen_cli)"
    )
    parser.add_argument(
        "--helpers-dir",
        help="Override the directory containing helper function modules.\n(Default: specified in config or ./helper_functions)"
    )

    # Behavior flags
    parser.add_argument(
        "--hide-reasoning",
        action="store_true",
        help="Attempt to hide model's reasoning steps (effectiveness may vary)." # This flag seems less relevant now with tool handling
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output from the model."
    )

    # Positional arguments for commands or direct chat
    parser.add_argument(
        "command",
        nargs="?", # Optional: if not provided, start interactive chat
        help=(
            "Optional command to execute immediately, or the start of a chat prompt.\n"
            "Available commands:\n"
            "  load <filepath>   - Load a file into the context (uses LOAD_FILE helper).\n"
            "  batch_load <dir> <pattern> - Load multiple files (uses BATCH_LOAD helper).\n"
            "  help              - Show available commands (non-interactive).\n"
            "If no command is given, starts an interactive chat session.\n"
            "If the input doesn't match a known command, it's treated as a chat prompt."
        )
    )
    parser.add_argument(
        "args",
        nargs="*", # Zero or more arguments for the command or chat prompt
        help="Arguments for the command, or the rest of the chat prompt."
    )

    return parser.parse_args()

# --- Configuration Loading (Application Level) ---
def load_app_config(args: argparse.Namespace) -> dict:
    """Loads configuration, applying command-line overrides."""
    # Load base config respecting CONFIG_DIR and MODELS_DIR env vars / args
    config = load_config(config_dir_override=args.config_dir, model_dir_override=args.model_dir)

    # Apply command-line override for helpers_dir *after* loading config
    if args.helpers_dir:
        config["helpers_dir"] = args.helpers_dir
        logger.info(f"Using helpers directory from command line: {args.helpers_dir}")
    elif "helpers_dir" not in config:
         # Fallback if not in config file either
         config["helpers_dir"] = DEFAULT_CONFIG["helpers_dir"]
         logger.info(f"Using default helpers directory: {config['helpers_dir']}")

    # Ensure helpers_dir is resolved to an absolute path
    config["helpers_dir"] = str(Path(config["helpers_dir"]).resolve())


    # You could add more config validation here if needed
    logger.debug(f"Final configuration: {json.dumps(config, indent=2)}")
    return config


# --- Command Handling ---
def handle_command(args: argparse.Namespace, session: QwenSession, helper_functions: Dict[str, Callable], config: dict):
    """Dispatches execution based on the parsed command-line arguments."""

    command = args.command.lower() if args.command else None
    cmd_args_str = " ".join(args.args)
    stream = not args.no_stream

    # --- No Command: Interactive Chat ---
    if command is None:
        interactive_chat(session, helper_functions, args.hide_reasoning)
        return

    # --- Specific Built-in Commands ---
    if command == "help":
        # Provide a more detailed help message for non-interactive use
        print("Qwen CLI Commands (Non-Interactive):")
        print("  qwen_cli load <filepath>          - Loads a single file using the LOAD_FILE helper.")
        print("  qwen_cli batch_load <dir> <glob>  - Loads multiple files using the BATCH_LOAD helper.")
        print("  qwen_cli <prompt...>            - Sends the prompt directly to the model for a single response.")
        print("  qwen_cli                        - Starts an interactive chat session.")
        print("\nUse '/help' within the interactive session for interactive commands.")
        print("\nConfiguration options (--model-dir, --config-dir, --helpers-dir) can precede commands.")

    # --- Commands Relying on Helper Functions ---
    elif command == "load":
        if "LOAD_FILE" in helper_functions:
            if not args.args:
                print("Usage: qwen_cli load <filepath>")
                return
            filepath = args.args[0] # Only take the first arg as filepath
            print(f"Executing LOAD_FILE helper for: {filepath}")
            result = helper_functions["LOAD_FILE"](filepath)
            # The helper should log success/failure. We might print the system message it returns.
            if result: print(result) # Print the system message returned by the helper
        else:
            print("Error: LOAD_FILE helper function is not available.")

    elif command == "batch_load":
        if "BATCH_LOAD" in helper_functions:
            if not args.args: # Need at least directory, pattern is optional in some implementations
                print("Usage: qwen_cli batch_load <directory> [<glob_pattern>]")
                return
            print(f"Executing BATCH_LOAD helper for: {cmd_args_str}")
            result = helper_functions["BATCH_LOAD"](cmd_args_str)
            if result: print(result)
        else:
            print("Error: BATCH_LOAD helper function is not available.")

    # --- Check if command matches any other loaded helper ---
    elif command.upper() in helper_functions:
         print(f"Executing {command.upper()} helper with args: {cmd_args_str}")
         result = helper_functions[command.upper()](cmd_args_str)
         if result: print(result)

    # --- Default: Treat as a single chat prompt ---
    else:
        prompt = " ".join([args.command] + args.args)
        print(f"Sending prompt to model (streaming: {stream}):\n---\n{prompt}\n---")
        session.chat(prompt, helper_functions, stream=stream, hide_reasoning=args.hide_reasoning)
        # The chat function handles printing the response (streaming or not)


# --- Main Execution ---
def main():
    """Main entry point for the Qwen CLI application."""
    args = parse_arguments()
    config = load_app_config(args)
    helper_functions, tool_prompts = load_helper_functions(config["helpers_dir"])

    # Initialize session (consider loading/saving sessions later)
    session = QwenSession(config, tool_prompts)

    # Load the model immediately - critical step
    if not session._ensure_model_loaded():
        print("\nCritical Error: Failed to load the AI model. Please check logs and configuration.")
        print(f"Model directory used: {config.get('model_dir')}")
        print(f"Cache directory used: {CACHE_DIR}")
        sys.exit(1) # Exit if model loading fails

    # Handle the command or start interactive chat
    handle_command(args, session, helper_functions, config)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
         # Catch sys.exit calls (e.g., from argparse help) for cleaner exit codes
         sys.exit(e.code)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user. Exiting.")
        sys.exit(130) # Standard exit code for Ctrl+C
    except Exception as e:
        logger.error(f"An unexpected critical error occurred: {e}", exc_info=True)
        # Optionally print a user-friendly message too
        print(f"\nAn unexpected error occurred: {e}. Please check the logs for details.", file=sys.stderr)
        sys.exit(1) # General error exit code
