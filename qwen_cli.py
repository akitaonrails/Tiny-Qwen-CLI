#!/usr/bin/env python3
"""
Qwen CLI - A simple command-line interface for interacting with Qwen models.
This tool allows loading source files into the context to have code-aware conversations.
Designed to work both locally and within a Docker container.
"""

import argparse
import importlib.util
import json
import logging
import os
import re
import readline  # For better command line input experience
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple

import transformers
from helper_functions.utils import get_language_from_extension

# Transformers & Torch
try:
    import torch
    from accelerate import Accelerator
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TextStreamer,
    )
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
def get_config_path() -> Path:
    config_dir = Path(os.environ.get("CONFIG_DIR", HOME_DIR / ".config" / "qwen_cli"))
    return config_dir / "config.json"

def load_config() -> dict:
    config = DEFAULT_CONFIG.copy()
    config_path = get_config_path()
    if config_path.exists():
        try:
            user_config = json.loads(config_path.read_text())
            config.update(user_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error decoding config file {config_path}: {e}. Using defaults.")
    else:
        get_config_path().parent.mkdir(parents=True, exist_ok=True)
        logger.info("No config.json found, using default settings.")
    return config

# --- Dynamic Helper Function Loading ---

def _ensure_helpers_dir(helpers_path: Path):
    """Ensures the helper directory and __init__.py exist."""
    if not helpers_path.exists():
        logger.info(f"Creating helper directory: {helpers_path}")
        helpers_path.mkdir(parents=True, exist_ok=True)
    init_file = helpers_path / "__init__.py"
    if not init_file.exists():
        logger.info(f"Creating {init_file}")
        init_file.touch()

def _load_module_from_path(py_file: Path) -> Optional[ModuleType]:
    """Loads a Python module dynamically from its file path."""
    module_name = py_file.stem
    if module_name.startswith("__"):
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        else:
            logger.warning(f"Could not create spec for module: {py_file}")
            return None
    except Exception as e:
        logger.error(f"Failed to load module {py_file}: {e}")
        return None

def _extract_helper_info(module: ModuleType) -> Optional[Tuple[str, Callable, str]]:
    """Extracts command name, function, and docstring from a helper module."""
    for attr_name in dir(module):
        if attr_name.startswith("handle_"):
            func = getattr(module, attr_name)
            if callable(func):
                command = attr_name[7:].upper()
                doc = func.__doc__ or ""
                return command, func, doc
    return None

def load_helper_functions(helpers_dir: str) -> Tuple[Dict[str, Callable], List[str]]:
    """
    Loads helper functions dynamically from Python files in the specified directory.

    Args:
        helpers_dir: The directory containing helper function modules.

    Returns:
        A tuple containing:
            - A dictionary mapping command names (e.g., "LOAD_FILE") to their handler functions.
            - A list of formatted tool prompts derived from function docstrings.
    """
    helpers: Dict[str, Callable] = {}
    tool_prompts: List[str] = []
    helpers_path = Path(helpers_dir)

    _ensure_helpers_dir(helpers_path)

    for py_file in helpers_path.glob("*.py"):
        module = _load_module_from_path(py_file)
        if module:
            helper_info = _extract_helper_info(module)
            if helper_info:
                command, func, doc = helper_info
                helpers[command] = func
                first_line = doc.strip().splitlines()[0] if doc.strip() else ""
                if first_line:
                    tool_prompts.append(f"[{command} args] – {first_line}")
                logger.debug(f"Loaded helper command: {command} from {py_file.name}")

    logger.info(f"Loaded {len(helpers)} helper functions from {helpers_dir}")
    return helpers, tool_prompts

# --- Parse Special Commands ---
def parse_special_commands(response: str) -> List[tuple]:
    pattern = r'\[([A-Z_]+)\s+([^\]]+)\]'
    return [(m.group(1), m.group(2).strip(), m.start(), m.end()) for m in re.finditer(pattern, response)]

# --- QwenSession Class ---
class QwenSession:
    _model = None
    _tokenizer = None
    _model_loading_lock = False

    def __init__(self, config: dict, tool_prompts: List[str]):
        self.config = config
        self.history = [{"role": "system", "content": build_system_prompt(tool_prompts)}]
        self.files_loaded = {}
        self.created_at = datetime.now().isoformat()
        self.last_used = self.created_at

    def to_dict(self) -> dict:
        return {
            'history': self.history,
            'files_loaded': self.files_loaded,
            'created_at': self.created_at,
            'last_used': self.last_used,
        }

    @classmethod
    def from_dict(cls, config: dict, tool_prompts: List[str], data: dict) -> 'QwenSession':
        session = cls(config, tool_prompts)
        session.history = data.get('history', [])
        session.files_loaded = data.get('files_loaded', {})
        session.created_at = data.get('created_at', session.created_at)
        session.last_used = data.get('last_used', session.last_used)
        return session

    def _ensure_model_loaded(self) -> bool: # Renamed for clarity
        """Ensures the model and tokenizer are loaded, loading if necessary."""
        if QwenSession._model and QwenSession._tokenizer:
            return True
        return self._load_model()

    def _load_model(self) -> bool:
        if QwenSession._model_loading_lock:
            logger.info("Model loading in progress by another session, waiting...")
            while QwenSession._model_loading_lock:
                time.sleep(1)
            return bool(QwenSession._model and QwenSession._tokenizer)

        QwenSession._model_loading_lock = True
        logger.info("Loading model and tokenizer...")
        try:
            self._initialize_model()
            logger.info("Model and tokenizer loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {e}")
            logger.error(traceback.format_exc()) # Log full traceback
            return False
        finally:
            QwenSession._model_loading_lock = False

    def _initialize_model(self):
        model_repo = self.config.get("model_repo", DEFAULT_CONFIG["model_repo"])
        model_dir = Path(self.config.get("model_dir", DEFAULT_CONFIG["model_dir"]))
        quantization = self.config.get("quantization", DEFAULT_CONFIG["quantization"]).lower()
        download_timeout = self.config.get("model_download_timeout", DEFAULT_CONFIG["model_download_timeout"])
        if not model_dir.exists():
            logger.info(f"Model directory {model_dir} not found. Cloning from {model_repo}...")
            model_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.run(["git", "clone", "--depth", "1", f"https://huggingface.co/{model_repo}", str(model_dir)], check=True, timeout=download_timeout)
                logger.info(f"Model cloned successfully to {model_dir}")
            except subprocess.TimeoutExpired:
                 logger.error(f"Timeout ({download_timeout}s) exceeded while cloning model.")
                 raise
            except subprocess.CalledProcessError as e:
                 logger.error(f"Git clone failed: {e}")
                 raise
            except Exception as e:
                 logger.error(f"An unexpected error occurred during model download: {e}")
                 raise

        logger.info(f"Loading tokenizer from {model_dir}...")
        QwenSession._tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, cache_dir=str(CACHE_DIR))
        logger.info("Tokenizer loaded.")

        model_kwargs = {"trust_remote_code": True, "device_map": "auto", "cache_dir": str(CACHE_DIR)}

        logger.info(f"Loading model config from {model_dir}...")
        model_config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
        logger.info("Model config loaded.")

        # Attention Optimization Logic
        attention_impl = "eager" # Default
        try:
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                 # Check if the model's attention class is SDPA compatible (heuristic)
                 # This requires knowing the specific attention class name for the Qwen model
                 # Example: transformers.models.qwen2.modeling_qwen2.Qwen2SdpaAttention
                 # If the model *already* uses an SDPA-specific class, setting attention_implementation might not be needed
                 # or could even cause conflicts. Let's assume we want to *try* flash_attention_2 if SDPA is available.
                 attention_impl = "sdpa" # Prefer SDPA if available
                 logger.info("PyTorch SDPA is available.")
                 # Try setting flash_attention_2 if SDPA is available, as it's often faster
                 try:
                     # Test if flash_attn is importable, indicating it's likely installed
                     import flash_attn
                     model_config.attention_implementation = "flash_attention_2"
                     attention_impl = "flash_attention_2"
                     logger.info("Setting attention implementation to 'flash_attention_2'.")
                 except ImportError:
                     logger.info("flash_attn not found, using default SDPA implementation.")
                     model_config.attention_implementation = "sdpa" # Explicitly set SDPA
                 except Exception as e:
                     logger.warning(f"Could not set flash_attention_2: {e}. Using default SDPA.")
                     model_config.attention_implementation = "sdpa"
            else:
                 logger.info("PyTorch SDPA not available. Checking for xFormers (FlashAttention).")
                 # Fallback to xformers/flash_attention if SDPA isn't available in PyTorch
                 try:
                     import xformers.ops
                     model_config.attention_implementation = "flash_attention_2" # xFormers provides FlashAttention
                     attention_impl = "flash_attention_2 (xFormers)"
                     logger.info("xFormers found. Setting attention implementation to 'flash_attention_2'.")
                 except ImportError:
                     logger.info("xFormers not found. Using default eager attention.")
                 except Exception as e:
                     logger.warning(f"Error enabling xFormers: {e}. Using default eager attention.")
        except Exception as e:
            logger.warning(f"Error during attention optimization detection: {e}. Using default eager attention.")

        logger.info(f"Selected attention implementation: {attention_impl}")

        # Quantization Logic
        if quantization == "4bit" and torch.cuda.is_available():
            logger.info("Applying 4-bit quantization.")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            model_kwargs["torch_dtype"] = torch.bfloat16 # Often recommended with 4bit
        elif quantization == "8bit" and torch.cuda.is_available():
            logger.info("Applying 8-bit quantization.")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            if quantization in ["4bit", "8bit"] and not torch.cuda.is_available():
                logger.warning(f"{quantization} quantization requires CUDA. Falling back to default dtype.")
            logger.info("Using default data type (no quantization or CPU).")
            model_kwargs["torch_dtype"] = "auto" # Let transformers decide based on availability

        logger.info(f"Loading model '{model_repo}' from {model_dir} with kwargs: { {k: v for k, v in model_kwargs.items() if k != 'quantization_config'} }...") # Avoid printing large config object
        QwenSession._model = AutoModelForCausalLM.from_pretrained(str(model_dir), config=model_config, **model_kwargs)
        logger.info("Model loaded.")

        # Verify attention implementation after loading
        final_attn_impl = getattr(QwenSession._model.config, "_attn_implementation", "unknown")
        logger.info(f"Model loaded with attention implementation: {final_attn_impl}")


    def _trim_history(self, max_tokens: int):
        # TODO: Implement history trimming based on token count
        # This is crucial for long conversations to avoid exceeding context limits.
        # Need to tokenize history messages and remove oldest user/assistant pairs
        # while preserving the system prompt and any file context messages.
        pass

    def chat(self, prompt: str, helper_functions: Dict[str, Callable], max_new_tokens=None, temperature=None, stream=True, hide_reasoning=False) -> bool:
        self.last_used = datetime.now().isoformat()
        if not self._ensure_model_loaded(): # Ensure model is loaded before chat
             print("Model is not loaded. Cannot proceed with chat.")
             return False

        if max_new_tokens is None:
            max_new_tokens = self.config.get("max_new_tokens", DEFAULT_CONFIG["max_new_tokens"])
        if temperature is None:
            temperature = self.config.get("temperature", DEFAULT_CONFIG["temperature"])
        self._trim_history(self.config.get("max_context_tokens", DEFAULT_CONFIG["max_context_tokens"]))

        self.history.append({"role": "user", "content": prompt})
        try:
            logger.debug(f"Generating response for prompt: '{prompt[:100]}...'")
            formatted_text = QwenSession._tokenizer.apply_chat_template(
                self.history, tokenize=False, add_generation_prompt=True
            )
            inputs = QwenSession._tokenizer([formatted_text], return_tensors="pt")
            inputs = {k: v.to(QwenSession._model.device) for k, v in inputs.items()}
            streamer = TextStreamer(QwenSession._tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None

            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "temperature": temperature if temperature > 0 else 1.0, # Temp must be > 0 for sampling
                "repetition_penalty": 1.1,
                "streamer": streamer,
            }
            logger.debug(f"Generation kwargs: {generation_kwargs}")

            out_ids = QwenSession._model.generate(**inputs, **generation_kwargs)
            generated = out_ids[0][inputs["input_ids"].shape[1]:]
            response_text = QwenSession._tokenizer.decode(generated, skip_special_tokens=True).strip()
            logger.debug(f"Raw response: '{response_text[:200]}...'")

            # Handle special commands
            commands_executed = False # Flag to track if a known command was processed
            commands = parse_special_commands(response_text)
            if commands:
                logger.info(f"Detected commands: {commands}")
                # We might need to handle multiple commands or decide how to proceed
                # For now, handle the first one found if it's a helper
                cmd_type, cmd_arg, _, _ = commands[0] # Process first command
                if cmd_type in helper_functions:
                    commands_executed = True # Mark that a known command was attempted
                    logger.info(f"Executing helper command: [{cmd_type} {cmd_arg}]")
                    try:
                        result = helper_functions[cmd_type](cmd_arg)
                        if result: # Assume result is system message content
                             self.history.append({"role": "system", "content": result})
                             # Decide if we need to re-prompt or just continue
                             # Option 1: Re-prompt implicitly (might be confusing)
                             # self.history.append({"role": "user", "content": "Please continue the analysis using the loaded file."}) # Implicit re-prompt
                             # Option 2: Just add result and let the next user input drive conversation
                             print(f"\n✅ [{cmd_type}] processed '{cmd_arg}'. Result added to context.")
                             # Don't automatically re-run chat here, let the user decide the next step
                        else:
                             logger.warning(f"Helper command [{cmd_type}] returned no result.")
                             # Inform user? Maybe add a system message?
                             self.history.append({"role": "system", "content": f"Note: Tool [{cmd_type} {cmd_arg}] executed but returned no output."})

                    except Exception as e:
                        logger.error(f"Error executing helper command [{cmd_type} {cmd_arg}]: {e}")
                        logger.error(traceback.format_exc())
                        # Inform the model/user about the failure
                        self.history.append({"role": "system", "content": f"Error: Failed to execute tool [{cmd_type} {cmd_arg}]. Reason: {e}"})
                # else: # Optional: Handle case where command is detected but not a known helper
                #    logger.warning(f"Detected command [{cmd_type}] is not a known helper function.")

            # Add assistant response to history ONLY if NO known command was executed.
            # The system message added during command processing is the intended history item.
            if not commands_executed:
                 if not stream and response_text: # Print non-streamed response if not empty and no command ran
                     print(response_text)
                 if response_text: # Add non-empty response to history if no command ran
                     self.history.append({"role": "assistant", "content": response_text})
            # If a command *was* executed AND we are not streaming, print the raw response
            # (containing the command) for context, even though it's not added to history.
            elif not stream and response_text:
                 print(response_text)


            return True # Indicate successful chat turn (even if helper failed)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.error(traceback.format_exc())
            # Attempt to remove the failed user prompt from history
            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()
            return False

    def list_files(self):
        if not self.files_loaded:
            print("No files loaded in this session.")
        else:
            # Assuming self.name was intended somewhere? Using session info instead.
            print(f"Files loaded in session created at {self.created_at}:")
            for i, (filepath, meta) in enumerate(self.files_loaded.items(), 1):
                 # Assuming meta might contain more info later, for now just path
                 print(f"  {i}. {filepath}") # TODO: Enhance with meta if available

    def clear_history(self, keep_files=True):
        if not self.history: return False # Should not happen if initialized correctly

        system_prompt_content = "System prompt not found." # Default
        if self.history[0]["role"] == "system":
            system_prompt_content = self.history[0]["content"]

        file_msgs = []
        if keep_files:
            # A more robust way to identify file messages might be needed
            # Relying on "[file:" prefix might be brittle.
            # Maybe helpers should return a specific structure?
            for msg in self.history:
                # Let's assume any system message NOT the initial prompt is file-related for now
                if msg.get("role") == "system" and msg.get("content") != system_prompt_content:
                    file_msgs.append(msg)

        self.history = [{"role": "system", "content": system_prompt_content}] + file_msgs
        logger.info("Conversation history cleared (keeping file context if enabled)")
        return True

# --- Interactive Chat ---
def interactive_chat(session, helper_functions, hide_reasoning=False):
    print(f"\nInteractive chat session started. Type 'bye' or press Ctrl+D to exit.\n")
    print("--- SYSTEM PROMPT ---")
    print(session.history[0]["content"])
    print("---------------------\n")

    original_sigint_handler = signal.getsignal(signal.SIGINT)
    def handle_exit_signal(sig, frame):
        print("\nExit signal received...")
        handle_exit()

    def handle_exit():
        print("\nExiting chat session. Goodbye!\n")
        # Restore original SIGINT handler before exiting if needed, though exit() might suffice
        signal.signal(signal.SIGINT, original_sigint_handler)
        sys.exit(0) # Cleanly exit the program

    # Register handlers for graceful exit
    signal.signal(signal.SIGINT, handle_exit_signal) # Ctrl+C
    # Handling Ctrl+D (EOFError) is done in the input loop

    while True:
        try:
            prompt = input("\n>>> ")
            prompt = prompt.strip() # Remove leading/trailing whitespace

            if prompt.lower() == "bye":
                handle_exit()
                break # Should not be reached due to handle_exit()

            if not prompt:
                # print("Ignoring empty input.") # Optional: uncomment to notify user
                continue # Skip empty input

            # Execute chat turn
            session.chat(prompt, helper_functions, stream=True, hide_reasoning=hide_reasoning) # Assume streaming for interactive

        except EOFError: # Catch Ctrl+D
            print("\nEOF (Ctrl+D) detected.")
            handle_exit()
            break
        except KeyboardInterrupt: # Catch Ctrl+C (redundant with signal handler, but safe)
             print("\nInterrupted by user (Ctrl+C). Type 'bye' or Ctrl+D to exit.")
             # Continue the loop after interruption
        except Exception as e:
            logger.error(f"Error in interactive chat loop: {e}")
            logger.error(traceback.format_exc())
            print(f"\nAn unexpected error occurred: {e}")
            # Decide whether to continue or exit on unexpected errors
            # For robustness, let's try to continue
            # handle_exit() # Uncomment to exit on any error

# --- Main CLI ---
def main():
    parser = argparse.ArgumentParser(
        description="Qwen CLI - Code-aware conversation tool",
        formatter_class=argparse.RawTextHelpFormatter # Keep formatting in help
    )
    # Configuration overrides
    parser.add_argument("--model-dir", "-m", help="Override model directory path.")
    parser.add_argument("--config-dir", "-c", help="Override configuration directory path (contains config.json).")
    parser.add_argument("--helpers-dir", help="Override helper functions directory path.")

    # Behavior flags
    parser.add_argument("--hide-reasoning", action="store_true", help="Attempt to hide model's internal reasoning/tool usage (experimental).")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output for non-interactive mode.")

    # Positional arguments for commands or direct prompts
    parser.add_argument("cmd", nargs="?", help=(
        "Optional command or the start of a direct chat prompt.\n"
        "Available commands:\n"
        "  new          : Start a new interactive chat session (default if no command).\n"
        "  list         : List files loaded in the current session context (TBD).\n"
        "  clear        : Clear conversation history (keeps loaded files).\n"
        "  help         : Show this help message (or use -h/--help).\n"
        "If 'cmd' is not one of these, it's treated as the start of a chat prompt."
    ))
    parser.add_argument("args", nargs="*", help="Arguments for commands, or the rest of the chat prompt.")

    args = parser.parse_args()

    # Apply environment variable overrides from CLI args
    if args.config_dir:
        os.environ["CONFIG_DIR"] = args.config_dir
        logger.info(f"Using config directory override: {args.config_dir}")
    if args.model_dir:
        os.environ["MODELS_DIR"] = args.model_dir
        logger.info(f"Using model directory override: {args.model_dir}")

    # Load configuration
    config = load_config()
    if args.helpers_dir:
        config["helpers_dir"] = args.helpers_dir # Override helpers_dir from args if provided
        logger.info(f"Using helpers directory override: {config['helpers_dir']}")

    # Load helper functions based on final config
    helper_functions, tool_prompts = load_helper_functions(config["helpers_dir"])

    # Initialize session (model loading is deferred until first use)
    session = QwenSession(config, tool_prompts)

    # --- Command Handling ---
    command = args.cmd.lower() if args.cmd else None
    command_args_str = " ".join(args.args)

    # Define command map (using lambdas for deferred execution)
    # Note: Model loading happens inside chat() or potentially specific commands if needed
    command_map = {
        "new": lambda: interactive_chat(session, helper_functions, args.hide_reasoning),
        "list": lambda: session.list_files(), # TODO: Implement session loading/persistence for this to be useful across runs
        "clear": lambda: session.clear_history(), # TODO: Same as above
        "help": lambda: parser.print_help(),
        # Add direct helper function calls if desired (e.g., for testing)
        # Example: Allow running 'qwen_cli load_file path/to/file' directly
        **{cmd.lower(): (lambda func=func, arg_str=command_args_str: print(func(arg_str) or f"{cmd} returned no output."))
           for cmd, func in helper_functions.items()}
    }

    if command in command_map:
        logger.info(f"Executing command: {command}")
        command_map[command]()
    elif command is None:
        # Default action: start interactive chat
        logger.info("No command specified, starting interactive chat.")
        interactive_chat(session, helper_functions, args.hide_reasoning)
    else:
        # Treat as a direct prompt (non-interactive)
        prompt = " ".join([args.cmd] + args.args)
        logger.info(f"Executing direct prompt (non-interactive): '{prompt[:100]}...'")
        # Ensure model is loaded before non-interactive chat
        if not session._ensure_model_loaded():
             print("Failed to load model. Cannot execute prompt.", file=sys.stderr)
             sys.exit(1)
        session.chat(prompt, helper_functions, stream=(not args.no_stream), hide_reasoning=args.hide_reasoning)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user (main). Exiting.")
        sys.exit(0) # Exit gracefully on Ctrl+C in main
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}", exc_info=True) # Log full traceback
        # logger.error(traceback.format_exc()) # Redundant if exc_info=True
        sys.exit(1) # Exit with error status
