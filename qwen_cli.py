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
from typing import List, Dict, Optional, Any, Callable
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
def load_config() -> dict:
    config = DEFAULT_CONFIG.copy()
    config_dir = Path(os.environ.get("CONFIG_DIR", HOME_DIR / ".config" / "qwen_cli"))
    config_path = config_dir / "config.json"
    if config_path.exists():
        try:
            user_config = json.loads(config_path.read_text())
            config.update(user_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error decoding config file {config_path}: {e}. Using defaults.")
    else:
        config_dir.mkdir(parents=True, exist_ok=True)
        logger.info("No config.json found, using default settings.")
    return config

# --- Dynamic Helper Function Loading ---
def load_helper_functions(helpers_dir: str) -> (Dict[str, Callable], List[str]):
    helpers = {}
    tool_prompts = []
    helpers_path = Path(helpers_dir)
    if not helpers_path.exists():
        helpers_path.mkdir(parents=True, exist_ok=True)
        (helpers_path / "__init__.py").touch()
    for py_file in helpers_path.glob("*.py"):
        module_name = py_file.stem
        if module_name.startswith("__"): continue
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            logger.error(f"Error loading helper module {module_name}: {e}")
            continue
        for attr_name in dir(module):
            if attr_name.startswith("handle_"):
                func = getattr(module, attr_name)
                command = attr_name[7:].upper()
                helpers[command] = func
                doc = func.__doc__ or ""
                first_line = doc.strip().splitlines()[0] if doc.strip() else ""
                if first_line:
                    tool_prompts.append(f"[{command} args] – {first_line}")
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


    # this next method is super long and convoluted and could use refactoring to better separate responsabilities
    def _ensure_model_loaded(self) -> bool:
        """Ensure the model and tokenizer are loaded, handles concurrent initialization"""
        if QwenSession._model and QwenSession._tokenizer:
            return True
        if QwenSession._model_loading_lock:
            logger.info("Model loading in progress by another session, waiting...")
            while QwenSession._model_loading_lock:
                time.sleep(1)
            return bool(QwenSession._model and QwenSession._tokenizer)

        QwenSession._model_loading_lock = True
        try:
            self._download_model_if_needed()
            self._load_tokenizer()
            model_config = self._configure_attention_optimization()
            model_kwargs = self._setup_quantization_config() 
            self._load_model_components(model_config, model_kwargs)
            self._log_final_attention_status()
            return True
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {e}")
            logger.error(traceback.format_exc())
            QwenSession._model_loading_lock = False
            return False

    def _download_model_if_needed(self) -> None:
        """Download model repository if not present locally"""
        model_dir = Path(self.config.get("model_dir", DEFAULT_CONFIG["model_dir"]))
        if not model_dir.exists():
            model_repo = self.config.get("model_repo", DEFAULT_CONFIG["model_repo"])
            download_timeout = self.config.get("model_download_timeout", DEFAULT_CONFIG["model_download_timeout"])
            model_dir.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading model from {model_repo}...")
            subprocess.run(
                ["git", "clone", "--depth", "1", f"https://huggingface.co/{model_repo}", str(model_dir)],
                check=False,
                timeout=download_timeout
            )

    def _load_tokenizer(self) -> None:
        """Initialize the tokenizer from pretrained weights"""
        model_dir = Path(self.config.get("model_dir", DEFAULT_CONFIG["model_dir"]))
        QwenSession._tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            cache_dir=str(CACHE_DIR)
        )

    def _configure_attention_optimization(self) -> AutoConfig:
        """Configure optimal attention implementation based on hardware and dependencies"""
        model_dir = Path(self.config.get("model_dir", DEFAULT_CONFIG["model_dir"]))
        model_config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
        sdpa_available = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        try:
            if sdpa_available and self._check_sdpa_compatibility():
                print("Qwen attention is SDPA-compatible and enabled!")
            else:
                self._try_configure_xformers(model_config)
        except Exception as e:
            logger.warning(f"Attention optimization error: {e}. Using default attention.")

        return model_config

    def _check_sdpa_compatibility(self) -> bool:
        """Check if model supports PyTorch's SDPA attention"""
        # Simple check since model may not be loaded yet - actual validation is runtime
        return True  # Assume compatible for initial configuration

    def _try_configure_xformers(self, model_config: AutoConfig) -> None:
        """Attempt to configure xFormers if available"""
        try:
            import xformers.ops  # noqa: F401
            model_config.attention_implementation = "flash_attention_2"
            print("xFormers enabled for attention optimization")
        except ImportError:
            print("xFormers not available - using default attention")
        except Exception as e:
            logger.warning(f"Error configuring xFormers: {e}")

    def _setup_quantization_config(self) -> Dict[str, Any]:
        """Configure quantization settings based on user configuration and hardware"""
        quantization = self.config.get("quantization", DEFAULT_CONFIG["quantization"]).lower()
        model_kwargs = {"trust_remote_code": True, "device_map": "auto", "cache_dir": str(CACHE_DIR)}

        if quantization == "4bit" and torch.cuda.is_available():
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif quantization == "8bit" and torch.cuda.is_available():
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["torch_dtype"] = "auto"

        return model_kwargs

    def _load_model_components(self, model_config: AutoConfig, model_kwargs: Dict[str, Any]) -> None:
        """Load the main model with prepared configuration and parameters"""
        model_dir = Path(self.config.get("model_dir", DEFAULT_CONFIG["model_dir"]))
        QwenSession._model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            config=model_config, 
            **model_kwargs
        )
        QwenSession._model_loading_lock = False

    def _log_final_attention_status(self) -> None:
        """Log final attention implementation status after model load"""
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            print("SDPA is available and active!")
        else:
            print("Using default attention implementation")

    def _trim_history(self, max_tokens: int):
        # identical trimming logic as before
        pass

    def chat(self, prompt: str, helper_functions: Dict[str, Callable], max_new_tokens=None, temperature=None, stream=True, hide_reasoning=False) -> bool:
        """Main entry point for chat interactions, coordinates the workflow"""
        self._update_session_timestamp()
        self._setup_generation_parameters(max_new_tokens, temperature)
        self._trim_history(self.config.get("max_context_tokens", DEFAULT_CONFIG["max_context_tokens"]))
        self.history.append({"role": "user", "content": prompt})

        try:
            response_text = self._generate_response(stream, self.max_new_tokens, self.temperature)
            should_continue = self._process_helper_commands(response_text, helper_functions, prompt, 
                                                          max_new_tokens, temperature, stream, hide_reasoning)
            
            if not should_continue:
                self._handle_response(response_text)
            
            return True
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.error(traceback.format_exc())
            return False

    def _update_session_timestamp(self):
        """Update the last used timestamp for the session"""
        self.last_used = datetime.now().isoformat()

    def _setup_generation_parameters(self, max_new_tokens, temperature):
        """Set default generation parameters if not provided"""
        self.max_new_tokens = max_new_tokens or self.config.get("max_new_tokens", DEFAULT_CONFIG["max_new_tokens"])
        self.temperature = temperature or self.config.get("temperature", DEFAULT_CONFIG["temperature"])

    def _generate_response(self, stream: bool, max_new_tokens: int, temperature: float) -> str:
        """Generate model response with current context"""
        formatted_text = QwenSession._tokenizer.apply_chat_template(
            self.history, tokenize=False, add_generation_prompt=True
        )
        inputs = QwenSession._tokenizer([formatted_text], return_tensors="pt")
        inputs = {k: v.to(QwenSession._model.device) for k, v in inputs.items()}
        
        streamer = TextStreamer(QwenSession._tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None
        out_ids = QwenSession._model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
            temperature=temperature, streamer=streamer, repetition_penalty=1.1
        )
        
        generated = out_ids[0][inputs["input_ids"].shape[1]:]
        return QwenSession._tokenizer.decode(generated, skip_special_tokens=True)

    def _process_helper_commands(self, response_text: str, helper_functions: dict, prompt: str, 
                               max_new_tokens: int, temperature: float, stream: bool, hide_reasoning: bool) -> bool:
        """Handle helper commands found in the response text"""
        for cmd_type, cmd_arg, start, end in parse_special_commands(response_text):
            if cmd_type in helper_functions:
                result = helper_functions[cmd_type](cmd_arg)
                if result:
                    # Remove the command from the response text that will be preserved
                    cleaned_response = response_text[:start] + response_text[end:]
                    self.history.extend([
                        {"role": "system", "content": result},
                        {"role": "assistant", "content": cleaned_response.strip()},
                        {"role": "user", "content": "Please continue the analysis using the loaded file."}
                    ])
                    print(f"✅ [{cmd_type}] processed '{cmd_arg}'")
                    # Use the system-generated follow-up prompt instead of original
                    self.chat( 
                        "Please continue the analysis using the loaded file.",
                        helper_functions,
                        max_new_tokens,
                        temperature,
                        stream,
                        hide_reasoning
                    )
                    return True
        return False

    def _handle_response(self, response_text: str):
        """Finalize response handling and update history"""
        self.history.append({"role": "assistant", "content": response_text})

    def list_files(self):
        if not self.files_loaded:
            print("No files loaded in this session.")
        else:
            print(f"Files loaded in session '{self.name}' :")
            for i, (filepath, meta) in enumerate(self.files_loaded.items(), 1):
                print(f"  {i}. {filepath}")

    def clear_history(self, keep_files=True):
        system_prompt = self.history[0]["content"]
        file_msgs = []
        if keep_files:
            for msg in self.history:
                if msg.get("role") == "system" and msg.get("content", "").startswith("[file:"):
                    file_msgs.append(msg)
        self.history = [{"role": "system", "content": system_prompt}] + file_msgs
        logger.info("Conversation history cleared")
        return True

# --- Interactive Chat ---
def interactive_chat(session, helper_functions, hide_reasoning=False):
    print(f"\nInteractive chat session started. Type 'bye' to exit.\n")
    print("SYSTEM PROMPT:")
    print(session.history[0]["content"])

    def handle_exit():  # Define a function to handle exiting
        print("\nExiting chat session. Goodbye!\n")
        exit(0)  # Cleanly exit the program

    def handle_ctrl_d(sig, frame):  # Handler for Ctrl+D
        print("\nCtrl+D detected.")
        handle_exit()

    signal.signal(signal.SIGHUP, handle_ctrl_d)  # Register the handler (SIGHUP is sent by Ctrl+D)

    while True:
        try:
            prompt = input("\n>>> ")
            if prompt.strip().lower() == "bye":
                print("Goodbye! Exiting chat session.")
                break
            if not prompt.strip():  # Check for empty prompt
                print("Ignoring empty input.")
                continue  # Skip the rest of the loop and ask for input again
            session.chat(prompt, helper_functions, hide_reasoning=hide_reasoning)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Type 'bye' to exit properly.")
        except EOFError:  # Catch Ctrl+D directly (alternative method)
            print("\nEOF (Ctrl+D) detected. Exiting...")
            handle_exit()
            break
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            logger.error(traceback.format_exc())
            print(f"An error occurred: {e}")

# --- Main CLI ---
def main():
    """Main entry point for the Qwen CLI application"""
    parser = argparse.ArgumentParser(description="Qwen CLI - Code-aware conversation tool")
    parser.add_argument("--model-dir","-m")
    parser.add_argument("--config-dir","-c")
    parser.add_argument("--helpers-dir")
    parser.add_argument("--hide-reasoning", action="store_true")
    parser.add_argument("cmd", nargs="?", help="Command or chat prompt")
    parser.add_argument("args", nargs="*", help="Arguments for commands or prompt")
    args = parser.parse_args()

    # Configure environment from command line arguments
    setup_environment(args)

    config = load_config()
    if args.helpers_dir:
        config["helpers_dir"] = args.helpers_dir

    helper_functions, tool_prompts = load_helper_functions(config["helpers_dir"])
    session = QwenSession(config, tool_prompts)

    # Load the model immediately after session initialization
    if not session._ensure_model_loaded():
        print("Failed to load model. Exiting.")
        return

    # Route command to appropriate handler
    command_handlers = {
        "new": lambda: handle_new_command(config, tool_prompts, helper_functions, args.hide_reasoning),
        "batch_load": lambda: handle_batch_load(helper_functions, args.args),
        "load": lambda: handle_load_file(helper_functions, args.args),
        "help": print_help,
        None: lambda: interactive_chat(session, helper_functions, args.hide_reasoning)
    }
    
    if args.cmd in command_handlers:
        command_handlers[args.cmd]()
    else:
        handle_generic_command(args, session, helper_functions, args.hide_reasoning)

def setup_environment(args):
    """Configure environment variables from command line arguments"""
    if args.config_dir:
        os.environ["CONFIG_DIR"] = args.config_dir
    if args.model_dir:
        os.environ["MODELS_DIR"] = args.model_dir

def handle_new_command(config, tool_prompts, helper_functions, hide_reasoning):
    """Handle new session creation"""
    new_session = QwenSession(config, tool_prompts)
    interactive_chat(new_session, helper_functions, hide_reasoning)

def handle_batch_load(helper_functions, args):
    """Handle batch load command"""
    if args:
        result = helper_functions["BATCH_LOAD"](" ".join(args))
        print(result or "No results from batch load")
    else:
        print("Usage: batch_load <directory> <pattern>")

def handle_load_file(helper_functions, args):
    """Handle single file load command"""
    if args:
        result = helper_functions["LOAD_FILE"](args[0])
        print(result or "File content unavailable")
    else:
        print("Usage: load <filepath>")

def print_help():
    """Display help information"""
    print("Available commands:\nnew, batch_load, load, list, clear, help")

def handle_generic_command(args, session, helper_functions, hide_reasoning):
    """Handle unrecognized commands and generic chat input"""
    lc = args.cmd.upper()
    if lc in helper_functions:
        result = helper_functions[lc](' '.join(args.args))
        print(result or f"{lc} returned no output.")
    else:
        session.chat(" ".join([args.cmd]+args.args), helper_functions, hide_reasoning)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
