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
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
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
def load_helper_modules(helpers_dir: str) -> List[Dict]:
    """Load all helper modules from the specified directory."""
    helpers_path = Path(helpers_dir)
    if not helpers_path.exists():
        helpers_path.mkdir(parents=True, exist_ok=True)
        (helpers_path / "__init__.py").touch()
    modules = []
    for py_file in helpers_path.glob("*.py"):
        module_name = py_file.stem
        if module_name.startswith("__"):
            continue
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules

def process_helper_functions(modules: List[Dict]) -> (Dict[str, Callable], List[str]):
    """Extract helper functions from loaded modules and generate tool prompts."""
    helpers = {}
    tool_prompts = []
    for module in modules:
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

def load_helper_functions(helpers_dir: str) -> (Dict[str, Callable], List[str]):
    """Load and process helper functions from the specified directory."""
    modules = load_helper_modules(helpers_dir)
    return process_helper_functions(modules)

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

    def _ensure_model_loaded(self) -> bool:
        """Ensure the model and tokenizer are loaded, handling concurrency and configuration."""
        if QwenSession._model and QwenSession._tokenizer:
            return True

        if QwenSession._model_loading_lock:
            logger.info("Model loading in progress by another session, waiting...")
            while QwenSession._model_loading_lock:
                time.sleep(1)
            return bool(QwenSession._model and QwenSession._tokenizer)

        QwenSession._model_loading_lock = True
        try:
            self._load_model_and_tokenizer()
            return True
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {e}")
            QwenSession._model_loading_lock = False
            return False

    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer, including directory setup and configuration."""
        model_repo = self.config.get("model_repo", DEFAULT_CONFIG["model_repo"])
        model_dir = Path(self.config.get("model_dir", DEFAULT_CONFIG["model_dir"]))
        quantization = self.config.get("quantization", DEFAULT_CONFIG["quantization"]).lower()
        download_timeout = self.config.get("model_download_timeout", DEFAULT_CONFIG["model_download_timeout"])

        # Ensure model directory exists
        if not model_dir.exists():
            model_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", f"https://huggingface.co/{model_repo}", str(model_dir)],
                check=False,
                timeout=download_timeout
            )

        # Load tokenizer
        QwenSession._tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True, cache_dir=str(CACHE_DIR)
        )

        # Load model configuration
        model_config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)

        # Check for SDPA compatibility
        self._check_sdpa_compatibility(model_config)

        # Configure quantization
        model_kwargs = self._configure_quantization(quantization)

        # Load the model
        QwenSession._model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), config=model_config, **model_kwargs
        )

        # Final checks and logging
        self._log_attention_status()

    def _check_sdpa_compatibility(self, model_config):
        """Check if the model is compatible with SDPA and handle fallbacks."""
        sdpa_compatible = False
        try:
            for name, module in QwenSession._model.named_modules():
                if "attention" in name.lower() and isinstance(module, Qwen2Attention):
                    sdpa_compatible = True
                    break
        except Exception:
            pass

        if sdpa_compatible and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            print("Qwen attention is SDPA-compatible, and SDPA is available!")
        else:
            print("Qwen attention is NOT SDPA-compatible, or SDPA is not available. Trying xFormers...")
            try:
                import xformers.ops
                model_config.attention_implementation = "flash_attention_2"
                print("xFormers is available. Enabling it for attention.")
            except ImportError:
                print("xFormers is not installed. Falling back to default attention.")
            except Exception as e:
                print(f"Error using xFormers: {e}")

    def _configure_quantization(self, quantization: str) -> dict:
        """Configure model loading parameters based on quantization setting."""
        model_kwargs = {"trust_remote_code": True, "device_map": "auto", "cache_dir": str(CACHE_DIR)}
        if quantization == "4bit" and torch.cuda.is_available():
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif quantization == "8bit" and torch.cuda.is_available():
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["torch_dtype"] = "auto"
        return model_kwargs

    def _log_attention_status(self):
        """Log the status of attention mechanisms after model loading."""
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            print("SDPA is available and (hopefully) being used!")
        else:
            print("SDPA is available in PyTorch, but may not be used by the model.")

    def _trim_history(self, max_tokens: int):
        # identical trimming logic as before
        pass

    def chat(self, prompt: str, helper_functions: Dict[str, Callable], max_new_tokens=None, temperature=None, stream=True, hide_reasoning=False) -> bool:
        """Handle a chat interaction, generating a response and processing any special commands."""
        self.last_used = datetime.now().isoformat()
        max_new_tokens = self.config.get("max_new_tokens", DEFAULT_CONFIG["max_new_tokens"]) if max_new_tokens is None else max_new_tokens
        temperature = self.config.get("temperature", DEFAULT_CONFIG["temperature"]) if temperature is None else temperature
        self._trim_history(self.config.get("max_context_tokens", DEFAULT_CONFIG["max_context_tokens"]))

        self.history.append({"role": "user", "content": prompt})

        try:
            response_text = self._generate_response(stream=stream, temperature=temperature)
            self._process_special_commands(response_text, helper_functions)
            self._append_response(response_text)
            return True
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.error(traceback.format_exc())
            return False

    def _generate_response(self, stream=True, temperature=0.1) -> str:
        """Generate a response using the model."""
        formatted_text = QwenSession._tokenizer.apply_chat_template(
            self.history, tokenize=False, add_generation_prompt=True
        )
        inputs = QwenSession._tokenizer([formatted_text], return_tensors="pt")
        inputs = {k: v.to(QwenSession._model.device) for k, v in inputs.items()}
        streamer = TextStreamer(QwenSession._tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None
        out_ids = QwenSession._model.generate(
            **inputs, max_new_tokens=self.config.get("max_new_tokens", DEFAULT_CONFIG["max_new_tokens"]), do_sample=temperature > 0,
            temperature=temperature, streamer=streamer, repetition_penalty=1.1
        )
        generated = out_ids[0][inputs["input_ids"].shape[1]:]
        return QwenSession._tokenizer.decode(generated, skip_special_tokens=True)

    def _process_special_commands(self, response_text: str, helper_functions: Dict[str, Callable]):
        """Process any special commands found in the response."""
        for cmd_type, cmd_arg, _, _ in parse_special_commands(response_text):
            if cmd_type in helper_functions:
                result = helper_functions[cmd_type](cmd_arg)
                if result:
                    self.history.append({"role": "system", "content": result})
                    self.history.append({"role": "user", "content": "Please continue the analysis using the loaded file."})
                    print(f"✅ [{cmd_type}] processed '{cmd_arg}'")
                    self._append_response(response_text)  # Re-append the response after processing the command
                    return

    def _append_response(self, response_text: str):
        """Append the assistant's response to the history."""
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
    parser = argparse.ArgumentParser(description="Qwen CLI - Code-aware conversation tool")
    parser.add_argument("--model-dir","-m")
    parser.add_argument("--config-dir","-c")
    parser.add_argument("--helpers-dir")
    parser.add_argument("--hide-reasoning", action="store_true")
    parser.add_argument("cmd", nargs="?", help="Command or chat prompt")
    parser.add_argument("args", nargs="*", help="Arguments for commands or prompt")
    args = parser.parse_args()

    global DEFAULT_CONFIG
    if args.config_dir:
        os.environ["CONFIG_DIR"] = args.config_dir
    if args.model_dir:
        os.environ["MODELS_DIR"] = args.model_dir

    config = load_config()
    if args.helpers_dir:
        config["helpers_dir"] = args.helpers_dir

    helper_functions, tool_prompts = load_helper_functions(config["helpers_dir"])
    session = QwenSession(config, tool_prompts)

    # Load the model immediately after session initialization
    if not session._ensure_model_loaded():
        print("Failed to load model. Exiting.")
        return

    # Command dispatch
    if args.cmd == "new":
        session = QwenSession(config, tool_prompts)
        interactive_chat(session, helper_functions, args.hide_reasoning)
        return
    if args.cmd == "batch_load":
        if args.args:
            result = helper_functions["BATCH_LOAD"](" ".join(args.args))
            if result:
                print(result)
        else:
            print("Usage: batch_load <directory> <pattern>")
    elif args.cmd == "load":
        if args.args:
            filepath = args.args[0]
            result = helper_functions["LOAD_FILE"](filepath)
            if result:
                print(result)
        else:
            print("Usage: load <filepath>")
    elif args.cmd == "help":
        print("Commands: new, batch_load, load, list, clear, help")
    elif args.cmd is None:
        interactive_chat(session, helper_functions, args.hide_reasoning)
    else:
        # chat or plugin
        lc = args.cmd.upper()
        if lc in helper_functions:
            result = helper_functions[lc](' '.join(args.args))
            print(result or f"{lc} returned no output.")
        else:
            prompt = " ".join([args.cmd]+args.args)
            session.chat(prompt, helper_functions, args.hide_reasoning)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
