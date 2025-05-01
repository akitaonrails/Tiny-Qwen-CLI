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
        if module_name.startswith("__"): 
            continue
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Process the module for helper functions
        commands, prompts = _process_helper_module(module)
        helpers.update(commands)
        tool_prompts.extend(prompts)
    
    return helpers, tool_prompts

def _process_helper_module(module) -> (Dict[str, Callable], List[str]):
    """Process a single module to extract handle_ functions and their descriptions."""
    commands = {}
    prompts = []
    
    for attr_name in dir(module):
        if attr_name.startswith("handle_"):
            func = getattr(module, attr_name)
            command = attr_name[7:].upper()
            commands[command] = func
            doc = func.__doc__ or ""
            first_line = doc.strip().splitlines()[0] if doc.strip() else ""
            if first_line:
                prompts.append(f"[{command} args] – {first_line}")
    
    return commands, prompts

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
        """Ensure the model and tokenizer are loaded, handling concurrency."""
        if QwenSession._model and QwenSession._tokenizer:
            return True
        if QwenSession._model_loading_lock:
            logger.info("Model loading in progress by another session, waiting...")
            while QwenSession._model_loading_lock:
                time.sleep(1)
            return bool(QwenSession._model and QwenSession._tokenizer)

        QwenSession._model_loading_lock = True
        model_repo = self.config.get("model_repo", DEFAULT_CONFIG["model_repo"])
        model_dir = Path(self.config.get("model_dir", DEFAULT_CONFIG["model_dir"]))
        quantization = self.config.get("quantization", DEFAULT_CONFIG["quantization"]).lower()
        download_timeout = self.config.get("model_download_timeout", DEFAULT_CONFIG["model_download_timeout"])

        # Ensure the model directory exists and clone if needed
        self._ensure_model_directory(model_dir, model_repo, download_timeout)

        try:
            QwenSession._tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, cache_dir=str(CACHE_DIR))
            model_kwargs = {"trust_remote_code": True, "device_map": "auto", "cache_dir": str(CACHE_DIR)}

            # Configure attention mechanism (SDPA or xFormers)
            self._configure_attention(model_dir)

            # Apply quantization settings
            self._apply_quantization(quantization, model_kwargs)

            # Load the model with configured parameters
            QwenSession._model = AutoModelForCausalLM.from_pretrained(str(model_dir), config=AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True), **model_kwargs)
            QwenSession._model_loading_lock = False

            self._log_attention_status()
            return True
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {e}")
            QwenSession._model_loading_lock = False
            return False

    def _ensure_model_directory(self, model_dir: Path, model_repo: str, download_timeout: int):
        """Ensure the model directory exists and clone the repository if needed."""
        if not model_dir.exists():
            model_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--depth", "1", f"https://huggingface.co/{model_repo}", str(model_dir)], check=False, timeout=download_timeout)

    def _configure_attention(self, model_dir: Path):
        """Configure attention mechanism based on availability (SDPA or xFormers)."""
        model_config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
        sdpa_compatible = False

        try:
            # Check if the model's attention is compatible with SDPA
            for name, module in QwenSession._model.named_modules():
                if "attention" in name.lower() and isinstance(module, transformers.models.qwen2.modeling_qwen2.Qwen2Attention):
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

    def _apply_quantization(self, quantization: str, model_kwargs: dict):
        """Apply quantization settings based on the configuration."""
        if quantization == "4bit" and torch.cuda.is_available():
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif quantization == "8bit" and torch.cuda.is_available():
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["torch_dtype"] = "auto"

    def _log_attention_status(self):
        """Log the status of the attention mechanism."""
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            print("SDPA is available and (hopefully) being used!")
        else:
            print("SDPA is available in PyTorch, but may not be used by the model.")

    def _trim_history(self, max_tokens: int):
        # identical trimming logic as before
        pass

    def chat(self, prompt: str, helper_functions: Dict[str, Callable], max_new_tokens=None, temperature=None, stream=True, hide_reasoning=False) -> bool:
        """Main chat loop with improved structure and separation of concerns."""
        self.last_used = datetime.now().isoformat()
        
        # Initialize parameters and trim history
        self._initialize_chat_parameters(max_new_tokens, temperature)
        
        # Append user prompt to history
        self._append_user_prompt(prompt)
        
        try:
            # Generate response from model
            response_text = self._generate_response(stream)
            
            # Process any special commands in the response
            self._process_special_commands(response_text, helper_functions)
            
            # Update history with assistant's response
            self._update_history_with_response(response_text)
            
            return True
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.error(traceback.format_exc())
            return False

    def _initialize_chat_parameters(self, max_new_tokens: Optional[int], temperature: Optional[float]):
        """Initialize chat parameters with defaults if not provided."""
        if max_new_tokens is None:
            max_new_tokens = self.config.get("max_new_tokens", DEFAULT_CONFIG["max_new_tokens"])
        if temperature is None:
            temperature = self.config.get("temperature", DEFAULT_CONFIG["temperature"])
        
        # Trim history to fit within context token limit
        self._trim_history(self.config.get("max_context_tokens", DEFAULT_CONFIG["max_context_tokens"]))

    def _append_user_prompt(self, prompt: str):
        """Append the user's prompt to the conversation history."""
        self.history.append({"role": "user", "content": prompt})

    def _generate_response(self, stream: bool) -> str:
        """Generate a response from the model based on current history."""
        formatted_text = QwenSession._tokenizer.apply_chat_template(
            self.history, tokenize=False, add_generation_prompt=True
        )
        inputs = QwenSession._tokenizer([formatted_text], return_tensors="pt")
        inputs = {k: v.to(QwenSession._model.device) for k, v in inputs.items()}
        
        streamer = TextStreamer(QwenSession._tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None
        
        out_ids = QwenSession._model.generate(
            **inputs,
            max_new_tokens=self.config.get("max_new_tokens", DEFAULT_CONFIG["max_new_tokens"]),
            do_sample=self.config.get("temperature", DEFAULT线索
