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
from typing import Any, Callable, Dict, List, Optional

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
def load_config():
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        return DEFAULT_CONFIG

# --- Load Helper Functions ---
def load_helper_functions(helpers_dir):
    helper_functions = {}
    tool_prompts = []

    for filename in os.listdir(helpers_dir):
        if filename.endswith('.py'):
            filepath = Path(helpers_dir) / filename
            spec = importlib.util.spec_from_file_location(filename[:-3], filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for attr_name in dir(module):
                if attr_name.startswith('handle_') and callable(getattr(module, attr_name)):
                    helper_functions[attr_name] = getattr(module, attr_name)
                    tool_prompts.append(f"{attr_name}: {getattr(module, attr_name).__doc__}")

    return helper_functions, tool_prompts

# --- Interactive Chat ---
def interactive_chat(session, helper_functions):
    print("Interactive chat session started. Type 'bye' to exit.")
    while True:
        try:
            prompt = input("You: ")
            if prompt.lower() == 'bye':
                break
            response = session.chat(prompt, helper_functions)
            print(f"Assistant: {response}")
        except KeyboardInterrupt:
            print("\nExiting chat session.")
            break

# --- Refactored QwenSession Class ---
class QwenSession:
    _model = None
    _tokenizer = None

    def __init__(self, config, tool_prompts):
        self.config = config
        self.tool_prompts = tool_prompts
        self.history = [{"role": "system", "content": build_system_prompt(tool_prompts)}]

    def _ensure_model_loaded(self) -> bool:
        if not QwenSession._model or not QwenSession._tokenizer:
            try:
                self._load_model()
                return True
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return False
        return True

    def _load_model(self):
        config = AutoConfig.from_pretrained(self.config["model_dir"])
        quantization_config = None
        torch_dtype = "auto"

        if self.config["quantization"] == "4bit" and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            torch_dtype = torch.bfloat16
        elif self.config["quantization"] == "8bit" and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        QwenSession._model = AutoModelForCausalLM.from_pretrained(
            self.config["model_dir"],
            config=config,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype
        )
        QwenSession._tokenizer = AutoTokenizer.from_pretrained(self.config["model_dir"])

    def _trim_history(self, max_tokens: int):
        # Implement history trimming logic here
        pass

    def chat(self, prompt: str, helper_functions: Dict[str, Callable], max_new_tokens=None, temperature=None, stream=True, hide_reasoning=False) -> bool:
        self.last_prompt = prompt  # Store the last prompt for testing purposes
        self.last_response = None  # Store the last response for testing purposes

        try:
            formatted_text = QwenSession._tokenizer.apply_chat_template(
                self.history, tokenize=False, add_generation_prompt=True
            )
            inputs = QwenSession._tokenizer([formatted_text], return_tensors="pt")
            inputs = {k: v.to(QwenSession._model.device) for k, v in inputs.items()}
            streamer = TextStreamer(QwenSession._tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None
            out_ids = QwenSession._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                streamer=streamer,
                repetition_penalty=1.1
            )
            generated = out_ids[0][inputs["input_ids"].shape[1]:]
            response = QwenSession._tokenizer.decode(generated, skip_special_tokens=True)
            self.last_response = response  # Store the last response for testing purposes

            # Check for [LOAD_FILE ...] commands in the response
            if "[LOAD_FILE" in response:
                import re
                matches = re.findall(r'\[LOAD_FILE\s+([^\]]+)\]', response)
                for match in matches:
                    if match in helper_functions:
                        file_content = helper_functions['handle_load_file'](match)
                        self.history.append({"role": "assistant", "content": file_content})
            return True
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.error(traceback.format_exc())
            return False

    def _generate_response(self, prompt: str, max_new_tokens: int, temperature: float, stream: bool) -> str:
        try:
            formatted_text = QwenSession._tokenizer.apply_chat_template(
                self.history, tokenize=False, add_generation_prompt=True
            )
            inputs = QwenSession._tokenizer([formatted_text], return_tensors="pt")
            inputs = {k: v.to(QwenSession._model.device) for k, v in inputs.items()}
            streamer = TextStreamer(QwenSession._tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None
            out_ids = QwenSession._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                streamer=streamer,
                repetition_penalty=1.1
            )
            generated = out_ids[0][inputs["input_ids"].shape[1]:]
            return QwenSession._tokenizer.decode(generated, skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.error(traceback.format_exc())
            return ""
