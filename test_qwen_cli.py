import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import importlib.util
import logging
import torch
from qwen_cli import QwenSession, load_helper_functions

class TestChatFunctionality(unittest.TestCase):
    """Test the main chat command processing flow"""
    
    def setUp(self):
        # Setup mock session and helpers
        self.config = {
            "model_repo": "mock/repo",
            "model_dir": "/mock/path",
            "quantization": "8bit",
            "max_context_tokens": 1000,
            "max_new_tokens": 100,
            "temperature": 0.1,
            "helpers_dir": "helper_functions"
        }
        
        # Mock helpers
        self.helpers = {
            "LOAD_FILE": MagicMock(return_value="[file: test.txt]\n```text\nmock content\n```"),
            "FETCH_URL": MagicMock(return_value="[URL: http://test.url]\n```\nmock content\n```")
        }
        
        # Create session with empty history
        self.session = QwenSession(self.config, [])

        # Mock model and tokenizer with proper input structure
        self.session._model = MagicMock()
        self.session._tokenizer = MagicMock()
        self.session._tokenizer.decode.return_value = "mock response"
        self.session._tokenizer.apply_chat_template.return_value = "mock template"
        # Mock the return value for tokenizer([text], return_tensors="pt")
        self.session._tokenizer.return_value = {
            "input_ids": MagicMock(spec=torch.Tensor),
            "attention_mask": MagicMock(spec=torch.Tensor)
        }
        # Mock tensor shape
        self.session._tokenizer.return_value["input_ids"].shape = [1, 10]
        
        # Bypass actual model loading
        self.session._ensure_model_loaded = MagicMock(return_value=True)
        
        # Initialize tokenizer and model class variables
        QwenSession._model = self.session._model
        QwenSession._tokenizer = self.session._tokenizer

    def test_load_file_command_processing(self):
        """Test that LOAD_FILE commands are detected and handled properly"""
        # Mock the model responses - first with command, then normal
        mock_responses = [
            "[LOAD_FILE test.txt] Let me analyze it.",  # First response with command
            "Analyzing file content..."                 # Second normal response
        ]
        self.session._tokenizer.decode.side_effect = mock_responses
        
        # Execute chat with sample prompt
        result = self.session.chat(
            prompt="Please analyze test.txt",
            helper_functions=self.helpers,
            stream=False
        )
        
        # Verify helper was called
        self.helpers["LOAD_FILE"].assert_called_once_with("test.txt")
        
        # Verify history updates
        self.assertEqual(len(self.session.history), 7)  # system + user + assistant (initial) + system + user (automatic follow-up) + user + assistant (second response)
        self.assertEqual(self.session.history[-3]["role"], "system")
        self.assertIn("mock content", self.session.history[-3]["content"])
        self.assertEqual(self.session.history[-2]["role"], "assistant")
        self.assertEqual(self.session.history[-1]["role"], "user")
        self.assertEqual(self.session.history[-1]["content"], "Please continue the analysis using the loaded file.")
        
        # Verify return value
        self.assertTrue(result)

    def test_chat_without_commands(self):
        """Test normal chat flow without special commands"""
        # Mock normal response without commands
        self.session._tokenizer.decode.return_value = "This is a normal response."
        # Clear history between tests
        self.session.history = [{"role": "system", "content": "test system prompt"}]
        
        result = self.session.chat(
            prompt="Hello",
            helper_functions=self.helpers,
            stream=False
        )
        
        # Verify no helpers called
        self.helpers["LOAD_FILE"].assert_not_called()
        self.helpers["FETCH_URL"].assert_not_called()
        
        # Verify history updated
        self.assertEqual(len(self.session.history), 3)  # system + user + assistant
        self.assertEqual(self.session.history[-1]["role"], "assistant")
        self.assertEqual(self.session.history[-1]["content"], "This is a normal response.")

class TestHelperFunctionsLoading(unittest.TestCase):
    """Test the dynamic loading of helper functions"""
    
    def setUp(self):
        # Set up a temporary directory structure
        self.test_helpers_dir = Path("/tmp/test_helpers")
        self.test_helpers_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample helper files
        (self.test_helpers_dir / "test_helper1.py").write_text(
            'def handle_test1(arg):\n    """Test1 command - helps with testing"""\n    return "test1: " + arg\n'
        )
        (self.test_helpers_dir / "test_helper2.py").write_text(
            'def handle_test2(arg):\n    """Test2 command - another test helper"""\n    return "test2: " + arg\n'
        )
        # Helper file without docstring
        (self.test_helpers_dir / "no_docstring.py").write_text(
            'def handle_test3(arg):\n    return "no docs"\n'
        )

    def test_loads_helpers_correctly(self):
        # Patch the module loading mechanism
        with patch("importlib.util.module_from_spec") as mock_module, \
             patch("importlib.util.spec_from_file_location") as mock_spec:
            
            # Set up mock module with docstrings
            mock_mod = MagicMock()
            mock_mod.handle_test1 = lambda x: "test1: " + x
            mock_mod.handle_test1.__doc__ = "Test1 command - helps with testing"
            mock_mod.handle_test2 = lambda x: "test2: " + x 
            mock_mod.handle_test2.__doc__ = "Test2 command - another test helper"
            mock_module.return_value = mock_mod
            
            # Execute
            helpers, tool_prompts = load_helper_functions(str(self.test_helpers_dir))
            
            # Verify helpers
            self.assertIn("TEST1", helpers)
            self.assertIn("TEST2", helpers)
            self.assertEqual(helpers["TEST1"]("foo"), "test1: foo")
            self.assertEqual(helpers["TEST2"]("bar"), "test2: bar")
            
            # Verify tool prompts
            self.assertIn("[TEST1 args] – Test1 command - helps with testing", tool_prompts)
            self.assertIn("[TEST2 args] – Test2 command - another test helper", tool_prompts)
            
            # Verify helper without docstring is excluded from prompts
            self.assertNotIn("TEST3", [cmd.split()[0][1:] for cmd in tool_prompts])

    def test_ignores_invalid_files(self):
        # Create invalid Python file
        (self.test_helpers_dir / "invalid.py").write_text("invalid syntax")  # Invalid Python syntax
        
        # Should load other helpers and ignore the invalid file
        with self.assertLogs(level="ERROR") as log:
            helpers, tool_prompts = load_helper_functions(str(self.test_helpers_dir))
            
        self.assertIn("TEST1", helpers)
        self.assertIn("Error loading helper module", log.output[0])

    def test_empty_directory_handling(self):
        empty_dir = Path("/tmp/empty_helpers")
        empty_dir.mkdir(exist_ok=True)
        
        helpers, tool_prompts = load_helper_functions(str(empty_dir))
        self.assertEqual(len(helpers), 0)
        self.assertEqual(len(tool_prompts), 0)

    def test_file_name_convention(self):
        # Should ignore files without .py extension
        (self.test_helpers_dir / "skip.txt").write_text(
            'def handle_skip(arg): """Should be skipped"""'
        )
        
        helpers, tool_prompts = load_helper_functions(str(self.test_helpers_dir))
        self.assertNotIn("SKIP", helpers)

    def tearDown(self):
        # Cleanup test files
        for f in self.test_helpers_dir.glob("*"):
            if f.is_file():
                f.unlink(missing_ok=True)
        # Remove __pycache__ directory if it exists
        pycache = self.test_helpers_dir / "__pycache__"
        if pycache.exists():
            import shutil
            shutil.rmtree(pycache)
        self.test_helpers_dir.rmdir()

if __name__ == "__main__":
    unittest.main()
