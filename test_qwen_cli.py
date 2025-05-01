import os
import unittest
from qwen_cli import load_helper_functions, build_system_prompt, QwenSession
from unittest.mock import patch, MagicMock

class TestHelperLoader(unittest.TestCase):
    def setUp(self):
        # Create a temporary helper file for testing
        self.test_dir = "test_helpers"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a sample helper with a docstring
        with open(os.path.join(self.test_dir, "test_helper.py"), "w") as f:
            f.write('''\"\"\"
Test helper function
\"\"\"
def handle_test(arg):
    \"\"\"[TEST_CMD args] – This is a test command\"\"\"
    return f"Processed {arg}"
''')

    def tearDown(self):
        # Clean up temporary files
        if os.path.exists(self.test_dir):
            for file in os.listdir(self.test_dir):
                file_path = os.path.join(self.test_dir, file)
                if os.path.isfile(file_path):  # Only remove files, not directories
                    os.remove(file_path)
            try:
                os.rmdir(self.test_dir)
            except OSError:
                pass  # Ignore if directory isn't empty

    def test_load_helper_functions(self):
        # Test that helper functions are loaded correctly
        helpers, tool_prompts = load_helper_functions(self.test_dir)
        
        self.assertTrue(helpers)  # Should find at least one helper
        
        # Check if any prompt contains the TEST_CMD command
        self.assertTrue(any("TEST_CMD" in prompt for prompt in tool_prompts))

    def test_system_prompt_changes(self):
        # Test that the system prompt changes when helpers are loaded
        base_prompt = build_system_prompt([])
        helpers, _ = load_helper_functions(self.test_dir)
        new_prompt = build_system_prompt(["TEST_CMD"])
        
        self.assertNotEqual(base_prompt, new_prompt)  # Prompt should change when tools are present

class TestQwenSession(unittest.TestCase):
    def setUp(self):
        # Create a mock config with helper functions
        self.config = {
            "model_repo": "test-model",
            "model_dir": "/test/models",
            "quantization": "8bit",
            "max_context_tokens": 120000,
            "max_new_tokens": 10000,
            "temperature": 0.1,
            "model_download_timeout": 1800,
            "helpers_dir": "helper_functions"
        }
        
        # Create mock helper functions
        self.helper_functions = {
            "LOAD_FILE": MagicMock(return_value="File content loaded")
        }
        
        # Create tool prompts from helpers
        self.tool_prompts = ["[LOAD_FILE args] – Load a file into the context"]
        
        # Initialize QwenSession with mocks
        self.session = QwenSession(self.config, self.tool_prompts)
        
        # Mock model and tokenizer
        self.session._ensure_model_loaded = MagicMock(return_value=True)
        self.session._model = MagicMock()
        self.session._tokenizer = MagicMock()

    def test_chat_normal_conversation(self):
        """Test basic chat functionality without special commands"""
        with patch.object(QwenSession, '_generate_response', 
                         return_value="Hello! How can I assist you today?"):
            # Initial history
            self.session.history = [{"role": "system", "content": "Base system prompt"}]
            
            # Chat
            success = self.session.chat("Hello", self.helper_functions)
            
            # Verify
            self.assertTrue(success)
            self.assertEqual(len(self.session.history), 3)
            self.assertEqual(self.session.history[1]["role"], "user")
            self.assertEqual(self.session.history[1]["content"], "Hello")
            self.assertEqual(self.session.history[2]["role"], "assistant")
            self.assertEqual(self.session.history[2]["content"], "Hello! How can I assist you today?")

    def test_chat_special_command(self):
        """Test chat with a special command that triggers a helper function"""
        # Mock the response to include a special command
        with patch.object(QwenSession, '_generate_response', 
                         return_value="[LOAD_FILE ./test.txt]"):
            # Initial history
            self.session.history = [{"role": "system", "content": "Base system prompt"}]
            
            # Chat
            success = self.session.chat("Please load the test file", self.helper_functions)
            
            # Verify helper was called
            self.helper_functions["LOAD_FILE"].assert_called_once_with("./test.txt")
            
            # Verify history has been updated with:
            # 1. User's original request
            # 2. System message from helper
            self.assertEqual(len(self.session.history), 3)
            self.assertEqual(self.session.history[1]["role"], "user")
            self.assertEqual(self.session.history[1]["content"], "Please load the test file")
            self.assertEqual(self.session.history[2]["role"], "system")
            self.assertEqual(self.session.history[2]["content"], "File content loaded")

    def test_chat_special_command_full_flow(self):
        """Test full flow of special command processing without recursion"""
        # Mock the response to include a special command
        with patch.object(QwenSession, '_generate_response', 
                         return_value="[LOAD_FILE ./test.txt]"):
            # Initial history
            self.session.history = [{"role": "system", "content": "Base system prompt"}]
            
            # Chat
            success = self.session.chat("Please load the test file", self.helper_functions)
            
            # Verify helper was called
            self.helper_functions["LOAD_FILE"].assert_called_once_with("./test.txt")
            
            # Verify history has been updated with:
            # 1. User's original request
            # 2. System message from helper
            # 3. User's follow-up prompt
            self.assertEqual(len(self.session.history), 5)
            self.assertEqual(self.session.history[1]["role"], "user")
            self.assertEqual(self.session.history[1]["content"], "Please load the test file")
            self.assertEqual(self.session.history[2]["role"], "system")
            self.assertEqual(self.session.history[2]["content"], "File content loaded")
            self.assertEqual(self.session.history[3]["role"], "user")
            self.assertEqual(self.session.history[3]["content"], "Please continue the analysis using the loaded file.")
