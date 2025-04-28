import os
import tempfile
import unittest
import torch
from qwen_cli import load_helper_functions, QwenSession
from pathlib import Path

class TestQwenCLIMethods(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for helper modules
        self.tmpdir = tempfile.TemporaryDirectory()
        self.helpers_dir = self.tmpdir.name

        # Create sample helper files
        helper1_path = os.path.join(self.helpers_dir, "helper1.py")
        with open(helper1_path, "w") as f:
            f.write('''def handle_example1(args):
    \"\"\"This is an example helper function for demonstration\"\"\"
    return "example1 result"''')
        
        helper2_path = os.path.join(self.helpers_dir, "helper2.py")
        with open(helper2_path, "w") as f:
            f.write('''def handle_example2(args):
    \"\"\"Another example helper function\"\"\"
    return "example2 result"''')
        
        helper3_path = os.path.join(self.helpers_dir, "helper3.py")
        with open(helper3_path, "w") as f:
            f.write('''def handle_example3(args):
    \"\"\"Third example helper function\"\"\"
    return "example3 result"''')

    def test_load_helper_modules(self):
        """Test that load_helper_functions correctly loads helper modules and collects tool prompts."""
        helpers, tool_prompts = load_helper_functions(self.helpers_dir)

        # Dynamically count the number of helper files
        expected_helper_count = len(list(Path(self.helpers_dir).glob("*.py")))
        self.assertEqual(len(helpers), expected_helper_count, "Should load all helper modules")

        # Expected tool prompts with correct format
        expected_prompts = [
            "[EXAMPLE1 args] – This is an example helper function for demonstration",
            "[EXAMPLE2 args] – Another example helper function",
            "[EXAMPLE3 args] – Third example helper function"
        ]

        # Check that all expected prompts are present (order-insensitive)
        for prompt in expected_prompts:
            self.assertIn(prompt, tool_prompts, f"Should include prompt: {prompt}")

        # Ensure all helper functions have docstrings
        for module in helpers:
            for attr_name in dir(module):
                if attr_name.startswith("handle_"):
                    func = getattr(module, attr_name)
                    self.assertIsNotNone(func.__doc__, f"Function {attr_name} should have a docstring")

    def test_chat_method(self):
        """Test the chat method with a mocked model response."""
        # Mock the model to return a fixed response
        class MockModel:
            def __init__(self):
                self.device = torch.device("cpu")  # Add device attribute

            def generate(self, **kwargs):
                return torch.tensor([[1, 2, 3, 4, 5]])

        class MockTokenizer:
            def __call__(self, *args, **kwargs):
                return self.apply_chat_template(*args, **kwargs)
            
            def apply_chat_template(self, history, **kwargs):
                # Return a mock dictionary with tensor-like structure
                return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
            
            def decode(self, token_ids, **kwargs):
                return "mocked response"

        # Create a session with mocked model and tokenizer
        session = QwenSession({}, [])

        # Set class variables directly
        QwenSession._model = MockModel()
        QwenSession._tokenizer = MockTokenizer()

        # Test chat with a sample prompt
        result = session.chat("Test prompt", {}, stream=True)

        self.assertTrue(result, "Chat should return True on success")
        self.assertGreater(len(session.history), 1, "History should be updated with user and assistant messages")
        self.assertEqual(session.history[-1]["role"], "assistant", "Last message should be from assistant")
        self.assertEqual(session.history[-1]["content"], "mocked response", "Should return mocked response")

    def tearDown(self):
        # Clean up the temporary directory
        self.tmpdir.cleanup()
