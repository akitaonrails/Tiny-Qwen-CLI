import unittest
from qwen_cli import load_helper_functions

class TestLoadHelperFunctions(unittest.TestCase):
    def test_load_helper_functions(self):
        helpers_dir = "helper_functions"
        expected_helpers = {
            "BATCH_LOAD": None,
            "LOAD_FILE": None,
            "FETCH_URL": None,
        }
        expected_tool_prompts = [
            "[BATCH_LOAD args] – If the user asks to read, load or analyze all the files from a relative path, such as ./src or similar,",
            "[LOAD_FILE args] – Whenever the user asks to read, load, analyze some code and provides a relative path, such as ./file.py or utils/utils.py or similar,",
            "[FETCH_URL args] – Whenever the user asks to read, load, research or consult a URL,",
        ]

        helpers, tool_prompts = load_helper_functions(helpers_dir)

        # Check if all expected helper functions are present
        for key in expected_helpers:
            self.assertIn(key, helpers)
            self.assertTrue(callable(helpers[key]))  # Ensure the function is callable

        # Check if all expected tool prompts are present
        for prompt in expected_tool_prompts:
            self.assertIn(prompt.split(" – ")[0], [tp.split(" – ")[0] for tp in tool_prompts])

if __name__ == "__main__":
    unittest.main()
