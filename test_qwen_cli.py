import unittest
from unittest.mock import patch, MagicMock
from io import StringIO
from qwen_cli import QwenSession, interactive_chat, load_helper_functions

class TestQwenCLI(unittest.TestCase):

    @patch('qwen_cli.load_config')
    @patch('qwen_cli.load_helper_functions')
    def setUp(self, mock_load_helpers, mock_load_config):
        mock_load_config.return_value = {
            "model_repo": "Qwen/Qwen2.5-Coder-14B-Instruct",
            "model_dir": "/models/Qwen2.5-Coder-14B-Instruct",
            "quantization": "8bit",
            "max_context_tokens": 120000,
            "max_new_tokens": 10000,
            "temperature": 0.1,
            "model_download_timeout": 1800,
            "helpers_dir": "helper_functions",
        }
        mock_load_helpers.return_value = ({}, [])
        self.session = QwenSession(mock_load_config(), [])

    @patch('qwen_cli.QwenSession._load_model')
    def test_session_initialization(self, mock_load_model):
        mock_load_model.return_value = True
        self.assertTrue(self.session._load_model())

    @patch('sys.stdout', new_callable=StringIO)
    def test_interactive_chat_exit(self, mock_stdout):
        with patch('builtins.input', side_effect=['bye']):
            try:
                interactive_chat(self.session, {})
            except SystemExit as e:
                self.assertEqual(e.code, 0)
        self.assertIn("Exiting chat session. Goodbye!", mock_stdout.getvalue())

    @patch('sys.stdout', new_callable=StringIO)
    def test_interactive_chat_command(self, mock_stdout):
        with patch('builtins.input', side_effect=['new', 'bye']):
            try:
                interactive_chat(self.session, {})
            except SystemExit as e:
                self.assertEqual(e.code, 0)
        self.assertIn("Interactive chat session started. Type 'bye' to exit.", mock_stdout.getvalue())
        self.assertIn("Starting a new chat session.", mock_stdout.getvalue())

    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_load_helper_functions(self, mock_module_from_spec, mock_spec_from_file_location):
        # Mocking the spec and module
        mock_spec = MagicMock()
        mock_module = MagicMock()
        mock_spec_from_file_location.return_value = mock_spec
        mock_module_from_spec.return_value = mock_module

        # Mocking the dir function to return a list of attributes including a callable 'handle_test'
        mock_dir = MagicMock(return_value=['handle_test'])
        mock_getattr = MagicMock(return_value=lambda x: "Test Function")
        with patch('os.listdir', return_value=['test.py']), \
             patch('builtins.dir', mock_dir), \
             patch('builtins.getattr', mock_getattr):
            helper_functions, tool_prompts = load_helper_functions("helper_functions")

        print(f"Helper functions: {helper_functions}")
        print(f"Tool prompts: {tool_prompts}")

        self.assertIn('handle_test', helper_functions)
        self.assertEqual(helper_functions['handle_test'](), "Test Function")
        self.assertIn('handle_test: Test Function', tool_prompts)

if __name__ == '__main__':
    unittest.main()
