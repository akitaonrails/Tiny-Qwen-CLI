import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, call, ANY

# Ensure the main script directory is in the path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Mock modules that might not be installed in a pure test environment
# or that we want to control explicitly
mock_torch = MagicMock()
mock_transformers = MagicMock()
mock_accelerate = MagicMock()
mock_readline = MagicMock()
mock_helper_utils = MagicMock()

sys.modules['torch'] = mock_torch
sys.modules['transformers'] = mock_transformers
sys.modules['accelerate'] = mock_accelerate
sys.modules['readline'] = mock_readline # Mock readline for non-interactive environments
sys.modules['helper_functions'] = MagicMock()
sys.modules['helper_functions.utils'] = mock_helper_utils

# Set dummy return values for mocked imports if needed by the module being tested
mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()
mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = MagicMock()
mock_transformers.AutoConfig.from_pretrained.return_value = MagicMock()
mock_transformers.BitsAndBytesConfig.return_value = MagicMock()
mock_transformers.TextStreamer.return_value = MagicMock()
mock_helper_utils.get_language_from_extension.return_value = "python"

# Now import the module under test AFTER mocking dependencies
import qwen_cli

class TestLoadHelperFunctions(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for dummy helper files
        self.test_dir = tempfile.mkdtemp()
        self.helpers_path = Path(self.test_dir)

    def tearDown(self):
        # Remove the temporary directory after the test
        shutil.rmtree(self.test_dir)

    def _create_dummy_helper(self, name, content):
        filepath = self.helpers_path / f"{name}.py"
        with open(filepath, "w") as f:
            f.write(content)
        return filepath

    def test_load_helpers_basic(self):
        """Test loading a simple valid helper function."""
        helper_content = """
import logging
logger = logging.getLogger(__name__)
def handle_test_command(arg_string):
    '''Test command description.'''
    return f"Processed: {arg_string}"
"""
        self._create_dummy_helper("test_helper", helper_content)

        helpers, tool_prompts = qwen_cli.load_helper_functions(str(self.helpers_path))

        self.assertIn("TEST_COMMAND", helpers)
        self.assertEqual(len(helpers), 1)
        self.assertTrue(callable(helpers["TEST_COMMAND"]))
        self.assertEqual(len(tool_prompts), 1)
        self.assertEqual(tool_prompts[0], "[TEST_COMMAND args] – Test command description.")
        # Test the loaded function indirectly
        self.assertEqual(helpers["TEST_COMMAND"]("hello"), "Processed: hello")

    def test_load_helpers_no_docstring(self):
        """Test loading a helper function without a docstring."""
        helper_content = """
def handle_no_doc(arg):
    pass
"""
        self._create_dummy_helper("no_doc_helper", helper_content)
        helpers, tool_prompts = qwen_cli.load_helper_functions(str(self.helpers_path))

        self.assertIn("NO_DOC", helpers)
        self.assertEqual(len(helpers), 1)
        self.assertEqual(len(tool_prompts), 0) # No docstring, no prompt

    def test_load_helpers_ignore_non_handle(self):
        """Test that functions not starting with 'handle_' are ignored."""
        helper_content = """
def not_a_handler(arg):
    '''Should be ignored.'''
    pass
def handle_real_handler(arg):
    '''Real one.'''
    return "Real"
"""
        self._create_dummy_helper("mixed_helper", helper_content)
        helpers, tool_prompts = qwen_cli.load_helper_functions(str(self.helpers_path))

        self.assertNotIn("NOT_A_HANDLER", helpers)
        self.assertIn("REAL_HANDLER", helpers)
        self.assertEqual(len(helpers), 1)
        self.assertEqual(len(tool_prompts), 1)
        self.assertEqual(tool_prompts[0], "[REAL_HANDLER args] – Real one.")

    def test_load_helpers_ignore_private_files(self):
        """Test that files starting with '__' are ignored."""
        helper_content = """
def handle_should_not_load(arg):
    '''This should not load.'''
    pass
"""
        self._create_dummy_helper("__init__", " ") # Create __init__.py
        self._create_dummy_helper("__private_helper", helper_content)
        helpers, tool_prompts = qwen_cli.load_helper_functions(str(self.helpers_path))

        self.assertEqual(len(helpers), 0)
        self.assertEqual(len(tool_prompts), 0)

    def test_load_helpers_import_error(self):
        """Test handling of files that cause import errors."""
        helper_content = "import non_existent_module"
        self._create_dummy_helper("bad_import_helper", helper_content)

        # Use patch to simulate the logger.error call
        with patch.object(qwen_cli.logger, 'error') as mock_logger_error:
            helpers, tool_prompts = qwen_cli.load_helper_functions(str(self.helpers_path))
            self.assertEqual(len(helpers), 0)
            self.assertEqual(len(tool_prompts), 0)
            mock_logger_error.assert_called_once()
            self.assertIn("Failed to load module", mock_logger_error.call_args[0][0])

    def test_ensure_helpers_dir_creation(self):
        """Test that the helpers directory and __init__.py are created if missing."""
        missing_helpers_path = self.helpers_path / "sub_helpers"
        self.assertFalse(missing_helpers_path.exists())

        qwen_cli._ensure_helpers_dir(missing_helpers_path)

        self.assertTrue(missing_helpers_path.exists())
        self.assertTrue((missing_helpers_path / "__init__.py").exists())

class TestQwenSessionChat(unittest.TestCase):

    def setUp(self):
        self.config = qwen_cli.DEFAULT_CONFIG.copy()
        self.tool_prompts = ["[MOCK_TOOL args] – Mock tool description."]
        self.session = qwen_cli.QwenSession(self.config, self.tool_prompts)

        # Mock model/tokenizer loading and generation
        self.session._ensure_model_loaded = MagicMock(return_value=True)
        qwen_cli.QwenSession._tokenizer = MagicMock()
        qwen_cli.QwenSession._model = MagicMock()
        qwen_cli.QwenSession._tokenizer.apply_chat_template = MagicMock(return_value="<formatted_prompt>")
        qwen_cli.QwenSession._tokenizer.return_value = {"input_ids": MagicMock(shape=(1, 10)), "attention_mask": MagicMock()} # Mock __call__
        qwen_cli.QwenSession._tokenizer.decode = MagicMock(return_value="Test response")
        qwen_cli.QwenSession._model.generate = MagicMock(return_value=[[0]*20]) # Dummy output IDs
        qwen_cli.QwenSession._model.device = 'cpu' # Mock device

        # Mock helper functions dict
        self.mock_helper_func = MagicMock(return_value="Helper result message")
        self.helper_functions = {"MOCK_TOOL": self.mock_helper_func}

    def test_chat_simple_response(self):
        """Test a simple chat interaction without tool usage."""
        qwen_cli.QwenSession._tokenizer.decode.return_value = "This is a simple reply."
        initial_history_len = len(self.session.history)

        success = self.session.chat("Hello there", self.helper_functions, stream=False)

        self.assertTrue(success)
        self.assertEqual(len(self.session.history), initial_history_len + 2) # user + assistant
        self.assertEqual(self.session.history[-2]['role'], 'user')
        self.assertEqual(self.session.history[-2]['content'], 'Hello there')
        self.assertEqual(self.session.history[-1]['role'], 'assistant')
        self.assertEqual(self.session.history[-1]['content'], 'This is a simple reply.')
        qwen_cli.QwenSession._model.generate.assert_called_once()
        self.mock_helper_func.assert_not_called()

    def test_chat_with_tool_call_success(self):
        """Test chat where the response triggers a successful tool call."""
        response_with_tool = "Okay, I need to use the tool. [MOCK_TOOL data_to_process]"
        qwen_cli.QwenSession._tokenizer.decode.return_value = response_with_tool
        initial_history_len = len(self.session.history)

        success = self.session.chat("Process this data", self.helper_functions, stream=False)

        self.assertTrue(success)
        # History should contain: user prompt, system message (tool result)
        # The assistant message containing the tool command itself is NOT added
        self.assertEqual(len(self.session.history), initial_history_len + 2)
        self.assertEqual(self.session.history[-2]['role'], 'user')
        self.assertEqual(self.session.history[-2]['content'], 'Process this data')
        self.assertEqual(self.session.history[-1]['role'], 'system')
        self.assertEqual(self.session.history[-1]['content'], 'Helper result message') # Result from mock_helper_func

        # Verify helper was called correctly
        self.mock_helper_func.assert_called_once_with("data_to_process")
        qwen_cli.QwenSession._model.generate.assert_called_once()

    def test_chat_with_tool_call_no_result(self):
        """Test chat where the tool call executes but returns None/empty."""
        response_with_tool = "Using the tool now... [MOCK_TOOL some_arg]"
        qwen_cli.QwenSession._tokenizer.decode.return_value = response_with_tool
        self.mock_helper_func.return_value = None # Simulate no result
        initial_history_len = len(self.session.history)

        success = self.session.chat("Run the tool", self.helper_functions, stream=False)

        self.assertTrue(success)
        # History: user prompt, system message (note about no output)
        self.assertEqual(len(self.session.history), initial_history_len + 2)
        self.assertEqual(self.session.history[-2]['role'], 'user')
        self.assertEqual(self.session.history[-1]['role'], 'system')
        self.assertIn("executed but returned no output", self.session.history[-1]['content'])

        self.mock_helper_func.assert_called_once_with("some_arg")
        qwen_cli.QwenSession._model.generate.assert_called_once()

    def test_chat_with_tool_call_exception(self):
        """Test chat where the tool call raises an exception."""
        response_with_tool = "Let me try this: [MOCK_TOOL bad_arg]"
        qwen_cli.QwenSession._tokenizer.decode.return_value = response_with_tool
        error_message = "Something went wrong"
        self.mock_helper_func.side_effect = Exception(error_message)
        initial_history_len = len(self.session.history)

        with patch.object(qwen_cli.logger, 'error') as mock_logger_error:
            success = self.session.chat("Trigger error", self.helper_functions, stream=False)

            self.assertTrue(success) # Chat turn itself succeeds, but tool fails
            # History: user prompt, system message (error notification)
            self.assertEqual(len(self.session.history), initial_history_len + 2)
            self.assertEqual(self.session.history[-2]['role'], 'user')
            self.assertEqual(self.session.history[-1]['role'], 'system')
            self.assertIn("Error: Failed to execute tool", self.session.history[-1]['content'])
            self.assertIn(error_message, self.session.history[-1]['content'])

            self.mock_helper_func.assert_called_once_with("bad_arg")
            qwen_cli.QwenSession._model.generate.assert_called_once()
            # Check that the error was logged
            mock_logger_error.assert_any_call(f"Error executing helper command [MOCK_TOOL bad_arg]: {error_message}")

    def test_chat_model_load_failure(self):
        """Test chat when model loading fails."""
        self.session._ensure_model_loaded = MagicMock(return_value=False) # Simulate failure

        with patch('builtins.print') as mock_print:
            success = self.session.chat("Hi", self.helper_functions)
            self.assertFalse(success)
            mock_print.assert_called_with("Model is not loaded. Cannot proceed with chat.")
            qwen_cli.QwenSession._model.generate.assert_not_called()

    def test_chat_generation_exception(self):
        """Test chat when model.generate raises an exception."""
        error_message = "Generation failed"
        qwen_cli.QwenSession._model.generate.side_effect = Exception(error_message)
        initial_history_len = len(self.session.history)

        with patch.object(qwen_cli.logger, 'error') as mock_logger_error:
            success = self.session.chat("Generate something", self.helper_functions)

            self.assertFalse(success)
            # History should revert to state before user prompt was added
            self.assertEqual(len(self.session.history), initial_history_len)
            self.assertNotEqual(self.session.history[-1]['role'], 'user') # Last message should not be the failed user prompt

            mock_logger_error.assert_any_call(f"Error generating response: {error_message}")


class TestInteractiveChat(unittest.TestCase):

    def setUp(self):
        self.config = qwen_cli.DEFAULT_CONFIG.copy()
        self.tool_prompts = []
        self.session = qwen_cli.QwenSession(self.config, self.tool_prompts)
        self.helper_functions = {}

        # Mock session.chat as it's tested separately
        self.session.chat = MagicMock(return_value=True)

    @patch('builtins.input')
    @patch('sys.exit')
    @patch('signal.signal') # Mock signal handling
    @patch('builtins.print') # To capture output
    def test_interactive_chat_normal_flow(self, mock_print, mock_signal, mock_exit, mock_input):
        """Test the normal flow: input -> chat -> input -> exit."""
        mock_input.side_effect = ["hello", "how are you?", "bye"]

        qwen_cli.interactive_chat(self.session, self.helper_functions)

        # Check calls to input
        self.assertEqual(mock_input.call_count, 3)

        # Check calls to session.chat
        expected_chat_calls = [
            call("hello", self.helper_functions, stream=True, hide_reasoning=False),
            call("how are you?", self.helper_functions, stream=True, hide_reasoning=False)
        ]
        self.session.chat.assert_has_calls(expected_chat_calls)
        self.assertEqual(self.session.chat.call_count, 2) # Not called for "bye"

        # Check exit call
        mock_exit.assert_called_once_with(0)
        # Check that goodbye message was printed
        mock_print.assert_any_call("\nExiting chat session. Goodbye!\n")

    @patch('builtins.input')
    @patch('sys.exit')
    @patch('signal.signal')
    @patch('builtins.print')
    def test_interactive_chat_eof_exit(self, mock_print, mock_signal, mock_exit, mock_input):
        """Test exiting via EOFError (Ctrl+D)."""
        mock_input.side_effect = EOFError

        qwen_cli.interactive_chat(self.session, self.helper_functions)

        mock_input.assert_called_once()
        self.session.chat.assert_not_called()
        mock_exit.assert_called_once_with(0)
        mock_print.assert_any_call("\nEOF (Ctrl+D) detected.")
        mock_print.assert_any_call("\nExiting chat session. Goodbye!\n")

    @patch('builtins.input')
    @patch('sys.exit')
    @patch('signal.signal')
    @patch('builtins.print')
    def test_interactive_chat_empty_input(self, mock_print, mock_signal, mock_exit, mock_input):
        """Test that empty input is ignored."""
        mock_input.side_effect = ["", "   ", "hello", "bye"] # Empty, whitespace only, valid, exit

        qwen_cli.interactive_chat(self.session, self.helper_functions)

        self.assertEqual(mock_input.call_count, 4)
        # session.chat should only be called for "hello"
        self.session.chat.assert_called_once_with("hello", self.helper_functions, stream=True, hide_reasoning=False)
        mock_exit.assert_called_once_with(0)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
