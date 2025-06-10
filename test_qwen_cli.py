#!/usr/bin/env python3
"""
Unit tests for qwen_cli.py, focusing on the load_helper_functions method.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import sys
import os

# Add the project root to the path so we can import qwen_cli
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen_cli import load_helper_functions, QwenSession, parse_special_commands


class TestLoadHelperFunctions(unittest.TestCase):
    """Test the dynamic helper function loading functionality."""

    def setUp(self):
        """Set up temporary directory structure for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.helpers_path = Path(self.temp_dir) / "test_helpers"
        self.helpers_path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py file
        (self.helpers_path / "__init__.py").touch()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def _create_helper_file(self, filename, content):
        """Helper method to create a helper function file."""
        file_path = self.helpers_path / filename
        file_path.write_text(content)
        return file_path

    def test_load_helper_functions_empty_directory(self):
        """Test loading from an empty helpers directory."""
        helpers, tool_prompts = load_helper_functions(str(self.helpers_path))
        
        self.assertEqual(helpers, {})
        self.assertEqual(tool_prompts, [])

    def test_load_helper_functions_single_helper(self):
        """Test loading a single helper function."""
        helper_content = '''
def handle_test_command(args):
    """Test command for unit testing"""
    return f"Test executed with: {args}"
'''
        self._create_helper_file("test_helper.py", helper_content)
        
        helpers, tool_prompts = load_helper_functions(str(self.helpers_path))
        
        # Check that the helper function was loaded
        self.assertIn("TEST_COMMAND", helpers)
        self.assertTrue(callable(helpers["TEST_COMMAND"]))
        
        # Check that the tool prompt was generated
        self.assertEqual(len(tool_prompts), 1)
        self.assertIn("[TEST_COMMAND args] – Test command for unit testing", tool_prompts)
        
        # Test that the function actually works
        result = helpers["TEST_COMMAND"]("hello world")
        self.assertEqual(result, "Test executed with: hello world")

    def test_load_helper_functions_multiple_helpers(self):
        """Test loading multiple helper functions from different files."""
        # Create first helper file
        helper1_content = '''
def handle_load_file(filepath):
    """Load a file from the filesystem"""
    return f"Loading file: {filepath}"

def handle_save_file(filepath):
    """Save content to a file"""
    return f"Saving to: {filepath}"
'''
        self._create_helper_file("file_ops.py", helper1_content)
        
        # Create second helper file
        helper2_content = '''
def handle_fetch_url(url):
    """Fetch content from a URL"""
    return f"Fetching: {url}"
'''
        self._create_helper_file("web_ops.py", helper2_content)
        
        helpers, tool_prompts = load_helper_functions(str(self.helpers_path))
        
        # Check that all helper functions were loaded
        expected_commands = {"LOAD_FILE", "SAVE_FILE", "FETCH_URL"}
        self.assertEqual(set(helpers.keys()), expected_commands)
        
        # Check that all are callable
        for command in expected_commands:
            self.assertTrue(callable(helpers[command]))
        
        # Check tool prompts
        self.assertEqual(len(tool_prompts), 3)
        prompt_texts = [prompt for prompt in tool_prompts]
        
        self.assertIn("[LOAD_FILE args] – Load a file from the filesystem", prompt_texts)
        self.assertIn("[SAVE_FILE args] – Save content to a file", prompt_texts)
        self.assertIn("[FETCH_URL args] – Fetch content from a URL", prompt_texts)

    def test_load_helper_functions_ignores_non_handle_functions(self):
        """Test that non-handle functions are ignored."""
        helper_content = '''
def handle_valid_command(args):
    """This should be loaded"""
    return "valid"

def regular_function(args):
    """This should be ignored"""
    return "ignored"

def another_handle_command(args):
    """This should also be ignored because it doesn't start with handle_"""
    return "also ignored"

def handle_another_valid(args):
    """This should be loaded too"""
    return "also valid"
'''
        self._create_helper_file("mixed_functions.py", helper_content)
        
        helpers, tool_prompts = load_helper_functions(str(self.helpers_path))
        
        # Only functions starting with "handle_" should be loaded
        expected_commands = {"VALID_COMMAND", "ANOTHER_VALID"}
        self.assertEqual(set(helpers.keys()), expected_commands)
        self.assertEqual(len(tool_prompts), 2)

    def test_load_helper_functions_ignores_dunder_files(self):
        """Test that __init__.py and other dunder files are ignored."""
        # Create a dunder file that would otherwise be loaded
        dunder_content = '''
def handle_dunder_command(args):
    """This should be ignored"""
    return "dunder"
'''
        self._create_helper_file("__dunder__.py", dunder_content)
        
        # Create a normal file
        normal_content = '''
def handle_normal_command(args):
    """This should be loaded"""
    return "normal"
'''
        self._create_helper_file("normal.py", normal_content)
        
        helpers, tool_prompts = load_helper_functions(str(self.helpers_path))
        
        # Only the normal file should be loaded
        self.assertEqual(set(helpers.keys()), {"NORMAL_COMMAND"})
        self.assertEqual(len(tool_prompts), 1)

    def test_load_helper_functions_handles_missing_docstring(self):
        """Test handling of functions without docstrings."""
        helper_content = '''
def handle_no_docstring(args):
    return "no docstring"

def handle_with_docstring(args):
    """This has a docstring"""
    return "with docstring"

def handle_empty_docstring(args):
    """"""
    return "empty docstring"
'''
        self._create_helper_file("docstring_test.py", helper_content)
        
        helpers, tool_prompts = load_helper_functions(str(self.helpers_path))
        
        # All functions should be loaded
        expected_commands = {"NO_DOCSTRING", "WITH_DOCSTRING", "EMPTY_DOCSTRING"}
        self.assertEqual(set(helpers.keys()), expected_commands)
        
        # Only the function with a proper docstring should generate a tool prompt
        self.assertEqual(len(tool_prompts), 1)
        self.assertIn("[WITH_DOCSTRING args] – This has a docstring", tool_prompts)

    def test_load_helper_functions_creates_directory_if_missing(self):
        """Test that the function creates the helpers directory if it doesn't exist."""
        non_existent_path = Path(self.temp_dir) / "non_existent_helpers"
        
        # Ensure the directory doesn't exist
        self.assertFalse(non_existent_path.exists())
        
        helpers, tool_prompts = load_helper_functions(str(non_existent_path))
        
        # Directory should now exist
        self.assertTrue(non_existent_path.exists())
        self.assertTrue((non_existent_path / "__init__.py").exists())
        
        # Should return empty results since no helper files exist
        self.assertEqual(helpers, {})
        self.assertEqual(tool_prompts, [])

    def test_load_helper_functions_multiline_docstring(self):
        """Test handling of multiline docstrings (should use only first line)."""
        helper_content = '''
def handle_multiline_doc(args):
    """First line of docstring
    
    This is the second line with more details.
    And this is the third line.
    """
    return "multiline"
'''
        self._create_helper_file("multiline_doc.py", helper_content)
        
        helpers, tool_prompts = load_helper_functions(str(self.helpers_path))
        
        self.assertEqual(len(tool_prompts), 1)
        self.assertIn("[MULTILINE_DOC args] – First line of docstring", tool_prompts)

    def test_load_helper_functions_command_name_conversion(self):
        """Test that function names are correctly converted to command names."""
        helper_content = '''
def handle_load_file(args):
    """Load file command"""
    return "load_file"

def handle_fetch_url_content(args):
    """Fetch URL content command"""
    return "fetch_url_content"

def handle_simple(args):
    """Simple command"""
    return "simple"
'''
        self._create_helper_file("name_conversion.py", helper_content)
        
        helpers, tool_prompts = load_helper_functions(str(self.helpers_path))
        
        expected_commands = {"LOAD_FILE", "FETCH_URL_CONTENT", "SIMPLE"}
        self.assertEqual(set(helpers.keys()), expected_commands)


class TestQwenSessionChat(unittest.TestCase):
    """Test the QwenSession chat method functionality."""

    def setUp(self):
        """Set up test session and mock helper functions."""
        self.config = {
            "max_context_tokens": 120000,
            "max_new_tokens": 1000,
            "temperature": 0.1,
        }
        self.tool_prompts = [
            "[LOAD_FILE args] – Load a file from the filesystem",
            "[FETCH_URL args] – Fetch content from a URL"
        ]
        self.session = QwenSession(self.config, self.tool_prompts)
        
        # Mock helper functions
        self.helper_functions = {
            "LOAD_FILE": Mock(return_value="[file: test.py]\n```python\nprint('Hello World')\n```"),
            "FETCH_URL": Mock(return_value="[URL: https://example.com]\n```\nExample content\n```")
        }

    @patch('qwen_cli.QwenSession._model')
    @patch('qwen_cli.QwenSession._tokenizer')
    def test_chat_simple_conversation(self, mock_tokenizer, mock_model):
        """Test a simple conversation without special commands."""
        # Mock tokenizer
        mock_tokenizer.apply_chat_template.return_value = "mocked_formatted_text"
        mock_tokenizer.return_value = {"input_ids": MagicMock()}
        mock_tokenizer.decode.return_value = "Hello! How can I help you today?"
        
        # Mock model
        mock_model.device = "cpu"
        mock_model.generate.return_value = [MagicMock()]
        
        # Ensure model is considered loaded
        QwenSession._model = mock_model
        QwenSession._tokenizer = mock_tokenizer
        
        result = self.session.chat(
            "Hello, how are you?", 
            self.helper_functions, 
            stream=False
        )
        
        self.assertTrue(result)
        self.assertEqual(len(self.session.history), 3)  # system + user + assistant
        self.assertEqual(self.session.history[1]["role"], "user")
        self.assertEqual(self.session.history[1]["content"], "Hello, how are you?")
        self.assertEqual(self.session.history[2]["role"], "assistant")
        self.assertEqual(self.session.history[2]["content"], "Hello! How can I help you today?")

    @patch('qwen_cli.QwenSession._model')
    @patch('qwen_cli.QwenSession._tokenizer')
    def test_chat_with_load_file_command(self, mock_tokenizer, mock_model):
        """Test conversation where assistant requests to load a file."""
        # Mock tokenizer
        mock_tokenizer.apply_chat_template.return_value = "mocked_formatted_text"
        mock_tokenizer.return_value = {"input_ids": MagicMock()}
        
        # First response: assistant requests to load file
        # Second response: assistant continues after file is loaded
        mock_tokenizer.decode.side_effect = [
            "I'll help you analyze that file. [LOAD_FILE test.py]",
            "Based on the loaded file, I can see it's a simple Python script that prints 'Hello World'. The code is straightforward and follows good practices."
        ]
        
        # Mock model
        mock_model.device = "cpu"
        mock_model.generate.return_value = [MagicMock()]
        
        # Ensure model is considered loaded
        QwenSession._model = mock_model
        QwenSession._tokenizer = mock_tokenizer
        
        result = self.session.chat(
            "Can you analyze test.py for me?", 
            self.helper_functions, 
            stream=False
        )
        
        self.assertTrue(result)
        
        # Verify the helper function was called
        self.helper_functions["LOAD_FILE"].assert_called_once_with("test.py")
        
        # Check history structure: system + user + system (file content) + user (continue) + user (original prompt again) + assistant (final response)
        # The original assistant response with [LOAD_FILE] is NOT kept in history due to recursive call
        self.assertEqual(len(self.session.history), 6)
        self.assertEqual(self.session.history[1]["content"], "Can you analyze test.py for me?")
        self.assertEqual(self.session.history[2]["role"], "system")
        self.assertIn("[file: test.py]", self.session.history[2]["content"])
        self.assertEqual(self.session.history[3]["role"], "user")
        self.assertEqual(self.session.history[3]["content"], "Please continue the analysis using the loaded file.")
        self.assertEqual(self.session.history[4]["role"], "user")
        self.assertEqual(self.session.history[4]["content"], "Can you analyze test.py for me?")
        self.assertEqual(self.session.history[5]["role"], "assistant")
        self.assertIn("Hello World", self.session.history[5]["content"])

    @patch('qwen_cli.QwenSession._model')
    @patch('qwen_cli.QwenSession._tokenizer')
    def test_chat_with_fetch_url_command(self, mock_tokenizer, mock_model):
        """Test conversation where assistant requests to fetch a URL."""
        # Mock tokenizer
        mock_tokenizer.apply_chat_template.return_value = "mocked_formatted_text"
        mock_tokenizer.return_value = {"input_ids": MagicMock()}
        
        # First response: assistant requests to fetch URL
        # Second response: assistant continues after URL is fetched
        mock_tokenizer.decode.side_effect = [
            "I'll fetch that URL for you. [FETCH_URL https://example.com]",
            "Based on the fetched content, I can see it contains example information that demonstrates the basic structure."
        ]
        
        # Mock model
        mock_model.device = "cpu"
        mock_model.generate.return_value = [MagicMock()]
        
        # Ensure model is considered loaded
        QwenSession._model = mock_model
        QwenSession._tokenizer = mock_tokenizer
        
        result = self.session.chat(
            "Can you check what's at https://example.com?", 
            self.helper_functions, 
            stream=False
        )
        
        self.assertTrue(result)
        
        # Verify the helper function was called
        self.helper_functions["FETCH_URL"].assert_called_once_with("https://example.com")
        
        # Check history structure: system + user + system (URL content) + user (continue) + user (original prompt again) + assistant (final response)
        # The original assistant response with [FETCH_URL] is NOT kept in history due to recursive call
        self.assertEqual(len(self.session.history), 6)
        self.assertEqual(self.session.history[2]["role"], "system")
        self.assertIn("[URL: https://example.com]", self.session.history[2]["content"])
        self.assertEqual(self.session.history[3]["role"], "user")
        self.assertEqual(self.session.history[3]["content"], "Please continue the analysis using the loaded file.")
        self.assertEqual(self.session.history[4]["role"], "user")
        self.assertEqual(self.session.history[4]["content"], "Can you check what's at https://example.com?")

    @patch('qwen_cli.QwenSession._model')
    @patch('qwen_cli.QwenSession._tokenizer')
    def test_chat_with_multiple_commands(self, mock_tokenizer, mock_model):
        """Test conversation with multiple special commands in one response."""
        # Mock tokenizer
        mock_tokenizer.apply_chat_template.return_value = "mocked_formatted_text"
        mock_tokenizer.return_value = {"input_ids": MagicMock()}
        
        # Response with multiple commands (only first one should be processed)
        mock_tokenizer.decode.side_effect = [
            "I'll load the file first [LOAD_FILE test.py] and then fetch [FETCH_URL https://example.com]",
            "Now I have the file content to analyze."
        ]
        
        # Mock model
        mock_model.device = "cpu"
        mock_model.generate.return_value = [MagicMock()]
        
        # Ensure model is considered loaded
        QwenSession._model = mock_model
        QwenSession._tokenizer = mock_tokenizer
        
        result = self.session.chat(
            "Load test.py and check example.com", 
            self.helper_functions, 
            stream=False
        )
        
        self.assertTrue(result)
        
        # Only the first command should be processed
        self.helper_functions["LOAD_FILE"].assert_called_once_with("test.py")
        self.helper_functions["FETCH_URL"].assert_not_called()
        
        # Verify the history structure shows the recursive call happened
        self.assertEqual(len(self.session.history), 6)  # system + user + system (file) + user (continue) + user (original prompt again) + assistant

    @patch('qwen_cli.QwenSession._model')
    @patch('qwen_cli.QwenSession._tokenizer')
    def test_chat_with_failed_helper_function(self, mock_tokenizer, mock_model):
        """Test conversation where helper function returns None (failure)."""
        # Mock tokenizer
        mock_tokenizer.apply_chat_template.return_value = "mocked_formatted_text"
        mock_tokenizer.return_value = {"input_ids": MagicMock()}
        mock_tokenizer.decode.return_value = "I'll try to load that file. [LOAD_FILE nonexistent.py]"
        
        # Mock model
        mock_model.device = "cpu"
        mock_model.generate.return_value = [MagicMock()]
        
        # Ensure model is considered loaded
        QwenSession._model = mock_model
        QwenSession._tokenizer = mock_tokenizer
        
        # Mock helper function to return None (failure)
        self.helper_functions["LOAD_FILE"].return_value = None
        
        result = self.session.chat(
            "Load nonexistent.py", 
            self.helper_functions, 
            stream=False
        )
        
        self.assertTrue(result)
        
        # Verify the helper function was called
        self.helper_functions["LOAD_FILE"].assert_called_once_with("nonexistent.py")
        
        # Should not add system message or continue recursively when helper returns None
        self.assertEqual(len(self.session.history), 3)  # system + user + assistant only

    @patch('qwen_cli.QwenSession._model')
    @patch('qwen_cli.QwenSession._tokenizer')
    def test_chat_with_unknown_command(self, mock_tokenizer, mock_model):
        """Test conversation with unknown special command."""
        # Mock tokenizer
        mock_tokenizer.apply_chat_template.return_value = "mocked_formatted_text"
        mock_tokenizer.return_value = {"input_ids": MagicMock()}
        mock_tokenizer.decode.return_value = "I'll try to use an unknown command. [UNKNOWN_CMD test]"
        
        # Mock model
        mock_model.device = "cpu"
        mock_model.generate.return_value = [MagicMock()]
        
        # Ensure model is considered loaded
        QwenSession._model = mock_model
        QwenSession._tokenizer = mock_tokenizer
        
        result = self.session.chat(
            "Do something unknown", 
            self.helper_functions, 
            stream=False
        )
        
        self.assertTrue(result)
        
        # No helper functions should be called
        self.helper_functions["LOAD_FILE"].assert_not_called()
        self.helper_functions["FETCH_URL"].assert_not_called()
        
        # Should just add the response normally
        self.assertEqual(len(self.session.history), 3)  # system + user + assistant only

    @patch('qwen_cli.QwenSession._model')
    @patch('qwen_cli.QwenSession._tokenizer')
    def test_chat_error_handling(self, mock_tokenizer, mock_model):
        """Test chat method error handling."""
        # Mock tokenizer to raise an exception
        mock_tokenizer.apply_chat_template.side_effect = Exception("Tokenizer error")
        
        # Ensure model is considered loaded
        QwenSession._model = mock_model
        QwenSession._tokenizer = mock_tokenizer
        
        result = self.session.chat(
            "This should fail", 
            self.helper_functions, 
            stream=False
        )
        
        self.assertFalse(result)
        
        # History should still contain the user message
        self.assertEqual(len(self.session.history), 2)  # system + user only
        self.assertEqual(self.session.history[1]["content"], "This should fail")

    def test_parse_special_commands(self):
        """Test the parse_special_commands function."""
        text = "Here's some text [LOAD_FILE test.py] and more [FETCH_URL https://example.com] text."
        
        commands = parse_special_commands(text)
        
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0][0], "LOAD_FILE")
        self.assertEqual(commands[0][1], "test.py")
        self.assertEqual(commands[1][0], "FETCH_URL")
        self.assertEqual(commands[1][1], "https://example.com")

    def test_parse_special_commands_with_complex_args(self):
        """Test parsing special commands with complex arguments."""
        text = "Load this [LOAD_FILE /path/to/file with spaces.py] and fetch [FETCH_URL https://example.com/path?param=value]"
        
        commands = parse_special_commands(text)
        
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0][1], "/path/to/file with spaces.py")
        self.assertEqual(commands[1][1], "https://example.com/path?param=value")

    def test_chat_parameter_defaults(self):
        """Test that chat method uses proper parameter defaults."""
        # Test with custom config values
        custom_config = {
            "max_context_tokens": 50000,
            "max_new_tokens": 2000,
            "temperature": 0.5,
        }
        session = QwenSession(custom_config, self.tool_prompts)
        
        max_new_tokens, temperature = session._prepare_chat_parameters(None, None)
        
        self.assertEqual(max_new_tokens, 2000)
        self.assertEqual(temperature, 0.5)
        
        # Test with explicit parameters
        max_new_tokens, temperature = session._prepare_chat_parameters(1500, 0.8)
        
        self.assertEqual(max_new_tokens, 1500)
        self.assertEqual(temperature, 0.8)


if __name__ == "__main__":
    unittest.main()
