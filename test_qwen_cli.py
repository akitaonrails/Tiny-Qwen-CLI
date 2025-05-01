#!/usr/bin/env python3
"""
Unit tests for qwen_cli.py, focusing on the load_helper_functions functionality.
"""

import unittest
from unittest.mock import patch, Mock, MagicMock
import sys
from pathlib import Path
import importlib.util
import types
import logging
from io import StringIO

# Import the functions to test
from qwen_cli import (
    load_helper_functions,
    _ensure_helpers_directory_exists,
    _find_python_modules,
    _load_python_module,
    _extract_handlers_from_module,
    _create_tool_prompt,
    QwenSession,
    parse_special_commands
)

# Configure logging for testing
logging.basicConfig(level=logging.DEBUG, stream=StringIO())


class TestLoadHelperFunctions(unittest.TestCase):
    """Test cases for the helper function loading mechanism."""

    def setUp(self):
        """Set up test environment."""
        # Create a patch for Path operations to avoid filesystem interactions
        self.path_patcher = patch('qwen_cli.Path')
        self.mock_path_class = self.path_patcher.start()
        
        # Create mock path objects for the test directories and files
        self.mock_helpers_dir = MagicMock(spec=Path)
        self.mock_path_class.return_value = self.mock_helpers_dir
        
        # Set up exists() to return True by default
        self.mock_helpers_dir.exists.return_value = True
        
        # Configure the mock path's glob method to return a list of mock paths
        self.mock_py_files = [
            Mock(spec=Path, stem='load_file'),
            Mock(spec=Path, stem='fetch_url'),
            Mock(spec=Path, stem='__init__')  # Should be ignored
        ]
        self.mock_helpers_dir.glob.return_value = self.mock_py_files

    def tearDown(self):
        """Clean up after tests."""
        self.path_patcher.stop()
        
    def test_ensure_helpers_directory_exists(self):
        """Test that the helpers directory is created if it doesn't exist."""
        # Set up the mock to indicate directory doesn't exist
        self.mock_helpers_dir.exists.return_value = False
        
        # Create a child path for __init__.py
        mock_init_path = MagicMock(spec=Path)
        # Use / operator which calls __truediv__ internally
        self.mock_helpers_dir.__truediv__.return_value = mock_init_path
        
        result = _ensure_helpers_directory_exists('helper_functions')
        
        # Verify directory was created
        self.mock_helpers_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        # Verify __init__.py was created
        self.mock_helpers_dir.__truediv__.assert_called_with('__init__.py')
        mock_init_path.touch.assert_called_once()
        
    def test_find_python_modules(self):
        """Test finding Python modules in the helpers directory."""
        # Mock the _load_python_module function
        with patch('qwen_cli._load_python_module') as mock_load_module:
            # Make sure __init__.py is filtered out before trying to load it
            mock_load_module.side_effect = [Mock(), Mock()]  # Only two successful loads
            
            modules = _find_python_modules(self.mock_helpers_dir)
            
            # Should find two modules (ignoring __init__.py)
            self.assertEqual(len(modules), 2)
            
            # Verify correct calls to load each non-init module
            self.assertEqual(mock_load_module.call_count, 2)

    def test_load_python_module(self):
        """Test loading a Python module from a file path."""
        mock_file_path = Mock(spec=Path)
        
        # Create a mock for spec_from_file_location
        with patch('importlib.util.spec_from_file_location') as mock_spec_from_file:
            mock_spec = Mock()
            mock_loader = Mock()
            mock_spec.loader = mock_loader
            mock_spec_from_file.return_value = mock_spec
            
            # Create a mock for module_from_spec
            with patch('importlib.util.module_from_spec') as mock_module_from_spec:
                mock_module = Mock()
                mock_module_from_spec.return_value = mock_module
                
                # Test successful module loading
                result = _load_python_module('test_module', mock_file_path)
                
                # Verify the correct calls were made
                mock_spec_from_file.assert_called_once_with('test_module', mock_file_path)
                mock_module_from_spec.assert_called_once_with(mock_spec)
                mock_loader.exec_module.assert_called_once_with(mock_module)
                self.assertEqual(result, mock_module)
                
            # Test exception handling
            with patch('importlib.util.module_from_spec') as mock_module_from_spec:
                mock_module_from_spec.side_effect = Exception("Module load error")
                
                result = _load_python_module('broken_module', mock_file_path)
                
                # Should return None on error
                self.assertIsNone(result)

    def test_extract_handlers_from_module(self):
        """Test extracting handler functions from a module."""
        # Create a mock module with some handler functions
        mock_module = types.ModuleType('test_module')
        
        # Add some handler functions
        def handle_load_file(filepath):
            """Load a file from the filesystem."""
            pass
        
        def handle_fetch_url(url):
            """Fetch content from a URL."""
            pass
        
        # Add a non-handler function
        def some_other_function():
            pass
        
        # Add the functions to the mock module
        mock_module.handle_load_file = handle_load_file
        mock_module.handle_fetch_url = handle_fetch_url
        mock_module.some_other_function = some_other_function
        
        # Extract handlers
        handlers = _extract_handlers_from_module(mock_module)
        
        # Should find two handlers
        self.assertEqual(len(handlers), 2)
        
        # Check handler names
        handler_names = [name for name, _ in handlers]
        self.assertIn('LOAD_FILE', handler_names)
        self.assertIn('FETCH_URL', handler_names)
        
        # Check handler functions
        handler_funcs = [func for _, func in handlers]
        self.assertIn(handle_load_file, handler_funcs)
        self.assertIn(handle_fetch_url, handler_funcs)

    def test_create_tool_prompt(self):
        """Test creating tool prompts from handler functions."""
        # Create a function with a docstring
        def handle_test(args):
            """This is a test handler function."""
            pass
        
        # Create a function without a docstring
        def handle_no_doc(args):
            pass
        
        # Test with docstring
        prompt = _create_tool_prompt('TEST', handle_test)
        self.assertEqual(prompt, '[TEST args] – This is a test handler function.')
        
        # Test without docstring
        prompt = _create_tool_prompt('NO_DOC', handle_no_doc)
        self.assertIsNone(prompt)

    def test_load_helper_functions_integration(self):
        """Integration test for the load_helper_functions function."""
        # Create mock modules
        mock_load_file_module = types.ModuleType('load_file')
        mock_fetch_url_module = types.ModuleType('fetch_url')
        
        # Create handler functions with docstrings
        def handle_load_file(filepath):
            """Load a file from the filesystem."""
            return f"Content of {filepath}"
        
        def handle_fetch_url(url):
            """Fetch content from a URL."""
            return f"Content from {url}"
        
        # Add handlers to modules
        mock_load_file_module.handle_load_file = handle_load_file
        mock_fetch_url_module.handle_fetch_url = handle_fetch_url
        
        # Set up the module loading to return our mock modules
        with patch('qwen_cli._ensure_helpers_directory_exists') as mock_ensure_dir:
            mock_ensure_dir.return_value = self.mock_helpers_dir
            
            with patch('qwen_cli._find_python_modules') as mock_find_modules:
                mock_find_modules.return_value = [mock_load_file_module, mock_fetch_url_module]
                
                # Call the function
                helpers, tool_prompts = load_helper_functions('helper_functions')
                
                # Check the results
                self.assertEqual(len(helpers), 2)
                self.assertIn('LOAD_FILE', helpers)
                self.assertIn('FETCH_URL', helpers)
                self.assertEqual(helpers['LOAD_FILE'], handle_load_file)
                self.assertEqual(helpers['FETCH_URL'], handle_fetch_url)
                
                # Check tool prompts
                self.assertEqual(len(tool_prompts), 2)
                self.assertIn('[LOAD_FILE args] – Load a file from the filesystem.', tool_prompts)
                self.assertIn('[FETCH_URL args] – Fetch content from a URL.', tool_prompts)

    def test_load_helper_functions_empty_directory(self):
        """Test load_helper_functions with an empty directory."""
        # Set up mock to return empty list for glob
        self.mock_helpers_dir.glob.return_value = []
        
        # Mock the find_python_modules to return empty list
        with patch('qwen_cli._find_python_modules') as mock_find_modules:
            mock_find_modules.return_value = []
            
            helpers, tool_prompts = load_helper_functions('helper_functions')
            
            # Should return empty collections
            self.assertEqual(len(helpers), 0)
            self.assertEqual(len(tool_prompts), 0)

    def test_load_helper_functions_with_real_modules(self):
        """Test load_helper_functions with more realistic module mocking."""
        # Create better mocks for file paths using MagicMock which handles special methods better
        mock_load_file_path = MagicMock(spec=Path)
        mock_load_file_path.stem = 'load_file'
        # Use __str__ method directly instead of trying to configure return_value
        mock_load_file_path.__str__.return_value = '/mock/helper_functions/load_file.py'
        
        mock_fetch_url_path = MagicMock(spec=Path)
        mock_fetch_url_path.stem = 'fetch_url'
        mock_fetch_url_path.__str__.return_value = '/mock/helper_functions/fetch_url.py'
        
        # Configure the mock_helpers_dir.glob to return these paths
        mock_helpers_dir = MagicMock(spec=Path)
        mock_helpers_dir.glob.return_value = [mock_load_file_path, mock_fetch_url_path]
        
        # Create handler functions with proper docstrings
        def handle_load_file(filepath):
            """Load a file from the filesystem."""
            return f"Processed {filepath} with handle_load_file"
        
        def handle_fetch_url(url):
            """Fetch content from a URL."""
            return f"Processed {url} with handle_fetch_url"
        
        # Create modules with the handler functions
        mock_load_file_module = types.ModuleType('load_file')
        mock_load_file_module.handle_load_file = handle_load_file
        
        mock_fetch_url_module = types.ModuleType('fetch_url')
        mock_fetch_url_module.handle_fetch_url = handle_fetch_url
        
        # Set up the test with complete mocking
        with patch('qwen_cli._ensure_helpers_directory_exists') as mock_ensure_dir:
            mock_ensure_dir.return_value = mock_helpers_dir
            
            with patch('qwen_cli._find_python_modules') as mock_find_modules:
                mock_find_modules.return_value = [mock_load_file_module, mock_fetch_url_module]
                
                # Call load_helper_functions
                helpers, tool_prompts = load_helper_functions('helper_functions')
                
                # Check the results
                self.assertEqual(len(helpers), 2)
                self.assertIn('LOAD_FILE', helpers)
                self.assertIn('FETCH_URL', helpers)
                
                # Test that the helpers actually work
                load_result = helpers['LOAD_FILE']('test.txt')
                self.assertEqual(load_result, "Processed test.txt with handle_load_file")
                
                fetch_result = helpers['FETCH_URL']('http://example.com')
                self.assertEqual(fetch_result, "Processed http://example.com with handle_fetch_url")
                
                # Check tool prompts
                self.assertEqual(len(tool_prompts), 2)
                self.assertIn('[LOAD_FILE args] – Load a file from the filesystem.', tool_prompts)
                self.assertIn('[FETCH_URL args] – Fetch content from a URL.', tool_prompts)


class TestQwenSessionChat(unittest.TestCase):
    """Test cases for the QwenSession chat method."""
    
    def setUp(self):
        """Set up test environment for QwenSession tests."""
        # Create basic configuration
        self.config = {
            "max_new_tokens": 100,
            "temperature": 0.1,
            "max_context_tokens": 1000
        }
        
        # Create tool prompts
        self.tool_prompts = [
            "[LOAD_FILE args] – Load a file from the filesystem.",
            "[FETCH_URL args] – Fetch content from a URL."
        ]
        
        # Create helper functions
        self.helper_functions = {
            "LOAD_FILE": self._mock_load_file,
            "FETCH_URL": self._mock_fetch_url
        }
        
        # Set up mocks for the model and tokenizer
        self.model_patcher = patch('qwen_cli.QwenSession._model', create=True)
        self.tokenizer_patcher = patch('qwen_cli.QwenSession._tokenizer', create=True)
        
        self.mock_model = self.model_patcher.start()
        self.mock_tokenizer = self.tokenizer_patcher.start()
        
        # Configure tokenizer mock
        self.mock_tokenizer.apply_chat_template.return_value = "formatted_chat_history"
        self.mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        self.mock_tokenizer.decode.return_value = "Model response"
        
        # Create a session with our mocked dependencies
        self.session = QwenSession(self.config, self.tool_prompts)
        
        # Patch process_commands to better control the flow
        self.process_commands_patcher = patch.object(self.session, '_process_commands')
        self.mock_process_commands = self.process_commands_patcher.start()
        self.mock_process_commands.return_value = (False, None)
        
        # Instead of patching _prepare_inputs and _generate_response, we'll patch at a higher level
        # Just patch the _model.generate part so we can control the responses
        self._patch_generate = patch.object(QwenSession._model, 'generate')
        self.mock_generate = self._patch_generate.start()
        
        # Configure _tokenizer.decode to return our specified responses
        self._patch_decode = patch.object(QwenSession._tokenizer, 'decode')
        self.mock_decode = self._patch_decode.start()
        
    def tearDown(self):
        """Clean up after tests."""
        self.model_patcher.stop()
        self.tokenizer_patcher.stop()
        self._patch_generate.stop()
        self._patch_decode.stop()
        self.process_commands_patcher.stop()
    
    def _mock_load_file(self, filepath):
        """Mock implementation of LOAD_FILE helper."""
        return f"[file: {filepath}]\n```python\nprint('This is a mock file content')\n```"
    
    def _mock_fetch_url(self, url):
        """Mock implementation of FETCH_URL helper."""
        return f"[URL: {url}]\n```\nThis is mock content from {url}\n```"
    
    def test_chat_basic_conversation(self):
        """Test basic conversation without special commands."""
        # Configure mock to return a basic response
        self.mock_generate.return_value = MagicMock()
        self.mock_decode.return_value = "This is a simple response from the model."
        
        # Call the chat method
        result = self.session.chat("Hello, how are you?", self.helper_functions)
        
        # Verify the result
        self.assertTrue(result)
        
        # Verify the conversation history was updated
        self.assertEqual(len(self.session.history), 3)  # System prompt + user message + assistant response
        self.assertEqual(self.session.history[-2]["role"], "user")
        self.assertEqual(self.session.history[-2]["content"], "Hello, how are you?")
        self.assertEqual(self.session.history[-1]["role"], "assistant")
        self.assertEqual(self.session.history[-1]["content"], "This is a simple response from the model.")
    
    def test_chat_with_load_file_command(self):
        """Test conversation with LOAD_FILE command in the response."""
        # Configure mocks for a response with LOAD_FILE command
        self.mock_decode.return_value = "I'll analyze that file for you. [LOAD_FILE test.py]"
        
        # Configure process_commands to simulate loading a file and returning for continuation
        self.mock_process_commands.side_effect = [
            (True, self._mock_load_file("test.py")),  # First call - process LOAD_FILE command
            (False, None)  # Second call - no commands in the follow-up response
        ]
        
        # Set up the second response after file is loaded
        self.mock_decode.side_effect = [
            "I'll analyze that file for you. [LOAD_FILE test.py]",
            "The file contains a simple print statement."
        ]
        
        # Reset the history to ensure we're starting fresh
        self.session.history = [{"role": "system", "content": "System prompt"}]
        
        # Call the chat method
        result = self.session.chat("Can you check my test.py file?", self.helper_functions)
        
        # Verify the result
        self.assertTrue(result)
        
        # Our history should have:
        # 1. System prompt
        # 2. User message
        # 3. System message with file content
        # 4. User continuation message
        # 5. Assistant final response
        self.assertEqual(len(self.session.history), 6)
        
        # Check that the second response was added to history
        self.assertEqual(self.session.history[-1]["role"], "assistant")
        self.assertEqual(self.session.history[-1]["content"], "The file contains a simple print statement.")
    
    def test_chat_with_fetch_url_command(self):
        """Test conversation with FETCH_URL command in the response."""
        # Configure mock for a response with FETCH_URL command
        self.mock_decode.side_effect = [
            "Let me check that website for you. [FETCH_URL https://example.com]",
            "The website is a simple example page."
        ]
        
        # Configure process_commands to simulate fetching a URL and returning for continuation
        self.mock_process_commands.side_effect = [
            (True, self._mock_fetch_url("https://example.com")),  # First call - process FETCH_URL command
            (False, None)  # Second call - no commands in the follow-up response
        ]
        
        # Reset the history to ensure we're starting fresh
        self.session.history = [{"role": "system", "content": "System prompt"}]
        
        # Call the chat method
        result = self.session.chat("Can you check the content of example.com?", self.helper_functions)
        
        # Verify the result
        self.assertTrue(result)
        
        # Our history should have 6 entries as in the load file test
        self.assertEqual(len(self.session.history), 6)
        
        # Check the final response
        self.assertEqual(self.session.history[-1]["role"], "assistant")
        self.assertEqual(self.session.history[-1]["content"], "The website is a simple example page.")
    
    def test_chat_with_error_handling(self):
        """Test that the chat method handles errors gracefully."""
        # Configure generate to raise an exception
        self.mock_generate.side_effect = Exception("Model error")
        
        # Reset the history to ensure we're starting fresh
        self.session.history = [{"role": "system", "content": "System prompt"}]
        
        # Call the chat method
        result = self.session.chat("Hello", self.helper_functions)
        
        # Verify the method returned False due to the error
        self.assertFalse(result)
        
        # Verify the conversation history was not updated with an assistant response
        # We should still have user message added
        self.assertEqual(len(self.session.history), 2)  # System prompt + user message only
        self.assertEqual(self.session.history[1]["role"], "user")
        self.assertEqual(self.session.history[1]["content"], "Hello")
    
    def test_chat_with_multiple_commands(self):
        """Test conversation with multiple commands in a response."""
        # Configure decode to simulate multiple commands
        self.mock_decode.side_effect = [
            "I'll analyze both files. [LOAD_FILE test1.py] Also, let me check [FETCH_URL https://example.com]",
            "Here's the analysis of the first file.",
            "And here's information about the website."
        ]
        
        # Configure process_commands to handle both commands in sequence
        self.mock_process_commands.side_effect = [
            (True, self._mock_load_file("test1.py")),         # First call - process LOAD_FILE
            (True, self._mock_fetch_url("https://example.com")), # Second call - process FETCH_URL
            (False, None)                                      # Third call - no more commands
        ]
        
        # Reset the history to ensure we're starting fresh
        self.session.history = [{"role": "system", "content": "System prompt"}]
        
        # Call the chat method
        result = self.session.chat("Can you check my test1.py file and also example.com?", self.helper_functions)
        
        # Verify the result
        self.assertTrue(result)
        
        # Check the final response
        self.assertEqual(self.session.history[-1]["role"], "assistant")
        self.assertEqual(self.session.history[-1]["content"], "And here's information about the website.")


if __name__ == '__main__':
    unittest.main()
