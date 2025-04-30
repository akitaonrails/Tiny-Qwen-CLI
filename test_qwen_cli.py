import unittest
from unittest.mock import patch, MagicMock, call
import sys
from pathlib import Path
import logging
import importlib.util

# Ensure the main script's directory is in the path for imports if running tests directly
# This might be needed depending on how tests are run. Adjust if necessary.
# sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the function to test AFTER potentially modifying sys.path
from qwen_cli import load_helper_functions

# Suppress logging during tests unless specifically testing log output
# We still patch specific loggers below to check calls were made
logging.disable(logging.CRITICAL)

# --- Mock Helper Functions ---
# Define mock functions outside the test methods to represent potential helpers

def mock_handle_load_file(filepath):
    """Loads the content of a specified file."""
    pass # Actual implementation not needed for testing load_helper_functions

def mock_handle_fetch_url(url):
    """Fetches content from a given URL."""
    pass

def mock_handle_no_docstring(arg):
    pass # No docstring provided

def mock_handle_multi_line_docstring(arg):
    """
    This is the first line.
    This is the second line and should be ignored for the prompt.
    """
    pass

# --- Test Class ---

class TestLoadHelperFunctions(unittest.TestCase):

    @patch('qwen_cli.Path.glob')
    @patch('qwen_cli.Path.is_dir')
    @patch('qwen_cli.Path.touch')
    @patch('qwen_cli.importlib.util.spec_from_file_location')
    @patch('qwen_cli.importlib.util.module_from_spec')
    @patch('qwen_cli.sys.modules', {}) # Isolate sys.modules for each test
    def test_load_successful(self, mock_module_from_spec, mock_spec_from_file_loc, mock_touch, mock_is_dir, mock_glob):
        """Test loading valid helper functions successfully."""
        mock_helpers_dir = "/fake/helpers"
        mock_is_dir.return_value = True

        # Define mock Python files Path objects
        mock_file1_path = MagicMock(spec=Path)
        mock_file1_path.name = "load_file.py"
        mock_file1_path.stem = "load_file"

        mock_file2_path = MagicMock(spec=Path)
        mock_file2_path.name = "fetch_url.py"
        mock_file2_path.stem = "fetch_url"

        mock_file3_path = MagicMock(spec=Path)
        mock_file3_path.name = "no_doc.py"
        mock_file3_path.stem = "no_doc"

        mock_file4_path = MagicMock(spec=Path)
        mock_file4_path.name = "multi_line.py"
        mock_file4_path.stem = "multi_line"

        mock_init_path = MagicMock(spec=Path)
        mock_init_path.name = "__init__.py"
        mock_init_path.stem = "__init__"

        mock_glob.return_value = [
            mock_file1_path,
            mock_file2_path,
            mock_file3_path,
            mock_file4_path,
            mock_init_path,
        ]

        # --- Mock importlib behavior ---
        mock_spec = MagicMock()
        mock_loader = MagicMock()
        # exec_module needs to exist but doesn't need to *do* anything here
        mock_loader.exec_module = MagicMock()
        mock_spec.loader = mock_loader

        # Map file paths to mock specs
        mock_spec_from_file_loc.side_effect = lambda name, path: mock_spec if path not in [mock_init_path] else None

        # Create distinct mock modules and PRE-ASSIGN the handlers
        mock_module1 = MagicMock(__name__="qwen_cli.helpers.load_file")
        mock_module1.handle_load_file = mock_handle_load_file

        mock_module2 = MagicMock(__name__="qwen_cli.helpers.fetch_url")
        mock_module2.handle_fetch_url = mock_handle_fetch_url

        mock_module3 = MagicMock(__name__="qwen_cli.helpers.no_doc")
        mock_module3.handle_no_docstring = mock_handle_no_docstring

        mock_module4 = MagicMock(__name__="qwen_cli.helpers.multi_line")
        mock_module4.handle_multi_line_docstring = mock_handle_multi_line_docstring

        # Configure module_from_spec to return the correct pre-configured mock module
        def module_side_effect(spec):
            if spec.name == "qwen_cli.helpers.load_file": return mock_module1
            if spec.name == "qwen_cli.helpers.fetch_url": return mock_module2
            if spec.name == "qwen_cli.helpers.no_doc": return mock_module3
            if spec.name == "qwen_cli.helpers.multi_line": return mock_module4
            return MagicMock() # Default

        mock_module_from_spec.side_effect = module_side_effect

        # --- Call the function ---
        helpers, tool_prompts = load_helper_functions(mock_helpers_dir)

        # --- Assertions ---
        # Check __init__.py was touched
        mock_touch.assert_called_once_with(exist_ok=True)

        # Check helpers dictionary
        self.assertEqual(len(helpers), 4)
        self.assertIn("LOAD_FILE", helpers)
        self.assertIs(helpers["LOAD_FILE"], mock_handle_load_file)
        self.assertIn("FETCH_URL", helpers)
        self.assertIs(helpers["FETCH_URL"], mock_handle_fetch_url)
        self.assertIn("NO_DOCSTRING", helpers)
        self.assertIs(helpers["NO_DOCSTRING"], mock_handle_no_docstring)
        self.assertIn("MULTI_LINE_DOCSTRING", helpers)
        self.assertIs(helpers["MULTI_LINE_DOCSTRING"], mock_handle_multi_line_docstring)

        # Check tool prompts list (order might vary based on glob, so check content)
        self.assertEqual(len(tool_prompts), 4)
        expected_prompts = [
            "[LOAD_FILE args] – Loads the content of a specified file.",
            "[FETCH_URL args] – Fetches content from a given URL.",
            "[NO_DOCSTRING args] – Executes the NO_DOCSTRING action.", # Default prompt
            "[MULTI_LINE_DOCSTRING args] – This is the first line.", # Only first line
        ]
        self.assertCountEqual(tool_prompts, expected_prompts) # Checks elements regardless of order

        # Check importlib calls
        expected_spec_calls = [
            call("qwen_cli.helpers.load_file", mock_file1_path),
            call("qwen_cli.helpers.fetch_url", mock_file2_path),
            call("qwen_cli.helpers.no_doc", mock_file3_path),
            call("qwen_cli.helpers.multi_line", mock_file4_path),
            # __init__.py is skipped before spec_from_file_location
        ]
        mock_spec_from_file_loc.assert_has_calls(expected_spec_calls, any_order=True)
        self.assertEqual(mock_spec_from_file_loc.call_count, 4)

        # Check exec_module was called for each valid spec
        self.assertEqual(mock_loader.exec_module.call_count, 4)
        # Verify exec_module was called with the correct mock modules
        mock_loader.exec_module.assert_has_calls([
            call(mock_module1), call(mock_module2), call(mock_module3), call(mock_module4)
        ], any_order=True)


    @patch('qwen_cli.Path.is_dir')
    @patch('qwen_cli.logger.warning') # Patch logger instance method
    def test_load_nonexistent_dir(self, mock_log_warning, mock_is_dir):
        """Test loading when the helpers directory doesn't exist."""
        mock_helpers_dir = "/fake/nonexistent"
        mock_is_dir.return_value = False

        helpers, tool_prompts = load_helper_functions(mock_helpers_dir)

        self.assertEqual(helpers, {})
        self.assertEqual(tool_prompts, [])
        mock_log_warning.assert_called_once_with(
            f"Helpers directory '{mock_helpers_dir}' not found or not a directory. No helpers loaded."
        )

    @patch('qwen_cli.Path.glob')
    @patch('qwen_cli.Path.is_dir')
    @patch('qwen_cli.Path.touch')
    @patch('qwen_cli.importlib.util.spec_from_file_location')
    @patch('qwen_cli.importlib.util.module_from_spec')
    @patch('qwen_cli.logger.error') # Patch logger instance method
    @patch('qwen_cli.sys.modules', {})
    def test_load_import_error(self, mock_log_error, mock_module_from_spec, mock_spec_from_file_loc, mock_touch, mock_is_dir, mock_glob):
        """Test loading when a helper module raises an error during import."""
        mock_helpers_dir = "/fake/helpers_with_error"
        mock_is_dir.return_value = True

        mock_error_path = MagicMock(spec=Path)
        mock_error_path.name = "error_helper.py"
        mock_error_path.stem = "error_helper"

        mock_good_path = MagicMock(spec=Path)
        mock_good_path.name = "good_helper.py"
        mock_good_path.stem = "good_helper"

        mock_glob.return_value = [mock_error_path, mock_good_path]

        # Mock spec/module loading
        mock_spec = MagicMock()
        mock_loader = MagicMock()
        mock_spec.loader = mock_loader

        # Define mock handle function for the good helper
        def mock_handle_good_helper(arg):
            """Good helper doc."""
            pass

        # Make exec_module raise an error ONLY for the error module.
        def exec_side_effect(module):
            if module.__name__ == "qwen_cli.helpers.error_helper":
                raise ImportError("Simulated import failure")
            # No need to assign attributes here anymore
            # elif module.__name__ == "qwen_cli.helpers.good_helper":
            #      pass

        mock_loader.exec_module.side_effect = exec_side_effect

        mock_spec_from_file_loc.return_value = mock_spec

        # Mock module creation, pre-assigning the handler for the good one
        mock_error_module = MagicMock(__name__="qwen_cli.helpers.error_helper")
        mock_good_module = MagicMock(__name__="qwen_cli.helpers.good_helper")
        mock_good_module.handle_good_helper = mock_handle_good_helper # Assign here

        def module_side_effect(spec):
             if spec.name == "qwen_cli.helpers.error_helper": return mock_error_module
             if spec.name == "qwen_cli.helpers.good_helper": return mock_good_module
             return MagicMock()
        mock_module_from_spec.side_effect = module_side_effect


        # --- Call the function ---
        helpers, tool_prompts = load_helper_functions(mock_helpers_dir)

        # --- Assertions ---
        # Only the good helper should be loaded
        self.assertEqual(len(helpers), 1)
        self.assertIn("GOOD_HELPER", helpers)
        self.assertIs(helpers["GOOD_HELPER"], mock_handle_good_helper) # Check it's the correct function
        self.assertEqual(len(tool_prompts), 1)
        self.assertIn("[GOOD_HELPER args] – Good helper doc.", tool_prompts)

        # Check that an error was logged for the failed import
        mock_log_error.assert_called()
        found_error_log = False
        for log_call in mock_log_error.call_args_list:
            # Check if the first argument of the call contains the expected string
            if isinstance(log_call.args, tuple) and len(log_call.args) > 0 and \
               "Failed to load helper module error_helper" in log_call.args[0]:
                found_error_log = True
                break
            # Handle cases where args might be different or kwargs are used if necessary
        self.assertTrue(found_error_log, "Expected error log for 'error_helper' not found.")


    @patch('qwen_cli.Path.glob')
    @patch('qwen_cli.Path.is_dir')
    @patch('qwen_cli.Path.touch')
    @patch('qwen_cli.importlib.util.spec_from_file_location')
    @patch('qwen_cli.importlib.util.module_from_spec')
    @patch('qwen_cli.sys.modules', {})
    def test_load_no_handle_function(self, mock_module_from_spec, mock_spec_from_file_loc, mock_touch, mock_is_dir, mock_glob):
        """Test loading a file that doesn't contain a 'handle_' function."""
        mock_helpers_dir = "/fake/helpers_no_handle"
        mock_is_dir.return_value = True

        mock_no_handle_path = MagicMock(spec=Path)
        mock_no_handle_path.name = "no_handle.py"
        mock_no_handle_path.stem = "no_handle"

        mock_glob.return_value = [mock_no_handle_path]

        # Mock spec/module loading
        mock_spec = MagicMock()
        mock_loader = MagicMock()
        mock_loader.exec_module = MagicMock() # Simulate successful execution
        mock_spec.loader = mock_loader

        mock_spec_from_file_loc.return_value = mock_spec

        # Create a mock module *without* any handle_ function attribute
        mock_module = MagicMock(__name__="qwen_cli.helpers.no_handle")
        # Ensure it doesn't accidentally have a handle_ attribute
        self.assertFalse(hasattr(mock_module, 'handle_something'))

        mock_module_from_spec.return_value = mock_module

        # --- Call the function ---
        helpers, tool_prompts = load_helper_functions(mock_helpers_dir)

        # --- Assertions ---
        self.assertEqual(helpers, {})
        self.assertEqual(tool_prompts, [])
        # Ensure exec_module was still called
        mock_loader.exec_module.assert_called_once_with(mock_module)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
