import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from qwen_cli import QwenSession, load_helper_functions, build_system_prompt, parse_special_commands

class TestQwenSessionChat(unittest.TestCase):
    def setUp(self):
        # Minimal config so _prepare_chat works without trimming
        self.config = {
            'max_new_tokens': 5,
            'temperature': 0.1,
            'max_context_tokens': 100
        }
        self.session = QwenSession(self.config, tool_prompts=[])
        # Patch out the actual model/tokenizer-based generation
        self.generate_patcher = patch.object(QwenSession, '_generate_response_text')
        self.mock_generate = self.generate_patcher.start()
        self.addCleanup(self.generate_patcher.stop)

    def test_simple_chat(self):
        # Simulate a plain response with no special commands
        self.mock_generate.return_value = 'hi there'
        result = self.session.chat('hello', helper_functions={}, stream=False)

        self.assertTrue(result)
        # The user prompt should be appended
        self.assertEqual(self.session.history[-2], {
            'role': 'user',
            'content': 'hello'
        })
        # The assistant response should be appended
        self.assertEqual(self.session.history[-1], {
            'role': 'assistant',
            'content': 'hi there'
        })

    def test_special_command(self):
        # First return contains a special command, second return is the follow-up response
        self.mock_generate.side_effect = [
            'do it [CMD arg1]',
            'final response'
        ]
        helper = MagicMock(return_value='loaded content')
        helpers = {'CMD': helper}

        result = self.session.chat('cmd', helpers, stream=False)

        self.assertTrue(result)
        # Ensure the helper was called with the correct argument
        helper.assert_called_with('arg1')

        # After the special command, the system content and continue prompt should be appended
        expected_system = {
            'role': 'system',
            'content': 'loaded content'
        }
        expected_user_continue = {
            'role': 'user',
            'content': 'Please continue the analysis using the loaded file.'
        }
        # Finally, the assistant's follow-up response should be appended
        expected_assistant = {
            'role': 'assistant',
            'content': 'final response'
        }

        self.assertIn(expected_system, self.session.history)
        self.assertIn(expected_user_continue, self.session.history)
        self.assertEqual(self.session.history[-1], expected_assistant)

    def test_load_file_interaction(self):
        # Stub the LOAD_FILE helper to track invocation
        stubbed_helper = MagicMock(return_value="formatted file content")
        helpers = {"LOAD_FILE": stubbed_helper}
        # Create a temporary test file
        test_file = Path("temp_test.py")
        test_file.write_text("print('hello')")
        try:
            # Simulate model sending a LOAD_FILE command first, then an analysis response
            self.mock_generate.side_effect = [
                f"[LOAD_FILE {test_file}]",
                "analysis done"
            ]
            result = self.session.chat('analyze file', helpers, stream=False)
            self.assertTrue(result)
            # Ensure the helper was invoked with the correct file path
            stubbed_helper.assert_called_once_with(str(test_file))
            # Check that the formatted file content was appended as a system message
            self.assertIn(
                {'role': 'system', 'content': 'formatted file content'},
                self.session.history
            )
            # Check that the user is prompted to continue analysis
            self.assertIn(
                {'role': 'user', 'content': 'Please continue the analysis using the loaded file.'},
                self.session.history
            )
            # Final assistant analysis response should be appended
            self.assertEqual(
                self.session.history[-1],
                {'role': 'assistant', 'content': 'analysis done'}
            )
        finally:
            test_file.unlink()

    def test_load_helper_functions_docstrings(self):
        # Dynamically load all helper functions and their prompts
        helpers, tool_prompts = load_helper_functions("helper_functions")
        # Expect at least one helper loaded
        self.assertIsInstance(helpers, dict)
        self.assertTrue(len(helpers) > 0)
        # Prompts should be a list and at least as many as helpers
        self.assertIsInstance(tool_prompts, list)
        self.assertGreaterEqual(len(tool_prompts), len(helpers))
        # For each helper, its docstring first line should appear in its tool prompt
        for cmd, func in helpers.items():
            doc = func.__doc__ or ""
            self.assertTrue(doc.strip(), f"{cmd} helper has no docstring")
            first_line = doc.strip().splitlines()[0]
            matching = [tp for tp in tool_prompts if tp.startswith(f"[{cmd} ")]
            self.assertTrue(matching, f"No prompt found for {cmd}")
            self.assertIn(first_line, matching[0])

    def test_build_system_prompt_includes_tool_prompts(self):
        # Ensure system prompt includes all tool prompts
        _, tool_prompts = load_helper_functions("helper_functions")
        system_prompt = build_system_prompt(tool_prompts)
        for tp in tool_prompts:
            self.assertIn(tp, system_prompt)

    def test_parse_special_commands(self):
        response = "Hello [ONE arg1] and [TWO arg two]"
        cmds = parse_special_commands(response)
        # Expect two parsed commands
        self.assertEqual(len(cmds), 2)
        self.assertEqual(cmds[0][0], 'ONE')
        self.assertEqual(cmds[0][1], 'arg1')
        self.assertEqual(cmds[1][0], 'TWO')
        self.assertEqual(cmds[1][1], 'arg two')

    def test_handle_special_commands_called_during_chat(self):
        # Stub the CMD helper and simulate two-step response to prevent recursion
        stubbed_helper = MagicMock(return_value="handled content")
        helpers = {'CMD': stubbed_helper}
        # First generate returns a special command, second returns a normal response
        self.mock_generate.side_effect = ["[CMD myfile.txt]", "final response"]
        result = self.session.chat("trigger", helpers, stream=False)
        self.assertTrue(result)
        # Verify the helper was called exactly once with correct argument
        stubbed_helper.assert_called_once_with("myfile.txt")
        # Verify the assistant appended the formatted content and final response
        self.assertIn({'role': 'system', 'content': 'handled content'}, self.session.history)
        self.assertIn({'role': 'user', 'content': 'Please continue the analysis using the loaded file.'}, self.session.history)
        self.assertEqual(self.session.history[-1], {'role': 'assistant', 'content': 'final response'})

if __name__ == '__main__':
    unittest.main()
