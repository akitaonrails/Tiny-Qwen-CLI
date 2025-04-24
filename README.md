# Qwen CLI (for Education Purposes ONLY)

A command-line interface for interacting with Qwen 2.5 Coder models with special focus on code-aware conversations.

This is not feature-complete, and I made it as an educational platform to understand how Web Chat A.I. like ChatGPT, Claude or even tools like Cursor or Windsurf work.

For a proper command-line tool client that integrates with many LLMs, choose [AIDER](https://aider.chat/).

## Features

- Interactive chat with Qwen 2.5 Coder model
- Code-aware conversations with file loading capabilities
- Context limit management
- Dynamic helper function loading system
- Docker container support

## Setup

### Requirements for development

- Python 3.8+
- PyTorch
- Transformers
- Docker (optional, for containerized usage)

### Directory Structure

```
.
├── qwen_cli.py        # Main CLI implementation
├── Dockerfile         # Docker configuration
├── dev.sh             # Development script for Docker
├── helpers/           # Helper functions directory
│   ├── __init__.py
│   ├── load_file.py   # File loading helper
│   ├── fetch_url.py   # URL fetching helper
│   └── batch_load.py  # Batch file loading helper
└── README.md          # This file
```

## Usage

### Basic Commands

```bash
# Start a new session
./qwen_cli.py

# Load a specific file at launch
./qwen_cli.py load path/to/file.py

# Batch load all Python files from the current directory
./qwen_cli.py batch_load . "*.py" 

# Hide intermediate reasoning in responses (not sure if chain of thought is really working)
./qwen_cli.py --hide-reasoning
```

### Docker Usage

```bash
# Start a containerized session
./dev.sh

# Load a file in the containerized session
./dev.sh load path/to/file.py

# Run a specific command
./dev.sh "What does this code do?"
```

## Special Chat Commands

During chat, the Qwen model can trigger special commands that will be processed by the CLI:

- `[LOAD_FILE path/to/file.py]` - Load a specific file into the conversation context
- `[FETCH_URL https://example.com]` - Fetch content from a URL and add it to context
- `[BATCH_LOAD directory pattern]` - Load all matching files from a directory

You don't need to type those command, just chat something like "load and analyze my file ./code.py, and find bugs."

To end a chat session, type `bye` and you'll be prompted to save the session history.

## Helper Functions

The CLI dynamically loads helper functions from the `helpers` directory. Each helper should:

1. Be in its own Python file
2. Contain a function named `handle_X` where X is the command name in uppercase
3. Take a single string argument and return a formatted string or None

Example:
```python
# helpers/my_helper.py
def handle_my_command(arguments):
    # Process arguments and return a result
    return f"[result: {arguments}]"
```

This would be accessible in chat via `[MY_COMMAND some arguments]`.

## Configuration

The CLI uses a configuration file located at `~/.config/qwen_codes/config.json`. Default settings include:

- Model repository: `Qwen/Qwen2.5-Coder-14B-Instruct`
- Quantization: `16bit` (options: `4bit`, `8bit`, `16bit`)
- Maximum context tokens: `12000` (try larger like 100,000 to be able to load large source code files)
- Maximum new tokens: `1024` (try larger, like 10,000 so it can answer large source code answers)
- Temperature: `0.1` (the smaller, the more exact, the larger, the more "creative")

## Extending

To add new helper functions:

1. Create a new Python file in the `helpers` directory
2. Implement a function named `handle_X` where X is the command name
3. The function will be automatically available via `[X arguments]` in chat

## Example Conversation

```
>>> What does the load_file.py file do?

This file contains a helper function that allows me to load file content into our conversation. Would you like me to look at it?

>>> Yes, please load it.

[LOAD_FILE helpers/load_file.py]
Added LOAD_FILE content to context: helpers/load_file.py

The load_file.py module contains a helper function that handles loading files into our conversation. The main functionality:

1. It defines a function called `handle_load_file` that takes a filepath as input
2. Validates that the file exists and is readable
3. Reads the content of the file
4. Determines the appropriate syntax highlighting language based on file extension
5. Returns the content formatted for inclusion in the conversation context

This helper is triggered when I suggest loading a file with the [LOAD_FILE path] command pattern in my responses.

>>> bye

Goodbye! Exiting chat session.
```
