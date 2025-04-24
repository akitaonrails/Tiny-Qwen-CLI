"""
Helper function for batch loading files in Qwen CLI.
This function will be dynamically loaded by the CLI.
"""

import logging
from pathlib import Path
import os
from helper_functions.utils import get_language_from_extension

logger = logging.getLogger("qwen_cli.helpers.batch_load")

def handle_batch_load(args_string):
    """
    If the user asks to read, load or analyze all the files from a relative path, such as ./src or similar,
    write down this command: `[BATCH_LOAD directory pattern]` (e.g., `[BATCH_LOAD ./docs *.txt]`).

    This will load all files matching a pattern from a directory.
    """
    logger.debug(f"Batch loading files with args: {args_string}")

    try:
        # Parse arguments
        args = args_string.strip().split()
        if len(args) >= 1:
            directory = args[0]
        else:
            directory = "."
        if len(args) >= 2:
            pattern = args[1]
        else:
            pattern = "*.py"

        # 1. Start with a Path object
        dir_path = Path(directory)

        # 2. Log initial state
        logger.debug(f"Initial directory: {directory}, is_absolute: {dir_path.is_absolute()}")

        # 3. If it's relative, try to make it absolute relative to /app
        if not dir_path.is_absolute():
            app_path = Path("/project")  # Docker mount point
            possible_path = app_path / dir_path
            logger.debug(f"Trying /project-relative path: {possible_path}")
            if possible_path.exists():
                dir_path = possible_path
                logger.debug(f"  Found: Using /project-relative path")
            else:
                # 4. If still not found, try relative to the current working directory
                cwd_path = Path.cwd()
                possible_path = cwd_path / dir_path
                logger.debug(f"Trying CWD-relative path: {possible_path}")
                if possible_path.exists():
                    dir_path = possible_path
                    logger.debug(f"  Found: Using CWD-relative path")
                else:
                    logger.debug("  Not found in /project or CWD")
        # 5. Log the final resolved path
        logger.debug(f"Resolved dir_path: {dir_path}")

        # 6. Final check: does the resolved path exist and is it a directory?
        if not dir_path.exists():
            logger.error(f"Directory not found: {dir_path} (original: {directory})")
            return f"Directory not found: {dir_path} (original: {directory})"

        if not dir_path.is_dir():
            logger.error(f"Path is not a directory: {dir_path} (original: {directory})")
            return f"Path is not a directory: {dir_path} (original: {directory})"

        # Find all matching files
        matching_files = list(dir_path.glob(pattern))
        if not matching_files:
            logger.warning(f"No files matching '{pattern}' found in {dir_path}")
            return f"No files matching '{pattern}' found in {dir_path}"

        # Configuration from environment variables (sensible defaults)
        max_file_size = int(os.getenv('MAX_FILE_SIZE', 100000))  # 100KB
        total_max_size = int(os.getenv('TOTAL_MAX_SIZE', 1000000))  # 1MB
        file_limit = int(os.getenv('FILE_LIMIT', 50))

        loaded_files = []
        total_size = 0

        for file_path in matching_files[:file_limit]:
            try:
                if file_path.is_file():  # Ensure it's a file
                    file_size = file_path.stat().st_size
                    if file_size <= max_file_size and total_size + file_size <= total_max_size:
                        content = file_path.read_text(encoding='utf-8', errors='replace')
                        language = get_language_from_extension(file_path)
                        formatted_content = f"[file: {file_path}]\n```{language}\n{content}\n```"
                        loaded_files.append(formatted_content)
                        total_size += file_size
                        logger.info(f"Loaded file: {file_path} ({file_size} bytes)")
                    else:
                        if file_size > max_file_size:
                            logger.warning(f"Skipping large file {file_path} ({file_size} bytes)")
                        else:
                            logger.warning("Reached total size limit. Skipping remaining files.")
                            break
                else:
                    logger.warning(f"Skipping non-file: {file_path}")  # Log non-files
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")

        if not loaded_files:
            logger.warning("No files were successfully loaded")
            return "No files were successfully loaded. Files may be too large or unreadable."

        result = "\n\n".join(loaded_files)
        logger.info(f"Successfully batch loaded {len(loaded_files)} files ({total_size} total bytes)")
        return result

    except Exception as e:
        logger.error(f"Error in batch_load: {e}")
        return None
