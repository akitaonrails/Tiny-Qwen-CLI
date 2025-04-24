"""
Helper function for loading files in Qwen CLI.
This function will be dynamically loaded by the CLI.
"""

import logging
from pathlib import Path
from helper_functions.utils import get_language_from_extension 

logger = logging.getLogger("qwen_cli.helpers.load_file")

def handle_load_file(filepath):
    """
    Whenever the user asks to read, load, analyze some code and provides a relative path, such as ./file.py or utils/utils.py or similar,
    write the command `[LOAD_FILE filepath]` (e.g., `[LOAD_FILE ./my_script.py]`).

    This command will load the file and return its content formatted for the model context.
    """
    logger.info(f"Loading file: {filepath}")
    
    try:
        # 1. Start with a Path object
        file_path = Path(filepath)
        
        # 2. Log initial state
        logger.debug(f"Initial filepath: {filepath}, is_absolute: {file_path.is_absolute()}")
        
        # 3. If it's relative, try to make it absolute relative to /app
        if not file_path.is_absolute():
            app_path = Path("/project")  # Docker mount point
            possible_path = app_path / file_path
            logger.debug(f"Trying /project-relative path: {possible_path}")
            if possible_path.exists():
                file_path = possible_path
                logger.debug(f"  Found: Using /project-relative path")
            else:
                # 4. If still not found, try relative to the current working directory
                # (This might be relevant in non-Docker scenarios)
                cwd_path = Path.cwd()
                possible_path = cwd_path / file_path
                logger.debug(f"Trying CWD-relative path: {possible_path}")
                if possible_path.exists():
                    file_path = possible_path
                    logger.debug(f"  Found: Using CWD-relative path")
                else:
                    logger.debug("  Not found in /project or CWD")
        
        # 5. Log the final resolved path
        logger.debug(f"Resolved file_path: {file_path}")
        
        # 6. Final check: does the resolved path exist?
        if not file_path.exists():
            logger.error(f"File not found: {file_path} (original: {filepath})")
            return None
            
        if not file_path.is_file():
            logger.error(f"Path is not a file: {file_path} (original: {filepath})")
            return None
            
        # 7. Read the file content
        content = file_path.read_text(encoding='utf-8', errors='replace')
        
        # 8. Get appropriate language for syntax highlighting
        language = get_language_from_extension(file_path)
        
        # 9. Format the content for the model
        formatted_content = f"[file: {file_path}]\\n```{language}\\n{content}\\n```"
        
        logger.info(f"Successfully loaded file: {file_path} ({len(content)} characters)")
        return formatted_content
        
    except Exception as e:
        logger.error(f"Error loading file {filepath}: {e}")
        return None
