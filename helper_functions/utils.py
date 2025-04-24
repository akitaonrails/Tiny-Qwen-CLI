from pathlib import Path

def get_language_from_extension(file_path: Path) -> str:
    """Determine language for markdown code block based on file extension."""
    ext = file_path.suffix.lower()
    lang_map = {
        '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
        '.html': 'html', '.css': 'css', '.java': 'java', '.c': 'c',
        '.cpp': 'cpp', '.go': 'go', '.rs': 'rust', '.rb': 'ruby',
        '.php': 'php', '.sh': 'bash', '.md': 'markdown', '.json': 'json',
        '.yaml': 'yaml', '.yml': 'yaml', '.xml': 'xml', '.sql': 'sql',
        '.dockerfile': 'dockerfile', '.tf': 'terraform', '.hcl': 'terraform',
        '.jsx': 'jsx', '.tsx': 'tsx'
    }
    return lang_map.get(ext, 'text')
