from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent


def load_prompt(name: str) -> str:
    """
    Load a prompt file from the prompts directory.

    Args:
        name: Prompt file stem without extension (e.g., "invoice_extraction_system").

    Raises:
        FileNotFoundError: If the prompt file cannot be located.
    """

    path = PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")
