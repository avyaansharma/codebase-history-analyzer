import os
import re
import subprocess
from pathlib import Path
from typing import Optional


def find_repo_root(start_path: Optional[str] = None) -> Path:
    """Finds the git repository root or returns current working directory."""
    current = Path(start_path or os.getcwd()).resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").is_dir():
            return parent
    return current


def get_archaeologist_dir(repo_path: Optional[str] = None) -> Path:
    """Returns the dedicated .archaeologist directory inside the repository."""
    root = find_repo_root(repo_path)
    arch_dir = root / ".archaeologist"
    arch_dir.mkdir(parents=True, exist_ok=True)
    return arch_dir


def get_default_db_path(repo_path: Optional[str] = None) -> str:
    """Returns the default SQLite database path."""
    if not repo_path:
        custom_url = os.getenv("DATABASE_URL")
        if custom_url and custom_url.startswith("sqlite:///"):
            return custom_url.replace("sqlite:///", "")
    return str(get_archaeologist_dir(repo_path) / "archaeologist.db")


def get_default_db_url(repo_path: Optional[str] = None) -> str:
    """Returns the default SQLite SQLAlchemy connection URL."""
    if not repo_path:
        custom_url = os.getenv("DATABASE_URL")
        if custom_url:
            return custom_url
    path_str = get_default_db_path(repo_path).replace("\\", "/")
    return f"sqlite:///{path_str}"


def get_default_bm25_path(repo_path: Optional[str] = None) -> str:
    """Returns default BM25 index path."""
    if not repo_path:
        custom = os.getenv("BM25_INDEX_PATH")
        if custom:
            return custom
    return str(get_archaeologist_dir(repo_path) / "bm25_index.bin")


def get_default_qdrant_path(repo_path: Optional[str] = None) -> str:
    """Returns default embedded Qdrant storage path."""
    if not repo_path:
        custom = os.getenv("QDRANT_STORAGE_PATH")
        if custom:
            return custom
    return str(get_archaeologist_dir(repo_path) / "qdrant_db")


def detect_github_remote(repo_path: Optional[str] = None) -> Optional[str]:
    """Extracts GitHub https URL from git remote origin if available."""
    root = find_repo_root(repo_path)
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if proc.returncode == 0:
            url = proc.stdout.strip()
            # Convert git@github.com:owner/repo.git or ssh://git@github.com/owner/repo.git
            ssh_match = re.match(r"^(?:ssh://)?git@github\.com[:/]([^/]+)/(.+?)(?:\.git)?/?$", url)
            if ssh_match:
                return f"https://github.com/{ssh_match.group(1)}/{ssh_match.group(2)}"
            if url.startswith("https://github.com/"):
                return url.removesuffix(".git").removesuffix("/")
    except Exception:
        pass
    return None


def calculate_window_since(window: Optional[str]) -> Optional[str]:
    """Calculates the ISO YYYY-MM-DD cutoff date from a window descriptor.

    Supported windows:
    - '6m', '6months', '6-months': ~6 months (182 days)
    - '1y', '1year', '1-year': 1 year (365 days)
    - '2y', '2years', '2-years': 2 years (730 days)
    - 'full', 'all': Full repository history (returns None)
    """
    if not window:
        return None
    normalized = window.strip().lower().replace(" ", "").replace("-", "")
    from datetime import datetime, timedelta
    now = datetime.now()
    if normalized in ("6m", "6months", "6month"):
        return (now - timedelta(days=182)).strftime("%Y-%m-%d")
    elif normalized in ("1y", "1year", "1years"):
        return (now - timedelta(days=365)).strftime("%Y-%m-%d")
    elif normalized in ("2y", "2year", "2years"):
        return (now - timedelta(days=730)).strftime("%Y-%m-%d")
    elif normalized in ("full", "all", "none"):
        return None
    else:
        raise ValueError(
            f"Invalid window '{window}'. Allowed options: '6m' (6 months), '1y' (1 year), '2y' (2 years), or 'full' (all history)."
        )
