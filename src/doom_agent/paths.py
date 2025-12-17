"""Shared filesystem paths for the doom_agent package."""
from pathlib import Path

# Repository root is two levels up from this file (src/doom_agent/paths.py -> src -> repo)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCENARIOS_DIR = SRC_ROOT / "doom_agent" / "scenarios"


def scenario_path(name: str) -> str:
    """Return an absolute path to a scenario resource (cfg or wad)."""
    return str(SCENARIOS_DIR / name)
