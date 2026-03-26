import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

if "structlog" not in sys.modules:
    sys.modules["structlog"] = types.ModuleType("structlog")
