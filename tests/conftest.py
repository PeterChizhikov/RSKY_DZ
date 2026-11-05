import importlib.util
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def orig():
    """Исходная (эталонная) версия main.py до доработки."""
    spec = importlib.util.spec_from_file_location("main_orig", ROOT / "tests" / "reference" / "main_orig.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
