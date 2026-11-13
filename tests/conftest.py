import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from thermotwin.physics import synthetic_dataset  # noqa: E402


@pytest.fixture(scope="session")
def small_df():
    return synthetic_dataset(n_profiles=4, hours=(0.25, 0.4), seed=7)
