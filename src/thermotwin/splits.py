"""Session-based splits. Never split the 2 Hz rows at random: neighbouring samples are
nearly identical, so a random split leaks the answer into training."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from .config import GROUP


def by_profiles(df: pd.DataFrame, test_profiles: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = df[GROUP].isin(test_profiles)
    if not mask.any():
        raise ValueError(f"none of {test_profiles} found in data")
    return df[~mask], df[mask]


def random_profiles(df: pd.DataFrame, n_test: int, seed: int = 0) -> list[int]:
    ids = np.array(sorted(df[GROUP].unique()))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(ids, size=n_test, replace=False).tolist())


def group_kfold(df: pd.DataFrame, k: int = 5, seed: int = 0) -> Iterator[tuple[list[int], list[int]]]:
    """Yield (train_profiles, val_profiles), balancing folds by session length."""
    sizes = df.groupby(GROUP).size().sample(frac=1.0, random_state=seed).sort_values(ascending=False)
    folds: list[list[int]] = [[] for _ in range(k)]
    load = [0] * k
    for pid, n in sizes.items():
        j = int(np.argmin(load))
        folds[j].append(int(pid))
        load[j] += int(n)
    all_ids = set(sizes.index.astype(int))
    for f in folds:
        yield sorted(all_ids - set(f)), sorted(f)


def random_row_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The WRONG way, kept only for the leakage experiment."""
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < test_frac
    return df[~mask], df[mask]
