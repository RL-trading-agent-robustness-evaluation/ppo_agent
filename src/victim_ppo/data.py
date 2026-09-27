from __future__ import annotations

from pathlib import Path

import pandas as pd

from victimagent.data.real_adapter import prepare_asset_bundle
from victimagent.data.splits import load_split_contract


DEVELOPMENT_CUTOFF = pd.Timestamp("2022-01-01")


def _read_before(path: Path, date_column: str) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(path, chunksize=50_000):
        dates = pd.to_datetime(chunk[date_column], errors="raise")
        chunks.append(chunk.loc[dates < DEVELOPMENT_CUTOFF].copy())
    if not chunks:
        raise ValueError(f"empty input: {path}")
    return pd.concat(chunks, ignore_index=True)


def load_development_bundles(root: Path, data_root: Path):
    """Build train and validation only; rows from the known period are discarded first."""
    market = _read_before(data_root / "data/processed/market_base_with_splits.csv", "date")
    events_path = data_root / "data/interim/distribution_events.csv"
    events_header = pd.read_csv(events_path, nrows=0)
    event_date = "disexdt" if "disexdt" in events_header.columns else "exdt"
    events = _read_before(events_path, event_date)
    split = load_split_contract(root / "victimagent/configs/splits.yaml")
    train = prepare_asset_bundle(market, events, split, ticker="SPY", partition="train")
    validation = prepare_asset_bundle(market, events, split, ticker="SPY", partition="validation")
    return train, validation

