"""Preprocessing objects that can only be fitted on the train partition."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from victimagent.data.splits import SplitContract


@dataclass
class TrainOnlyZScore:
    """Simple auditable z-score scaler with a train-only fit guard."""

    columns: tuple[str, ...]
    mean_: pd.Series | None = None
    scale_: pd.Series | None = None

    def fit(
        self,
        frame: pd.DataFrame,
        contract: SplitContract,
        *,
        date_column: str = "date",
    ) -> "TrainOnlyZScore":
        if date_column not in frame.columns:
            raise KeyError(f"missing date column: {date_column}")
        missing = set(self.columns) - set(frame.columns)
        if missing:
            raise KeyError(f"missing feature columns: {sorted(missing)}")

        dates = pd.to_datetime(frame[date_column])
        if len(frame) == 0:
            raise ValueError("cannot fit on an empty frame")
        if dates.min() < contract.train.start or dates.max() > contract.train.end:
            raise ValueError("preprocessing fit data must lie entirely inside train")

        values = frame.loc[:, self.columns].astype(float)
        mean = values.mean()
        scale = values.std(ddof=0)
        if mean.isna().any() or scale.isna().any():
            raise ValueError("fit data contains insufficient or missing values")
        if (scale == 0).any():
            constant = scale.index[scale == 0].tolist()
            raise ValueError(f"constant features cannot be standardized: {constant}")

        self.mean_ = mean
        self.scale_ = scale
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit must be called before transform")
        result = frame.copy()
        result.loc[:, self.columns] = (
            result.loc[:, self.columns].astype(float) - self.mean_
        ) / self.scale_
        return result

    def state(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("scaler has not been fitted")
        return tuple(self.mean_.tolist()), tuple(self.scale_.tolist())

