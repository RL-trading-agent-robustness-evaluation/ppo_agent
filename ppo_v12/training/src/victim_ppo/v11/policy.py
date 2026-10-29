"""Pure scheduling and sampling rules; no prices or policy buffers owned here."""
import math
import numpy as np
import pandas as pd

def window_probabilities(session_count, length=63, half_life=126, recent_mass=.8):
    if length < 1 or half_life <= 0 or not 0 <= recent_mass <= 1:
        raise ValueError('invalid sampler parameters')
    count = session_count - length
    if count < 1:
        raise ValueError('no complete causal training window')
    starts = np.arange(count)
    ages = count - 1 - starts
    exponential = np.exp2(-ages / half_life)
    probabilities = recent_mass * exponential / exponential.sum() + (1-recent_mass)/count
    return starts, ages, probabilities

def align_vix(raw, sessions, limit=3):
    """Align without backfill; preserve invalid runs for causal failure at use."""
    frame = raw.copy()
    frame['observation_date'] = pd.to_datetime(frame['observation_date']).dt.normalize()
    if frame['observation_date'].duplicated().any():
        raise ValueError('duplicate VIX dates')
    values = pd.to_numeric(frame['VIXCLS'], errors='coerce')
    values = values.where(np.isfinite(values) & (values > 0))
    series = pd.Series(values.to_numpy(), index=frame['observation_date']).reindex(sessions)
    filled = series.ffill(limit=limit)
    return pd.DataFrame({'raw': series, 'value': filled, 'filled': series.isna() & filled.notna()}, index=sessions)

def period_for_vix(value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError('VIX must be finite positive index points')
    raw = 126 * (20 / value)**2
    return raw, int(np.clip(math.floor(raw+.5), 21, 252))

def schedule_at(vix, index, last_update, fixed=False, smoke=False):
    # Validate the immediately prior session even in fixed control; a fourth
    # missing session cannot be hidden by sparse evaluation of update dates.
    if index < 5:
        raise ValueError('five lagged VIX sessions required')
    window = vix.iloc[index-5:index]
    if not np.isfinite(window['value']).all():
        raise ValueError('VIX missing beyond three-session fill limit')
    mean = float(window['value'].mean())
    raw, period = period_for_vix(mean)
    if fixed:
        period = 126
    if smoke:
        period = 1  # smoke-only controller, never included in formal records
    elapsed = index-last_update
    return dict(vix_mean=mean, vix_window=window.reset_index().to_dict('records'),
                raw_period=raw, period=period, elapsed=elapsed,
                due=elapsed >= period, reason='fixed' if fixed else 'vix')
