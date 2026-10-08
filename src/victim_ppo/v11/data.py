from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from victimagent.data.splits import DateRange
from victimagent.v8.features import build_v8_features, V8_CONTEXT_FEATURES, V8Bundle
from victimagent.environment import DividendEvent
from victimagent.v9.wrappers import spy_log_returns
from victimagent.v9.study import v9_bundle
from victimagent.evaluation import scaler_sha256
from .policy import align_vix

class MarketData:
    def __init__(self, root, config, end='2021-12-31'):
        raw = config.raw
        market = pd.read_csv(root/raw['market'], parse_dates=['date'])
        self.market = market.loc[market.date <= pd.Timestamp(end)].reset_index(drop=True)
        events = pd.read_csv(root/raw['events'], parse_dates=['disexdt','dispaydt'])
        self.events = events.loc[events.disexdt <= pd.Timestamp(end)].copy()
        features = build_v8_features(self.market)
        eligible = features.loc[(features.date >= '2010-01-01') &
            np.isfinite(features[list(V8_CONTEXT_FEATURES)].to_numpy()).all(axis=1)]
        self.sessions = pd.DatetimeIndex(eligible.date)
        if len(self.sessions) < 252:
            raise ValueError('252 valid post-2010 sessions required')
        self.fit = DateRange(self.sessions[0], self.sessions[251])
        initial, _ = self.bundle(self.sessions[251])
        self.builder = initial.observation_builder
        self.scaler_hash = scaler_sha256(self.builder)
        vix = pd.read_csv(root/raw['vix'])
        self.vix = align_vix(vix, self.sessions)

    def bundle(self, cutoff, *, start=None):
        """Prefix features/scaler fit never inspect rows after cutoff."""
        cutoff = pd.Timestamp(cutoff)
        if cutoff < self.fit.end:
            raise ValueError('cutoff precedes initialization')
        market = self.market.loc[self.market.date <= cutoff].copy()
        events = self.events.loc[self.events.disexdt <= cutoff].copy()
        selection = DateRange(pd.Timestamp(start) if start is not None else self.sessions[0], cutoff)
        if not hasattr(self, 'builder'):
            return v9_bundle(market, events, fit=self.fit, select=selection)
        # Transform with the original frozen builder; never call fit again.
        features=build_v8_features(market)
        scaled=self.builder.transform_context(features,require_complete=False)
        rows=scaled.loc[selection.contains(scaled.date)].reset_index(drop=True)
        if len(rows)<3 or not np.isfinite(rows[list(V8_CONTEXT_FEATURES)].to_numpy()).all():
            raise ValueError('incomplete selected features')
        if 'open_tradable' in rows and not rows.open_tradable.astype(bool).all():
            raise ValueError('nontradable open')
        truth=rows[['date','open_raw','close_raw','dgs3mo_pct']].copy()
        observed=rows[['date',*V8_CONTEXT_FEATURES]].copy()
        observed.insert(1,'observed_close',rows.close_raw.to_numpy())
        observed['ma_fast_above_slow']=(rows.feature_sma_20>rows.feature_sma_60).astype(np.int8)
        observed['trend_above_sma_200']=(rows.feature_price>rows.feature_sma_200).astype(np.int8)
        dates=pd.DatetimeIndex(rows.date)
        dividends=[]
        for row in events.loc[events.disexdt.isin(dates)].itertuples():
            pay=pd.Timestamp(row.dispaydt)
            if dates[0]<=pay<=dates[-1] and pay not in dates:
                pay=dates[dates.searchsorted(pay)]
            dividends.append(DividendEvent(event_id=row.event_id,ex_date=row.disexdt,
                pay_date=pay,per_share_amount=float(row.disdivamt)))
        bundle=V8Bundle(truth,observed,self.builder,tuple(dividends),metadata={'rows':len(rows)})
        spy=spy_log_returns(truth,rows.cash_distribution)
        if scaler_sha256(self.builder)!=self.scaler_hash:
            raise RuntimeError('frozen scaler changed')
        return bundle,spy
