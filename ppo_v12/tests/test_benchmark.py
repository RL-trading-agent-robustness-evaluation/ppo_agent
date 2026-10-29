import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT.parent/'ppo_v11/victimagent/src')]
from v12_benchmark import FractionalBudgetLedger, monthly_schedule, run_monthly
from victimagent.environment import DividendEvent, SplitEvent

class Builder:
    context_features = ()
    observation_features = tuple(range(20))
    def compose(self,*args,**kwargs): return np.zeros(20,dtype=np.float32)

def bundle(dates,prices,yield_pct=0.,dividends=(),splits=()):
    truth=pd.DataFrame(dict(date=pd.to_datetime(dates),open_raw=prices,close_raw=prices,dgs3mo_pct=yield_pct))
    observed=pd.DataFrame(dict(date=truth.date,observed_close=prices))
    return SimpleNamespace(market_true=truth,market_obs=observed,observation_builder=Builder(),
        dividend_events=dividends,split_events=splits)

class Tests(unittest.TestCase):
    def test_schedule_boundary_and_budget(self):
        dates=pd.bdate_range('2022-01-03','2025-12-31')
        schedule=monthly_schedule(dates)
        self.assertEqual(len(schedule),48)
        self.assertEqual(schedule[1]['execution_date'],'2022-01-04')
        self.assertAlmostEqual(sum(p['budget'] for p in schedule.values()),1_000_000.,places=8)
        for i,p in schedule.items():
            self.assertLess(pd.Timestamp(p['decision_date']),pd.Timestamp(p['execution_date']))
            self.assertEqual(p['execution_date'],str(next(d for d in dates[1:] if str(d.to_period('M'))==p['month']).date()))

    def test_constant_path(self):
        b=bundle(['2022-01-03','2022-01-04','2022-02-01','2022-02-02'],[100.]*4)
        rows,purchases,m=run_monthly(b,1000.)
        self.assertEqual(len(purchases),2)
        self.assertAlmostEqual(m['final_nav'],1000/1.001,places=10)
        self.assertAlmostEqual(m['total_transaction_cost'],1000*.001/1.001,places=10)
        self.assertAlmostEqual(m['final_cash'],0.)
        self.assertEqual(m['number_of_trades'],2)

    def test_fixed_cash_variable_price(self):
        b=bundle(['2022-01-03','2022-01-04','2022-02-01','2022-02-02'],[100.,100.,50.,80.])
        rows,p,m=run_monthly(b,1000.)
        self.assertAlmostEqual(p[1]['trade']['delta_shares'],2*p[0]['trade']['delta_shares'])
        self.assertAlmostEqual(m['final_nav'],(500/1.001/100+500/1.001/50)*80,places=9)

    def test_events_and_lagged_interest(self):
        dates=['2022-01-03','2022-01-04','2022-01-05','2022-02-01','2022-02-02']
        event=DividendEvent('div',pd.Timestamp('2022-01-05'),pd.Timestamp('2022-02-01'),2.)
        b=bundle(dates,[100.]*5,5.,(event,))
        rows,p,m=run_monthly(b,1000.)
        q=500/1.001/100
        self.assertAlmostEqual(m['total_dividends'],2*q)
        self.assertAlmostEqual(rows[0]['cash_interest'],1000*((1.05)**(1/365)-1))
        self.assertAlmostEqual(rows[2]['cash_interest'],rows[1]['cash_available']*((1.05)**(27/365)-1))
        self.assertGreater(m['final_cash'],m['total_dividends'])
        self.assertEqual(m['final_receivable'],0.)
        self.assertEqual([x['budget'] for x in p],[500.,500.])

    def test_split_fractional_and_conservation(self):
        ledger=FractionalBudgetLedger(1000.)
        ledger.budget=333.
        trade=ledger.execute_target(1.,101.,transaction_cost_bps=10.)
        self.assertAlmostEqual(trade.nav_open_post,1000-trade.transaction_cost)
        q=ledger.shares
        ledger.apply_split(1.5)
        self.assertAlmostEqual(ledger.shares,1.5*q)
        self.assertAlmostEqual(ledger.snapshot(101/1.5).nav,trade.nav_open_post)
        split=SplitEvent('split',pd.Timestamp('2022-01-05'),2.)
        b=bundle(['2022-01-03','2022-01-04','2022-01-05','2022-02-01','2022-02-02'],
            [100.,100.,50.,50.,50.],splits=(split,))
        rows,p,m=run_monthly(b,1000.)
        self.assertAlmostEqual(m['final_nav'],1000/1.001,places=9)

    def test_ex_date_purchase_not_entitled(self):
        event=DividendEvent('div',pd.Timestamp('2022-01-04'),pd.Timestamp('2022-01-05'),2.)
        b=bundle(['2022-01-03','2022-01-04','2022-01-05'],[100.]*3,dividends=(event,))
        self.assertEqual(run_monthly(b,1000.)[2]['total_dividends'],0.)

    def test_no_future_price_schedule(self):
        dates=pd.to_datetime(['2022-01-03','2022-01-04','2022-02-01'])
        a=bundle(dates,[100.,100.,50.])
        b=bundle(dates,[100.,100.,9999.])
        ra,pa,_=run_monthly(a,1000.)
        rb,pb,_=run_monthly(b,1000.)
        self.assertEqual(ra[0],rb[0])
        self.assertEqual([p['budget'] for p in pa],[p['budget'] for p in pb])

    def test_missing_month_and_insufficient_cash(self):
        with self.assertRaises(ValueError):
            monthly_schedule(pd.to_datetime(['2022-01-03','2022-01-04','2022-03-01']))
        ledger=FractionalBudgetLedger(10.)
        ledger.budget=20.
        with self.assertRaises(RuntimeError): ledger.execute_target(1.,100.)

    def test_benchmark_does_not_use_policy_observation(self):
        class StrictBuilder(Builder):
            def compose(self,*args,**kwargs):
                raise AssertionError('benchmark must not compose policy account features')
        b=bundle(['2022-01-03','2022-01-04','2022-01-05'],[101.]*3)
        b.observation_builder=StrictBuilder()
        self.assertAlmostEqual(run_monthly(b,1000.)[2]['final_nav'],1000/1.001)

if __name__=='__main__': unittest.main()
