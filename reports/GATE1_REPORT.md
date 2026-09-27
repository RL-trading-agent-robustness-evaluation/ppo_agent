# Gate 1 Data Quality Report

**Overall status: PASS**

Generated: `2026-09-27T08:18:12.954672+00:00`

| Check | Status | Result |
|---|---|---|
| `schema.market_daily` | **PASS** | Required columns are present. |
| `schema.distributions` | **PASS** | Required columns are present. |
| `market.valid_dates` | **PASS** | All market dates parse. |
| `market.asset_identity` | **PASS** | PERMNO-to-ticker mapping matches the frozen contract. |
| `market.unique_monotonic_dates` | **PASS** | Dates are unique and monotonic within each PERMNO. |
| `market.ohlc_bounds` | **PASS** | OHLC bounds hold. |
| `market.core_values` | **PASS** | Core prices and volume are finite and valid. |
| `market.cross_asset_calendar` | **PASS** | All three ETFs have the same observed CRSP trading dates. |
| `market.external_exchange_calendar` | **WARN** | No external NYSE calendar is bundled. The pipeline uses identical CRSP sessions across three active ETFs as its canonical calendar; add an exchange-calendar package for an independent formal check. |
| `market.regular_trading_rows` | **PASS** | All rows are active, regular-way observations. |
| `distributions.required_dates` | **PASS** | All supplied events have ex-date and payment date. |
| `distributions.daily_amount_reconciliation` | **PASS** | All overlapping distribution amounts reconcile within source precision. |
| `distributions.coverage_through_test_end` | **PASS** | Distribution event coverage includes the locked test end. |
| `cash.DGS3MO.dates` | **PASS** | Cash-rate dates parse and are unique. |
| `cash.DGS3MO.missing_observations` | **PASS** | Missing FRED values are retained in raw and excluded before backward-looking as-of alignment; they usually represent non-publication dates. |
| `cash.DTB3.dates` | **PASS** | Cash-rate dates parse and are unique. |
| `cash.DTB3.missing_observations` | **PASS** | Missing FRED values are retained in raw and excluded before backward-looking as-of alignment; they usually represent non-publication dates. |
| `cash.coverage_for_warmup` | **PASS** | Cash-rate snapshots cover the required warm-up start. |
| `cash.primary_available_in_formal_periods` | **PASS** | Lagged DGS3MO is available for every formal train, validation, and test row. |
| `processed.raw_traceability` | **PASS** | Every processed row maps to one unique raw market row. |
| `processed.no_normalization_fit` | **PASS** | No normalization was fit in the data pipeline. |
| `metadata.download_timestamps` | **WARN** | Original download timestamps were not embedded in the supplied files; checksums still freeze the exact snapshots. |

## Blocking items

- None.

## Warnings and documented exceptions

- `market.external_exchange_calendar`: No external NYSE calendar is bundled. The pipeline uses identical CRSP sessions across three active ETFs as its canonical calendar; add an exchange-calendar package for an independent formal check.
- `metadata.download_timestamps`: Original download timestamps were not embedded in the supplied files; checksums still freeze the exact snapshots.

## Decision

Gate 1 passes. The team may proceed to environment development.
