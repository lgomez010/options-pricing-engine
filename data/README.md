# Data

This directory holds market data used by the pricing engine.

## Sample Data
The `sample/` subdirectory contains small, committed datasets for testing
and demonstration (e.g., a handful of option chains).

## Larger Datasets
Larger datasets (historical options chains, tick data) are excluded via
`.gitignore`. Use `download_data.py` to fetch them:

```bash
python data/download_data.py
```

## Sources
- CBOE delayed quotes (free, for demonstration)
- Yahoo Finance via `yfinance` (for underlying price histories)
