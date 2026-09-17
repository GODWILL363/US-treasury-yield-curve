# US Treasury Yield Curve Modeling

Fits the **Nelson-Siegel** and **Nelson-Siegel-Svensson** models to US Treasury
par yields. For each date it estimates the term structure and reports the three
readings rates desks care about: **level**, **slope**, and **curvature**. It
also tracks how those factors move over time.

Built with Python (NumPy, SciPy, pandas, Matplotlib).

## Data (real)

`data/treasury_yields.csv` holds real daily constant-maturity Treasury par
yields (1M to 30Y) for 2018-2019, sourced from the US Treasury / Federal
Reserve H.15 series. Refresh or extend it any time:

```bash
# direct from the US Treasury XML feed
pip install ust
python scripts/fetch_yields.py --source treasury --start 2018 --end 2019

# Federal Reserve H.15 via FRED (needs a free FRED_API_KEY)
pip install pandas-datareader
export FRED_API_KEY=your_key
python scripts/fetch_yields.py --source fred --start 2018-01-01 --end 2019-12-31

# public GitHub mirror of the Treasury feed, no key needed
python scripts/fetch_yields.py --source github --start 2018 --end 2019
```

## Quick start

```bash
pip install -r requirements.txt
python run_analysis.py
```

Outputs in `outputs/`:
- `curve_fit.png` fitted curve for the most recent date
- `factors.png` level, slope, and curvature over time
- `ns_factors.csv` the same factors as a table

## Use your own data

`run_analysis.py --data your_file.csv`

Wide format: first column `date`, then one column per maturity in years
(headers like `0.25, 1, 2, 10, 30`), values in percent.

## Model

Nelson-Siegel:

    y(t) = b0 + b1 * (1 - exp(-t/L)) / (t/L) + b2 * [ (1 - exp(-t/L)) / (t/L) - exp(-t/L) ]

Svensson adds a second curvature term with its own decay `L2`. Here `t` is
maturity in years and `L`, `L2` are decay parameters.

## Tests

```bash
python tests/test_models.py
```

## Layout

```
src/yield_curve/   models.py, fit.py, plotting.py
data/              treasury_yields.csv
scripts/           fetch_yields.py
tests/             test_models.py
run_analysis.py
```

## License

MIT. See LICENSE.
