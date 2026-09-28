# Stock screener

A screener that automates the stock-picking method from
[this @stockweatherman TikTok](https://www.tiktok.com/@stockweatherman/video/7673898839849340174):
find strong companies that are quietly consolidating *before* they run, instead of buying after a 600% move.

## The rules

| Stage | Rule | How it's checked |
|---|---|---|
| Trend | Long-term uptrend | Price above the 12-month SMA (month-end closes) |
| Trend | No local consolidation below the averages | Price above the 150-day and 200-day SMAs |
| Trend | No penny stocks | Price of at least $10 (`--min-price`) |
| Chart | Volatility contraction pattern (VCP) | Flat ceiling, higher lows, price within 12% of the ceiling, not broken out, not parabolic |
| Fundamentals | Room to grow, but not too volatile | Market cap between $2B and $100B |
| Fundamentals | The business makes money | Operating cash flow above zero |
| Fundamentals | A real company | More than 20 employees |
| Fundamentals | P/E trending up | Price ÷ trailing-12-month EPS at each earnings date, upward slope |
| Refinement | Insiders haven't dumped it | Insiders hold at least 5% |
| Refinement | Growing cash | Free cash flow trending up, and positive |
| Refinement | Growing earnings | TTM EPS trending up and higher than a year ago |

In the video the chart check is done by eye. Here it's automated, but the output is still a shortlist to look through yourself.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

```bash
.venv/bin/python screener.py                        # whole US market, weekly charts
.venv/bin/python screener.py --timeframe daily
.venv/bin/python screener.py --tickers DRS CAKE --all
.venv/bin/python screener.py --help                 # every threshold is adjustable
```

A full run takes a few minutes. Price and fundamentals data are cached in `.cache/` for the day.

Results are saved to `results/screen_<date>_<timeframe>.csv`, with each rule's value and whether it passed:

- **FINAL**: passes every rule
- **watch**: good chart and fundamentals, but fails a refinement rule
- **near-miss**: passes everything except one VCP rule, so it's worth a manual look

Not financial advice.
