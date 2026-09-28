#!/usr/bin/env python3
"""
Screens US stocks for early-stage setups, following the method from
@stockweatherman's TikTok on finding stocks before they run.

Stages, cheapest first so Yahoo's per-ticker endpoints only see survivors:

  1. Universe      all US-listed common stocks
  2. Trend         price >= $10 and above the 150d, 200d and 12-month SMAs
  3. VCP           volatility contraction pattern: flat ceiling, higher lows,
                   price still inside the base, no parabolic run-up
  4. Fundamentals  market cap $2B-$100B, positive operating cash flow,
                   more than 20 employees, P/E trending up
  5. Refinement    insiders still holding, free cash flow and EPS trending up

Not financial advice. Treat the output as a shortlist to review by eye.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"
RESULTS = ROOT / "results"

SYMBOL_FILES = (
    ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "Symbol"),
    ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "ACT Symbol"),
)
NON_COMMON = re.compile(r"\b(warrants?|rights?|units?|preferred|notes|debentures?)\b|%", re.IGNORECASE)
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126 Safari/537.36")

STATUS_ORDER = ["FINAL", "watch", "near-miss", "rejected"]


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def cached(name: str) -> Path:
    CACHE.mkdir(exist_ok=True)
    return CACHE / f"{name}_{date.today()}"


def http_get(url: str, tries: int = 4) -> str:
    for attempt in range(tries):
        try:
            r = requests.get(url, headers={"User-Agent": BROWSER_UA}, timeout=30)
            r.raise_for_status()
            return r.text
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("unreachable")


def load_universe() -> list[str]:
    path = cached("universe").with_suffix(".json")
    if path.exists():
        return json.loads(path.read_text())

    frames = []
    for url, symbol_col in SYMBOL_FILES:
        df = pd.read_csv(io.StringIO(http_get(url)), sep="|")
        df = df[(df["ETF"] == "N") & (df["Test Issue"] == "N")]
        frames.append(df.rename(columns={symbol_col: "symbol"})[["symbol", "Security Name"]])

    df = pd.concat(frames).dropna()
    df = df[~df["symbol"].str.contains(r"[$^]")]
    df = df[~df["Security Name"].str.contains(NON_COMMON)]
    # Yahoo writes share classes with a dash: BRK.B -> BRK-B
    symbols = sorted(set(df["symbol"].str.strip().str.replace(".", "-", regex=False)))

    path.write_text(json.dumps(symbols))
    return symbols


def download_prices(tickers: list[str], chunk: int = 400, use_cache: bool = True) -> dict[str, pd.DataFrame]:
    key = hashlib.md5(",".join(tickers).encode()).hexdigest()[:10]
    path = cached(f"prices_{key}").with_suffix(".pkl")
    if use_cache and path.exists():
        return pickle.loads(path.read_bytes())

    prices: dict[str, pd.DataFrame] = {}
    too_short: set[str] = set()

    def fetch(symbols: list[str], threads: bool) -> None:
        data = yf.download(symbols, period="2y", auto_adjust=True, group_by="ticker",
                           threads=threads, progress=False)
        for t in symbols:
            if t not in data.columns.get_level_values(0):
                continue
            df = data[t].dropna(subset=["Close"])
            if len(df) >= 260:
                prices[t] = df
            elif len(df):
                too_short.add(t)

    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        log(f"  downloading prices {i + 1}-{i + len(batch)} / {len(tickers)}")
        fetch(batch, threads=True)
        # Yahoo silently drops a few tickers from big threaded requests.
        missing = [t for t in batch if t not in prices and t not in too_short]
        if missing:
            time.sleep(2)
            fetch(missing, threads=False)

    path.write_bytes(pickle.dumps(prices))
    return prices


def check_trend(df: pd.DataFrame, min_price: float) -> dict:
    close = df["Close"]
    price = float(close.iloc[-1])
    sma150 = float(close.rolling(150).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])
    sma12m = float(close.resample("ME").last().rolling(12).mean().iloc[-1])
    return {
        "price": round(price, 2),
        "sma150": round(sma150, 2),
        "sma200": round(sma200, 2),
        "sma12m": round(sma12m, 2),
        "ok_price": price >= min_price,
        "ok_sma150": price > sma150,
        "ok_sma200": price > sma200,
        "ok_sma12m": price > sma12m,
    }


@dataclass
class VCP:
    passed: bool = False
    score: float = 0.0
    fails: list[str] = field(default_factory=list)
    ceiling: float = np.nan
    below_ceiling_pct: float = np.nan
    contractions: int = 0
    depths_pct: str = ""
    volume_ratio: float = np.nan
    above_sma50_pct: float = np.nan
    return_6m_pct: float = np.nan


def swing_points(high: np.ndarray, low: np.ndarray, k: int) -> tuple[list[int], list[int]]:
    """Indices of bars that are the highest high / lowest low within k bars either side."""
    highs, lows = [], []
    for i in range(k, len(high) - k):
        window = slice(i - k, i + k + 1)
        if high[i] == high[window].max():
            highs.append(i)
        if low[i] == low[window].min():
            lows.append(i)
    return highs, lows


def detect_vcp(df: pd.DataFrame, timeframe: str = "weekly", ceiling_tol: float = 0.06,
               max_below_ceiling: float = 0.12, breakout_tol: float = 0.02,
               max_above_sma50: float = 0.25, max_return_6m: float = 1.0) -> VCP:
    if timeframe == "weekly":
        bars = df.resample("W-FRI").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
        base_len, k = 30, 2
    else:
        bars, base_len, k = df, 130, 5

    close = df["Close"]
    price = float(close.iloc[-1])
    above_sma50 = price / float(close.rolling(50).mean().iloc[-1]) - 1
    return_6m = price / float(close.iloc[-126]) - 1
    avg_vol_50 = df["Volume"].iloc[-50:].mean()
    volume_ratio = float(df["Volume"].iloc[-10:].mean() / avg_vol_50) if avg_vol_50 > 0 else np.nan

    vcp = VCP(volume_ratio=round(volume_ratio, 2),
              above_sma50_pct=round(above_sma50 * 100, 1),
              return_6m_pct=round(return_6m * 100, 1))

    # He wants in before the run, not after a vertical move.
    if above_sma50 > max_above_sma50:
        vcp.fails.append("parabolic_vs_sma50")
    if return_6m > max_return_6m:
        vcp.fails.append("parabolic_6m")

    base = bars.iloc[-base_len:]
    high, low = base["High"].to_numpy(), base["Low"].to_numpy()
    swing_highs, swing_lows = swing_points(high, low, k)
    if len(swing_highs) < 2:
        vcp.fails.append("too_few_swing_highs")
        return vcp

    # The latest bar counts too, otherwise a fresh breakout would look like it's inside the base.
    ceiling = float(max(high[swing_highs].max(), high[-1]))
    near_ceiling = [i for i in swing_highs if high[i] >= ceiling * (1 - ceiling_tol)]

    if len(near_ceiling) < 2:
        vcp.fails.append("no_flat_ceiling")
    # A falling last high means a symmetric triangle, which he explicitly avoids.
    if high[swing_highs[-1]] < ceiling * (1 - ceiling_tol):
        vcp.fails.append("lower_highs")

    base_start = near_ceiling[0] if near_ceiling else len(base)
    base_lows = [i for i in swing_lows if i > base_start]
    depths = [(ceiling - low[i]) / ceiling for i in base_lows]
    if len(base_lows) < 2:
        vcp.fails.append("too_few_contractions")
    elif any(low[b] <= low[a] for a, b in zip(base_lows, base_lows[1:])):
        vcp.fails.append("not_higher_lows")
    if depths and depths[0] > 0.5:
        vcp.fails.append("first_pullback_too_deep")

    below_ceiling = (ceiling - price) / ceiling
    if below_ceiling > max_below_ceiling:
        vcp.fails.append("too_far_below_ceiling")
    if price > ceiling * (1 + breakout_tol):
        vcp.fails.append("already_broken_out")

    # Rough ranking for manual review: more contractions, a tight last pullback,
    # drying-up volume and price sitting right under the ceiling all score higher.
    score = min(len(base_lows), 4) * 10
    if depths:
        score += max(0.0, 25 - depths[-1] * 250)
    if not np.isnan(volume_ratio):
        score += max(0.0, min(20, (1 - volume_ratio) * 40))
    score += max(0.0, 15 - below_ceiling * 150)
    score -= 15 * len(vcp.fails)

    vcp.passed = not vcp.fails
    vcp.score = round(score, 1)
    vcp.ceiling = round(ceiling, 2)
    vcp.below_ceiling_pct = round(below_ceiling * 100, 2)
    vcp.contractions = len(base_lows)
    vcp.depths_pct = " > ".join(f"{d * 100:.1f}" for d in depths)
    return vcp


class YahooThrottled(Exception):
    pass


def fetch_info(t: yf.Ticker) -> dict:
    # When throttled, Yahoo returns a near-empty dict instead of an error.
    info = t.info
    if "quoteType" not in info:
        raise YahooThrottled(t.ticker)
    if info["quoteType"] != "EQUITY" or not info.get("marketCap"):
        raise ValueError("not a common stock")
    return info


def with_backoff(fn, tries: int = 5):
    delay = 10.0
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            if attempt == tries - 1:
                raise
            if isinstance(e, YahooThrottled) or "Rate" in type(e).__name__ or "Too Many" in str(e):
                log(f"    rate limited, sleeping {delay:.0f}s")
                time.sleep(delay)
                delay *= 2
            else:
                time.sleep(1)


def trend_slope(values: list[float]) -> float:
    """Regression slope per step, scaled by the average magnitude so tickers are comparable."""
    y = np.asarray(values, dtype=float)
    if len(y) < 3:
        return np.nan
    slope = np.polyfit(np.arange(len(y)), y, 1)[0]
    return float(slope / (np.mean(np.abs(y)) or 1.0))


def ttm_eps_history(t: yf.Ticker) -> list[tuple[str, float]]:
    # Earnings dates go back ~4 years; the quarterly statements only ~5 quarters.
    try:
        dates = with_backoff(lambda: t.get_earnings_dates(limit=20))
        quarterly = dates["Reported EPS"].dropna().sort_index()
        quarterly.index = quarterly.index.tz_localize(None)
    except Exception:
        quarterly = pd.Series(dtype=float)

    if quarterly.empty:
        stmt = t.quarterly_income_stmt
        if "Diluted EPS" in stmt.index:
            quarterly = stmt.loc["Diluted EPS"].dropna().sort_index()

    ttm = quarterly.rolling(4).sum().dropna()
    return [(d.strftime("%Y-%m-%d"), float(v)) for d, v in ttm.items()]


def pe_history(close: pd.Series, ttm_eps: list[tuple[str, float]]) -> list[tuple[str, float]]:
    close = close.copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    history = []
    for day, eps in ttm_eps:
        before = close[:day]
        if eps > 0 and len(before):
            history.append((day, float(before.iloc[-1]) / eps))
    if ttm_eps and ttm_eps[-1][1] > 0:
        history.append((str(close.index[-1].date()), float(close.iloc[-1]) / ttm_eps[-1][1]))
    return history


def fcf_history(t: yf.Ticker) -> list[tuple[str, float]]:
    annual = t.cashflow
    if "Free Cash Flow" not in annual.index:
        return []
    yearly = annual.loc["Free Cash Flow"].dropna().sort_index()
    history = [(d.strftime("%Y"), float(v)) for d, v in yearly.items()]

    quarterly = t.quarterly_cashflow
    if "Free Cash Flow" in quarterly.index:
        q = quarterly.loc["Free Cash Flow"].dropna().sort_index()
        if len(q) >= 4 and (yearly.empty or q.index[-1] > yearly.index[-1]):
            history.append(("TTM", float(q.iloc[-4:].sum())))
    return history


def fetch_fundamentals(ticker: str, prices: pd.DataFrame, use_cache: bool = True) -> dict:
    path = cached(f"fund_{ticker}").with_suffix(".json")
    if use_cache and path.exists():
        return json.loads(path.read_text())

    t = yf.Ticker(ticker)
    info = with_backoff(lambda: fetch_info(t))
    data = {k: info.get(k) for k in (
        "shortName", "sector", "industry", "marketCap", "operatingCashflow",
        "fullTimeEmployees", "heldPercentInsiders")}
    data["ttm_eps"] = ttm_eps_history(t)
    data["pe"] = pe_history(prices["Close"], data["ttm_eps"])
    try:
        data["fcf"] = fcf_history(t)
    except Exception:
        data["fcf"] = []

    path.write_text(json.dumps(data))
    return data


def check_fundamentals(f: dict, args: argparse.Namespace) -> dict:
    market_cap = f.get("marketCap") or 0
    op_cashflow = f.get("operatingCashflow")
    employees = f.get("fullTimeEmployees")
    insiders = f.get("heldPercentInsiders")
    pe = [v for _, v in f["pe"]][-args.trend_points:]
    eps = [v for _, v in f["ttm_eps"]]
    fcf = [v for _, v in f["fcf"]]

    pe_slope = trend_slope(pe)
    eps_slope = trend_slope(eps[-args.trend_points:])
    fcf_slope = trend_slope(fcf)
    eps_yoy = (eps[-1] - eps[-5]) / abs(eps[-5]) if len(eps) >= 5 and eps[-5] else np.nan

    def rounded(x, n=3):
        return None if x is None or np.isnan(x) else round(x, n)

    return {
        "name": f.get("shortName"),
        "sector": f.get("sector"),
        "industry": f.get("industry"),
        "market_cap_b": round(market_cap / 1e9, 2),
        "op_cashflow_m": None if op_cashflow is None else round(op_cashflow / 1e6, 1),
        "employees": employees,
        "insiders_pct": None if insiders is None else round(insiders * 100, 1),
        "pe_now": round(pe[-1], 1) if pe else None,
        "pe_slope": rounded(pe_slope),
        "eps_ttm": round(eps[-1], 2) if eps else None,
        "eps_yoy_pct": rounded(eps_yoy * 100, 1),
        "eps_slope": rounded(eps_slope),
        "fcf_history_m": " > ".join(f"{v / 1e6:.0f}" for v in fcf),
        "fcf_slope": rounded(fcf_slope),

        "ok_market_cap": args.min_mcap * 1e9 <= market_cap <= args.max_mcap * 1e9,
        "ok_op_cashflow": op_cashflow is not None and op_cashflow > 0,
        # Yahoo leaves this blank for plenty of real companies, so missing counts as a pass.
        "ok_employees": employees is None or employees > args.min_employees,
        "ok_pe_trend": pe_slope > 0,

        "ok_insiders": insiders is not None and insiders * 100 >= args.min_insider_pct,
        "ok_fcf_trend": fcf_slope > 0 and fcf[-1] > 0,
        "ok_eps_trend": eps_slope > 0 and eps[-1] > 0 and not eps_yoy <= 0,
    }


TREND_RULES = ["ok_price", "ok_sma150", "ok_sma200", "ok_sma12m"]
FUNDAMENTAL_RULES = ["ok_market_cap", "ok_op_cashflow", "ok_employees", "ok_pe_trend"]
REFINEMENT_RULES = ["ok_insiders", "ok_fcf_trend", "ok_eps_trend"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="*", help="screen only these tickers instead of the whole market")
    p.add_argument("--timeframe", choices=["weekly", "daily"], default="weekly")
    p.add_argument("--min-price", type=float, default=10.0, help="set to 0 to allow cheaper stocks")
    p.add_argument("--min-mcap", type=float, default=2.0, help="in billions")
    p.add_argument("--max-mcap", type=float, default=100.0, help="in billions")
    p.add_argument("--min-employees", type=int, default=20)
    p.add_argument("--min-insider-pct", type=float, default=5.0)
    p.add_argument("--trend-points", type=int, default=8, help="quarters used for the P/E and EPS trend")
    p.add_argument("--ceiling-tol", type=float, default=0.06, help="how close swing highs must be to count as flat")
    p.add_argument("--max-below-ceiling", type=float, default=0.12)
    p.add_argument("--near-miss", type=int, default=1, help="also check fundamentals for charts failing this many VCP rules")
    p.add_argument("--all", action="store_true", help="check fundamentals for every ticker (useful with --tickers)")
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    use_cache = not args.no_cache

    tickers = [t.upper().replace(".", "-") for t in args.tickers] if args.tickers else load_universe()
    log(f"Universe: {len(tickers)} tickers")
    prices = download_prices(tickers, use_cache=use_cache)

    rows = []
    for ticker, df in prices.items():
        vcp = detect_vcp(df, args.timeframe, args.ceiling_tol, args.max_below_ceiling)
        rows.append({"ticker": ticker, **check_trend(df, args.min_price),
                     **{f"vcp_{k}": v for k, v in asdict(vcp).items()}})
    if not rows:
        log("No price data.")
        return
    screen = pd.DataFrame(rows)
    screen["pass_trend"] = screen[TREND_RULES].all(axis=1)

    shortlist = screen if args.all else screen[
        screen["pass_trend"] & (screen["vcp_fails"].str.len() <= args.near_miss)]
    log(f"Fetching fundamentals for {len(shortlist)} tickers...")
    fundamentals = []
    for i, ticker in enumerate(shortlist["ticker"], 1):
        log(f"  [{i}/{len(shortlist)}] {ticker}")
        try:
            f = fetch_fundamentals(ticker, prices[ticker], use_cache)
            fundamentals.append({"ticker": ticker, **check_fundamentals(f, args)})
        except Exception as e:
            log(f"    failed: {e}")
        time.sleep(1)

    fund = pd.DataFrame(fundamentals) if fundamentals else pd.DataFrame(columns=["ticker"])
    out = shortlist.merge(fund, on="ticker", how="left")
    for col in FUNDAMENTAL_RULES + REFINEMENT_RULES:
        out[col] = out[col].fillna(False).astype(bool) if col in out else False

    chart_ok = out["pass_trend"] & out["vcp_passed"]
    fund_ok = out[FUNDAMENTAL_RULES].all(axis=1)
    refine_ok = out[REFINEMENT_RULES].all(axis=1)
    almost_vcp = out["pass_trend"] & (out["vcp_fails"].str.len() <= args.near_miss)
    out["status"] = np.select(
        [chart_ok & fund_ok & refine_ok, chart_ok & fund_ok, almost_vcp & fund_ok & refine_ok],
        ["FINAL", "watch", "near-miss"], default="rejected")
    out["status"] = pd.Categorical(out["status"], STATUS_ORDER, ordered=True)
    out["vcp_fails"] = out["vcp_fails"].str.join(",")
    out = out.sort_values(["status", "vcp_score"], ascending=[True, False])

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"screen_{date.today()}_{args.timeframe}.csv"
    out.to_csv(path, index=False)

    counts = {
        "universe": len(tickers),
        "2y of price history": len(prices),
        "trend": int(screen["pass_trend"].sum()),
        "trend + VCP": int((screen["pass_trend"] & screen["vcp_passed"]).sum()),
        "trend + VCP + fundamentals": int((chart_ok & fund_ok).sum()),
        "all rules": int((out["status"] == "FINAL").sum()),
    }
    print("Stage counts:")
    for name, n in counts.items():
        print(f"  {name:28s} {n}")

    cols = ["ticker", "status", "name", "price", "market_cap_b", "vcp_score", "vcp_contractions",
            "vcp_depths_pct", "vcp_below_ceiling_pct", "vcp_fails", "pe_slope", "insiders_pct",
            "fcf_slope", "eps_yoy_pct"]
    cols = [c for c in cols if c in out]
    candidates = out[out["status"] != "rejected"]
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 28)
    print()
    print(candidates[cols].to_string(index=False) if len(candidates) else "No candidates today.")
    print(f"\nFull results: {path}")
    print("Not financial advice. Check every chart yourself before buying anything.")


if __name__ == "__main__":
    main()
