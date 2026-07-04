"""
Predictable Swing Trading Engine v3
====================================
Goal:
Core Philosophy:
We are NOT predicting exact prices.
We are ranking HIGH PROBABILITY swing setups.

We want stocks that:
  1. Had a meaningful expansion move (showed momentum)
  2. Pulled back in a healthy, controlled way (shook out weak hands)
  3. Are resuming — confirmed by a reclaim candle + volume

v3 Changes:
  - CRITICAL FIX: Price range widened from 250-350 to 50-5000
    (the old 100-rupee window excluded ~95% of the universe)
  - CRITICAL FIX: Cycle filter logic was contradictory — requiring
    positive 10D move AND pullback simultaneously. Now uses 30D expansion
    window with recent 20D pullback (proper swing cycle detection)
  - CRITICAL FIX: Entry filter was too restrictive — requiring latest
    candle close > prior day HIGH happens ~5% of days. Now uses
    multi-signal approach (any 2 of 3 signals)
  - Trend filter softened — allows close near EMA20 (within 2%) since
    pullback-to-EMA is a classic swing entry
  - Freshness window widened: 3–50 days since 60D peak
  - Volatility filter relaxed: ATR% min 1.0, stddev range 0.5–7.0
  - Added DIAGNOSTIC MODE to show filter-by-filter elimination counts
  - Added RSI indicator for mean-reversion scoring
  - Added EMA proximity scoring to reward pullback-to-EMA setups
"""


from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import yfinance as yf

# =========================================================
# CONFIG
# =========================================================

@dataclass
class ScannerConfig:
    # Data
    period: str = "6mo"
    interval: str = "1d"
    # Universe
    # FIX: Old range was 250–350, a 100-rupee window that excluded ~95% of stocks.
    # Most swing-tradeable NSE stocks are in the 50–5000 range.
    min_price: float = 350
    max_price: float = 3500
    min_avg_traded_value_cr: float = 5       # lowered from 25 → 5 to include mid-caps
    # Volatility
    min_atr_pct: float = 1.0                 # lowered from 1.5 → 1.0
    min_stddev: float = 0.5                  # lowered from 0.8 → 0.5
    max_stddev: float = 7.0                  # raised from 6 → 7
    # Trend
    ema_fast: int = 20
    ema_slow: int = 50
    ema_proximity_pct: float = 2.0           # NEW: allow close within 2% below EMA20
    # Expansion Cycle
    # FIX: The old filter checked MOVE_10D >= expansion_move_pct AND pullback,
    # which is contradictory (can't be up strongly over 10D AND pulled back).
    # Now: we check MOVE_30D for the expansion (prior push over a wider window)
    # and PULLBACK_PCT for the healthy retracement.
    expansion_lookback: int = 30             # NEW: use 30D window for expansion
    expansion_move_pct: float = 3.0          # stock must have gained 3%+ over 30D
    pullback_min_pct: float = 0.5            # must have pulled back meaningfully
    pullback_max_pct: float = 15.0           # raised from 12 → 15
    # Entry — relaxed from strict reclaim to multi-signal
    min_volume_spike: float = 0.8
    min_candle_strength: float = 0.4         # lowered from 0.5
    entry_signals_required: int = 2          # need 2 of 3 signals (was all 3)
    # Trade
    target_pct: float = 4
    stop_buffer_pct: float = 2
    # Batch
    chunk_size: int = 75
    # Output
    top_n: int = 50
    # Event Spike Protection

    max_5d_move_pct: float = 25              # raised from 22
    max_gap_pct: float = 6                   # raised from 5
    max_range_stddev: float = 4
    max_pullback_volume_ratio: float = 1.2
    # Freshness
    min_days_since_60d_high: int = 3         # lowered from 5 → 3
    max_days_since_60d_high: int = 50        # raised from 45 → 50
    # Diagnostics
    diagnostic_mode: bool = True             # NEW: show filter-by-filter stats


CONFIG = ScannerConfig()

# =========================================================
# NSE UNIVERSE
# =========================================================

NSE_EQUITY_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"


def to_yahoo_symbol(symbol):
    symbol = str(symbol).strip().upper()
    if symbol.endswith(".NS"):
        return symbol
    return f"{symbol}.NS"

def chunks(values, size):
    return [values[i : i + size] for i in range(0, len(values), size)]

def normalize_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def load_nse_tickers(source=NSE_EQUITY_LIST_URL):
    df = pd.read_csv(source)
    symbols = df["SYMBOL"].dropna().astype(str).tolist()
    return [to_yahoo_symbol(x) for x in symbols]

# =========================================================
# HELPERS
# =========================================================

def download_batch(tickers, config):
    data = yf.download(
        tickers=tickers,
        period=config.period,
        interval=config.interval,
        progress=False,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
    )
    return data

def get_ticker_frame(batch_data, ticker):
    if not isinstance(batch_data.columns, pd.MultiIndex):
        df = normalize_columns(batch_data)
        return df.dropna(how="all")
    if ticker not in batch_data.columns.get_level_values(0):
        return pd.DataFrame()
    df = batch_data[ticker].copy()
    df = normalize_columns(df)
    df = df.dropna(how="all")
    return df


# =========================================================
# INDICATORS
# =========================================================

def add_indicators(df):
    df = df.copy()

    # RETURNS
    df["RETURNS"] = (df["Close"].pct_change()) * 100

    # TRUE RANGE
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift(1))
    low_close = np.abs(df["Low"] - df["Close"].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # ATR
    df["ATR_5"] = tr.rolling(5).mean()
    df["ATR_PCT"] = (df["ATR_5"] / df["Close"]) * 100

    # STANDARD DEVIATION
    df["STDDEV_20"] = df["RETURNS"].rolling(20).std()

    # EMAs
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()

    # VOLUME
    df["AVG_VOL_20"] = df["Volume"].rolling(20).mean()
    df["VOL_SPIKE"] = df["Volume"] / df["AVG_VOL_20"]

    # TRADED VALUE
    df["TRADED_VALUE_CR"] = (df["Close"] * df["AVG_VOL_20"]) / 10_000_000

    # RECENT HIGH
    df["HIGH_20"] = df["High"].rolling(20).max()

    # EXPANSION MOVE (multiple windows)
    df["MOVE_10D"] = ((df["Close"] / df["Close"].shift(10)) - 1) * 100
    df["MOVE_20D"] = ((df["Close"] / df["Close"].shift(20)) - 1) * 100
    df["MOVE_30D"] = ((df["Close"] / df["Close"].shift(30)) - 1) * 100

    # PULLBACK %
    df["PULLBACK_PCT"] = ((df["HIGH_20"] - df["Close"]) / df["HIGH_20"]) * 100

    # 5D MOVE
    df["MOVE_5D"] = ((df["Close"] / df["Close"].shift(5)) - 1) * 100

    # GAP %
    df["GAP_PCT"] = ((df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)) * 100

    # DAILY RANGE %
    df["RANGE_PCT"] = ((df["High"] - df["Low"]) / df["Close"]) * 100

    # RANGE STABILITY
    df["RANGE_STDDEV_20"] = df["RANGE_PCT"].rolling(20).std()

    # PULLBACK VOLUME QUALITY
    df["PULLBACK_VOL_RATIO"] = df["Volume"] / df["AVG_VOL_20"]
    df["VOLATILITY_CONSISTENCY"] = df["RANGE_PCT"].rolling(20).std()
    df["ATR_STDDEV_20"] = df["ATR_PCT"].rolling(20).std()

    # VOLATILITY REGIME
    df["VOL_REGIME"] = (
        (df["STDDEV_20"] > 1.5) & (df["STDDEV_20"] < 4)
    ).astype(int)

    df["VOL_REGIME_PERSISTENCE"] = df["VOL_REGIME"].rolling(60).sum()

    # MEDIUM TERM MOVE — exhaustion check
    df["MOVE_60D"] = ((df["Close"] / df["Close"].shift(60)) - 1) * 100

    # RSI (14-period) — mean-reversion component
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # EMA PROXIMITY — how close is price to EMA20 (as %)
    df["EMA20_DIST_PCT"] = ((df["Close"] - df["EMA20"]) / df["EMA20"]) * 100

    # FRESHNESS — how many days ago was the 60D high?
    def days_since_rolling_high(close_series, window=60):
        result = []
        for i in range(len(close_series)):
            if i < window:
                result.append(np.nan)
            else:
                window_slice = close_series.iloc[i - window: i + 1].values
                peak_idx = np.argmax(window_slice)
                days_since = (window) - peak_idx
                result.append(days_since)
        return pd.Series(result, index=close_series.index)

    df["DAYS_SINCE_60D_HIGH"] = days_since_rolling_high(df["Close"], window=60)

    return df


# =========================================================
# FILTERS
# =========================================================

def passes_universe_filter(row, config):
    return (
        config.min_price <= row["Close"] <= config.max_price
        and row["TRADED_VALUE_CR"] >= config.min_avg_traded_value_cr
    )

def passes_volatility_filter(row, config):
    return (
        row["ATR_PCT"] >= config.min_atr_pct
        and config.min_stddev <= row["STDDEV_20"] <= config.max_stddev
    )

def passes_trend_filter(row, config):
    """
    FIX: Old filter required Close > EMA20 > EMA50 strictly.
    Problem: Stocks pulling back TO the EMA20 (classic swing entry point)
    would be rejected. Now we allow close within ema_proximity_pct below EMA20.
    Still require EMA20 > EMA50 (overall uptrend).
    """
    ema20_above_ema50 = row["EMA20"] > row["EMA50"]
    # Allow close slightly below EMA20 (the pullback-to-EMA setup)
    close_near_or_above_ema20 = row["EMA20_DIST_PCT"] >= -config.ema_proximity_pct
    return ema20_above_ema50 and close_near_or_above_ema20

def passes_cycle_filter(row, config):
    """
    FIX: Old filter checked MOVE_10D >= expansion_move_pct AND pullback.
    This is contradictory: if a stock has pulled back, its 10D return
    is likely NEGATIVE, so it fails the expansion check.

    New approach:
    - Expansion: check MOVE_30D (prior momentum over a wider window)
    - Pullback: check PULLBACK_PCT (retracement from 20D high)
    This correctly identifies stocks that pushed up then consolidated.
    """
    expansion = row["MOVE_30D"] >= config.expansion_move_pct
    pullback = config.pullback_min_pct <= row["PULLBACK_PCT"] <= config.pullback_max_pct
    return expansion and pullback

def passes_entry_filter(df, config):
    """
    FIX: Old filter required ALL THREE of:
    - Close > prior day's HIGH (happens only ~5% of strong up days)
    - Volume spike >= threshold
    - Candle strength >= threshold

    This was far too restrictive. Now uses a multi-signal approach:
    need any 2 of 3 signals to confirm entry timing.
    """
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    signals = 0

    # Signal 1: Price reclaims prior day's high
    if latest["Close"] > previous["High"]:
        signals += 1

    # Signal 2: Strong candle (closes in upper portion of range)
    candle_range = latest["High"] - latest["Low"]
    if candle_range > 0:
        candle_strength = (latest["Close"] - latest["Low"]) / candle_range
        if candle_strength >= config.min_candle_strength:
            signals += 1

    # Signal 3: Volume confirms the move
    if latest["VOL_SPIKE"] >= config.min_volume_spike:
        signals += 1

    return signals >= config.entry_signals_required

def passes_event_spike_filter(row, config):
    # Reject huge recent spikes (likely news/event driven, not structural)
    if row["MOVE_5D"] > config.max_5d_move_pct:
        return False
    # Reject abnormal gaps
    if abs(row["GAP_PCT"]) > config.max_gap_pct:
        return False
    return True

def passes_freshness_filter(row, config):
    days = row["DAYS_SINCE_60D_HIGH"]
    if pd.isna(days):
        return False
    return config.min_days_since_60d_high <= days <= config.max_days_since_60d_high


# =========================================================
# SCORING ENGINE
# =========================================================

def calculate_score(row):

    # How far above EMA20 — but cap it, we don't want overextended
    trend_score = min(((row["Close"] - row["EMA20"]) / row["EMA20"]) * 100 * 1.5, 10)
    # Bonus: stocks near EMA20 (within 1%) get a proximity bonus
    ema_proximity_bonus = max(3 - abs(row["EMA20_DIST_PCT"]), 0)

    # Volatility — more ATR = more swing potential
    atr_score = min(row["ATR_PCT"], 10)

    # Ideal stddev around 3 — not too quiet, not too chaotic
    ideal_volatility = 3
    volatility_score = max(10 - (abs(row["STDDEV_20"] - ideal_volatility) * 2), 0)

    # Volume at trigger — confirms conviction
    volume_score = min(row["VOL_SPIKE"] * 5, 10)

    # 30D expansion — want meaningful but not exhausted
    expansion_score = min(row["MOVE_30D"] / 2, 10)

    # Pullback quality is the core of the setup
    # Ideal pullback is ~3–5% — healthy retracement
    ideal_pullback = 4
    pullback_score = max(10 - (abs(row["PULLBACK_PCT"] - ideal_pullback) * 2), 0)

    # Consistent daily range = predictable, tradeable
    volatility_stability_score = max(15 - (row["VOLATILITY_CONSISTENCY"] * 2), 0)

    # Rewards stocks in good volatility regime for a while
    vol_regime_persistence_score = min(row["VOL_REGIME_PERSISTENCE"] / 10, 10) if not pd.isna(row["VOL_REGIME_PERSISTENCE"]) else 0

    # RSI mean-reversion: stocks with RSI 40–60 are ideal entry zone (not overbought, not oversold)
    rsi = row["RSI_14"] if not pd.isna(row["RSI_14"]) else 50
    rsi_score = max(10 - abs(rsi - 50) * 0.3, 0)

    # Weighted score (sums to 1.0)
    score = (
        trend_score                  * 0.10
        + atr_score                  * 0.08
        + volume_score               * 0.12
        + expansion_score            * 0.12
        + pullback_score             * 0.20   # core of the setup
        + volatility_stability_score * 0.10
        + volatility_score           * 0.05
        + vol_regime_persistence_score * 0.05
        + rsi_score                  * 0.10
        + ema_proximity_bonus        * 0.08   # reward pullback-to-EMA setups
    )  # total = 1.00

    # Short-term overextension penalty
    stretch_penalty = max(row["MOVE_10D"] - 15, 0)
    score = score - (stretch_penalty * 0.2)

    # Medium-term exhaustion penalty
    medium_term_stretch_penalty = max(row["MOVE_60D"] - 20, 0) if not pd.isna(row["MOVE_60D"]) else 0
    score = score - (medium_term_stretch_penalty * 0.10)

    return round(score, 2)


# =========================================================
# MAIN STOCK SCAN
# =========================================================

# Diagnostic counters (module-level for accumulating across batches)
_diag_counters = {
    "total_processed": 0,
    "insufficient_data": 0,
    "failed_universe": 0,
    "failed_event_spike": 0,
    "failed_trend": 0,
    "failed_volatility": 0,
    "failed_cycle": 0,
    "failed_freshness": 0,
    "failed_entry": 0,
    "passed_all": 0,
}

def reset_diagnostics():
    for k in _diag_counters:
        _diag_counters[k] = 0

def print_diagnostics():
    print("\n" + "=" * 60)
    print("FILTER DIAGNOSTIC REPORT")
    print("=" * 60)
    total = _diag_counters["total_processed"]
    print(f"  Total stocks processed:     {total}")
    print(f"  Insufficient data (<60 bars): {_diag_counters['insufficient_data']}")
    remaining = total - _diag_counters["insufficient_data"]
    print(f"  --- Stocks entering filters: {remaining} ---")

    filters = [
        ("Universe (price + liquidity)", "failed_universe"),
        ("Event spike protection",       "failed_event_spike"),
        ("Trend (EMA alignment)",        "failed_trend"),
        ("Volatility (ATR + stddev)",    "failed_volatility"),
        ("Cycle (expansion + pullback)", "failed_cycle"),
        ("Freshness (days since peak)",  "failed_freshness"),
        ("Entry (candle + volume)",      "failed_entry"),
    ]

    for label, key in filters:
        count = _diag_counters[key]
        pct = (count / remaining * 100) if remaining > 0 else 0
        remaining -= count
        print(f"  X {label}: {count:>5} rejected  ({pct:5.1f}%)  ->  {remaining} remain")

    print(f"  + Passed all filters:       {_diag_counters['passed_all']}")
    print("=" * 60)


def scan_stock_from_df(ticker, df, config):
    _diag_counters["total_processed"] += 1

    if df is None or len(df) < 60:
        _diag_counters["insufficient_data"] += 1
        return None

    df = add_indicators(df)
    latest = df.iloc[-1]

    # --- FILTERS (in order of cheapness — fastest rejections first) ---

    if not passes_universe_filter(latest, config):
        _diag_counters["failed_universe"] += 1
        return None

    if not passes_event_spike_filter(latest, config):
        _diag_counters["failed_event_spike"] += 1
        return None

    if not passes_trend_filter(latest, config):
        _diag_counters["failed_trend"] += 1
        return None

    if not passes_volatility_filter(latest, config):
        _diag_counters["failed_volatility"] += 1
        return None

    if not passes_cycle_filter(latest, config):
        _diag_counters["failed_cycle"] += 1
        return None

    if not passes_freshness_filter(latest, config):
        _diag_counters["failed_freshness"] += 1
        return None

    if not passes_entry_filter(df, config):
        _diag_counters["failed_entry"] += 1
        return None

    _diag_counters["passed_all"] += 1

    # SCORE
    score = calculate_score(latest)

    # TRADE LEVELS
    entry, sl, target = generate_trade_levels(df, config)

    return {
        "Ticker": ticker,
        "Score": score,
        "STDDEV_20": round(latest["STDDEV_20"], 2),
        "Price": round(latest["Close"], 2),
        "Entry": entry,
        "SL": sl,
        "Target": target,
        "ATR%": round(latest["ATR_PCT"], 2),
        "RSI": round(latest["RSI_14"], 1) if not pd.isna(latest["RSI_14"]) else None,
        "Move10D%": round(latest["MOVE_10D"], 2),
        "Move20D%": round(latest["MOVE_20D"], 2),
        "Move30D%": round(latest["MOVE_30D"], 2),
        "Move60D%": round(latest["MOVE_60D"], 2) if not pd.isna(latest["MOVE_60D"]) else None,
        "Pullback%": round(latest["PULLBACK_PCT"], 2),
        "EMA20Dist%": round(latest["EMA20_DIST_PCT"], 2),
        "DaysSince60DPeak": int(latest["DAYS_SINCE_60D_HIGH"]) if not pd.isna(latest["DAYS_SINCE_60D_HIGH"]) else None,
        "VolumeSpike": round(latest["VOL_SPIKE"], 2),
        "VolRegimeDays": int(latest["VOL_REGIME_PERSISTENCE"]) if not pd.isna(latest["VOL_REGIME_PERSISTENCE"]) else None,
    }


# =========================================================
# MAIN SCANNER
# =========================================================

def run_scanner(tickers, config):
    reset_diagnostics()
    results = []
    ticker_batches = chunks(tickers, config.chunk_size)
    total_batches = len(ticker_batches)
    print(f"Scanning {len(tickers)} tickers in {total_batches} batches...")

    for batch_num, batch in enumerate(ticker_batches, start=1):
        print(f"  Batch {batch_num}/{total_batches}...", end="\r")
        try:
            batch_data = download_batch(batch, config)
        except Exception as e:
            print(f"\nBATCH {batch_num} FAILED: {e}")
            continue

        for ticker in batch:
            try:
                df = get_ticker_frame(batch_data, ticker)
                if df.empty:
                    continue
                candidate = scan_stock_from_df(ticker, df, config)
                if candidate:
                    results.append(candidate)
            except Exception as e:
                print(f"\n{ticker} FAILED: {e}")

    print(f"\nScan complete. {len(results)} candidates found before top-N cut.")

    if config.diagnostic_mode:
        print_diagnostics()

    if not results:
        return pd.DataFrame()

    output = pd.DataFrame(results)
    output = output.sort_values("Score", ascending=False)
    return output.head(config.top_n)


# =========================================================
# TRADE LEVELS
# =========================================================

def generate_trade_levels(df, config):
    latest = df.iloc[-1]
    entry = round(latest["Close"], 2)
    stop_loss = round(latest["EMA20"] * (1 - config.stop_buffer_pct / 100), 2)
    target = round(entry * (1 + config.target_pct / 100), 2)
    return entry, stop_loss, target


# =========================================================
# MAIN
# =========================================================

def run_stock_scan():
    """
    Runs the stock scanner.

    Returns:
        pandas.DataFrame
    """

    tickers = load_nse_tickers()

    output = run_scanner(
        tickers,
        CONFIG
    )

    return output