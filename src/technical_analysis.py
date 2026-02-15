import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd


@dataclass
class TechnicalSnapshot:
    """Complete technical analysis snapshot for a symbol at a point in time."""

    symbol: str
    timestamp: str

    # Price
    current_price: float
    prev_close: float
    change_percent: float

    # RSI
    rsi_14: float
    rsi_signal: str  # "oversold" / "overbought" / "neutral"

    # MACD
    macd_line: float
    macd_signal: float
    macd_histogram: float
    macd_crossover: str  # "bullish_crossover" / "bearish_crossover" / "none"

    # Moving Averages
    sma_20: float
    sma_50: float
    sma_200: float
    price_vs_sma20: str  # "above" / "below"
    price_vs_sma50: str
    price_vs_sma200: str
    golden_cross: bool
    death_cross: bool

    # Bollinger Bands
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    bb_position: str  # "above_upper" / "below_lower" / "within"

    # Volume
    current_volume: int
    avg_volume_20: float
    volume_ratio: float
    volume_signal: str  # "high" / "normal" / "low"

    # Support & Resistance
    support_level: float
    resistance_level: float
    distance_to_support: float
    distance_to_resistance: float

    # Trend
    trend: str  # "strong_uptrend" / "uptrend" / "sideways" / "downtrend" / "strong_downtrend"
    trend_strength: float

    # --- NEW: ADX ---
    adx: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None
    adx_signal: Optional[str] = None  # "strong_trend" / "weak_trend" / "no_trend"

    # --- NEW: Stochastic RSI ---
    stoch_rsi_k: Optional[float] = None
    stoch_rsi_d: Optional[float] = None
    stoch_rsi_signal: Optional[str] = None  # "oversold" / "overbought" / "neutral"

    # --- NEW: ATR ---
    atr_14: Optional[float] = None
    atr_percent: Optional[float] = None  # ATR as % of price (volatility measure)

    # --- NEW: VWAP ---
    vwap: Optional[float] = None
    price_vs_vwap: Optional[str] = None  # "above" / "below"

    # --- NEW: Fibonacci Retracement ---
    fib_levels: Optional[Dict[str, float]] = None  # {level: price}
    nearest_fib_support: Optional[float] = None
    nearest_fib_resistance: Optional[float] = None

    # --- NEW: Ichimoku Cloud ---
    ichimoku_tenkan: Optional[float] = None
    ichimoku_kijun: Optional[float] = None
    ichimoku_senkou_a: Optional[float] = None
    ichimoku_senkou_b: Optional[float] = None
    ichimoku_signal: Optional[str] = None  # "above_cloud" / "below_cloud" / "in_cloud"

    # --- NEW: OBV ---
    obv: Optional[float] = None
    obv_sma_20: Optional[float] = None
    obv_signal: Optional[str] = None  # "bullish" / "bearish" / "neutral"

    # --- NEW: Pivot Points ---
    pivot_point: Optional[float] = None
    pivot_r1: Optional[float] = None
    pivot_r2: Optional[float] = None
    pivot_r3: Optional[float] = None
    pivot_s1: Optional[float] = None
    pivot_s2: Optional[float] = None
    pivot_s3: Optional[float] = None

    # --- NEW: Candlestick Patterns ---
    candlestick_pattern: Optional[str] = None  # e.g. "bullish_engulfing", "doji", "hammer"
    candlestick_bias: Optional[str] = None  # "bullish" / "bearish" / "neutral"

    # --- NEW: Relative Strength vs NIFTY ---
    rs_vs_nifty: Optional[float] = None  # ratio > 1 = outperforming
    rs_signal: Optional[str] = None  # "outperforming" / "underperforming" / "neutral"

    # --- NEW: 52-week metrics ---
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    pct_from_52w_high: Optional[float] = None
    pct_from_52w_low: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class TechnicalAnalysis:
    """Stateless technical indicator calculator.

    All methods are static/class methods that take pandas Series as input
    and return computed values. Designed to work with the DataBuffer's
    200-day daily OHLCV data.
    """

    # -------------------------------------------------------------------------
    # EXISTING indicators
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_rsi(closes: pd.Series, period: int = 14) -> float:
        """Calculate RSI using the Exponential Weighted Moving Average method."""
        if len(closes) < period + 1:
            return 50.0

        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

        last_avg_loss = avg_loss.iloc[-1]
        if last_avg_loss == 0:
            return 100.0

        rs = avg_gain.iloc[-1] / last_avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)

    @staticmethod
    def calculate_macd(
        closes: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[float, float, float]:
        """Calculate MACD line, signal line, and histogram."""
        if len(closes) < slow + signal:
            return 0.0, 0.0, 0.0

        ema_fast = closes.ewm(span=fast, adjust=False).mean()
        ema_slow = closes.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return (
            float(macd_line.iloc[-1]),
            float(signal_line.iloc[-1]),
            float(histogram.iloc[-1]),
        )

    @staticmethod
    def calculate_sma(closes: pd.Series, period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(closes) < period:
            return float(closes.mean()) if len(closes) > 0 else 0.0
        return float(closes.tail(period).mean())

    @staticmethod
    def calculate_bollinger_bands(
        closes: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands (upper, middle, lower)."""
        if len(closes) < period:
            return 0.0, 0.0, 0.0

        window = closes.tail(period)
        middle = float(window.mean())
        std = float(window.std())
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        return upper, middle, lower

    @staticmethod
    def calculate_support_resistance(
        highs: pd.Series, lows: pd.Series, closes: pd.Series, lookback: int = 20
    ) -> Tuple[float, float]:
        """Calculate support and resistance levels from recent price action."""
        if len(highs) < lookback:
            if len(highs) > 0:
                return float(lows.min()), float(highs.max())
            return 0.0, 0.0

        current_price = float(closes.iloc[-1])
        recent_highs = highs.tail(lookback)
        recent_lows = lows.tail(lookback)

        support_candidates = recent_lows[recent_lows <= current_price]
        if len(support_candidates) > 0:
            support = float(support_candidates.max())
        else:
            support = float(recent_lows.min())

        resistance_candidates = recent_highs[recent_highs >= current_price]
        if len(resistance_candidates) > 0:
            resistance = float(resistance_candidates.min())
        else:
            resistance = float(recent_highs.max())

        return support, resistance

    @staticmethod
    def detect_trend(
        closes: pd.Series, sma_20: float, sma_50: float, sma_200: float
    ) -> Tuple[str, float]:
        """Detect trend direction using multiple moving average alignment."""
        if len(closes) == 0:
            return "sideways", 50.0

        current_price = float(closes.iloc[-1])
        score = 0.0

        if current_price > sma_20:
            score += 25
        if current_price > sma_50:
            score += 25
        if current_price > sma_200:
            score += 20
        if sma_20 > sma_50:
            score += 15
        if sma_50 > sma_200:
            score += 15

        if score >= 85:
            return "strong_uptrend", score
        elif score >= 60:
            return "uptrend", score
        elif score >= 40:
            return "sideways", score
        elif score >= 20:
            return "downtrend", score
        else:
            return "strong_downtrend", score

    @staticmethod
    def detect_macd_crossover(
        closes: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> str:
        """Detect MACD crossover by comparing current vs previous histogram sign."""
        if len(closes) < slow + signal + 1:
            return "none"

        ema_fast = closes.ewm(span=fast, adjust=False).mean()
        ema_slow = closes.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        if len(histogram) < 2:
            return "none"

        prev_hist = float(histogram.iloc[-2])
        curr_hist = float(histogram.iloc[-1])

        if prev_hist <= 0 and curr_hist > 0:
            return "bullish_crossover"
        elif prev_hist >= 0 and curr_hist < 0:
            return "bearish_crossover"
        return "none"

    # -------------------------------------------------------------------------
    # NEW professional indicators
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_adx(highs: pd.Series, lows: pd.Series, closes: pd.Series,
                      period: int = 14) -> Tuple[float, float, float]:
        """Calculate Average Directional Index (ADX) with DI+ and DI-.

        Returns: (adx, plus_di, minus_di)
        """
        if len(closes) < period * 2:
            return 0.0, 0.0, 0.0

        high = highs.values.astype(float)
        low = lows.values.astype(float)
        close = closes.values.astype(float)

        # True Range
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))

        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Smoothed using Wilder's method (EMA with alpha=1/period)
        # Formula: New = Prior * (n-1)/n + Current/n = Prior - Prior/n + Current/n
        def wilder_smooth(data, n):
            result = np.zeros_like(data, dtype=float)
            if len(data) < n:
                return result
            # Initial value is SMA of first n periods
            result[n - 1] = np.mean(data[:n])
            for i in range(n, len(data)):
                result[i] = result[i - 1] * (n - 1) / n + data[i] / n
            return result

        atr_smooth = wilder_smooth(tr, period)
        plus_dm_smooth = wilder_smooth(plus_dm, period)
        minus_dm_smooth = wilder_smooth(minus_dm, period)

        # DI+ and DI- (with NaN protection)
        with np.errstate(divide='ignore', invalid='ignore'):
            plus_di = np.where(atr_smooth > 0, 100 * plus_dm_smooth / atr_smooth, 0.0)
            minus_di = np.where(atr_smooth > 0, 100 * minus_dm_smooth / atr_smooth, 0.0)
            plus_di = np.nan_to_num(plus_di, nan=0.0, posinf=100.0, neginf=0.0)
            minus_di = np.nan_to_num(minus_di, nan=0.0, posinf=100.0, neginf=0.0)

            # DX and ADX
            di_sum = plus_di + minus_di
            dx = np.where(di_sum > 0, 100 * np.abs(plus_di - minus_di) / di_sum, 0.0)
            dx = np.nan_to_num(dx, nan=0.0, posinf=100.0, neginf=0.0)

        adx = wilder_smooth(dx[period - 1:], period)

        if len(adx) == 0:
            return 0.0, 0.0, 0.0

        # Clamp ADX to valid range (0-100)
        adx_value = min(100.0, max(0.0, float(adx[-1])))
        plus_di_value = min(100.0, max(0.0, float(plus_di[-1])))
        minus_di_value = min(100.0, max(0.0, float(minus_di[-1])))

        return adx_value, plus_di_value, minus_di_value

    @staticmethod
    def calculate_stochastic_rsi(closes: pd.Series, rsi_period: int = 14,
                                  stoch_period: int = 14, k_smooth: int = 3,
                                  d_smooth: int = 3) -> Tuple[float, float]:
        """Calculate Stochastic RSI (%K and %D).

        Returns: (stoch_rsi_k, stoch_rsi_d)
        """
        if len(closes) < rsi_period + stoch_period + d_smooth:
            return 50.0, 50.0

        # Compute RSI series
        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period).mean()
        avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi_series = 100 - (100 / (1 + rs))
        rsi_series = rsi_series.fillna(50.0)

        # Stochastic of RSI
        rsi_min = rsi_series.rolling(window=stoch_period).min()
        rsi_max = rsi_series.rolling(window=stoch_period).max()
        rsi_range = rsi_max - rsi_min
        stoch_rsi = ((rsi_series - rsi_min) / rsi_range.replace(0, np.nan)).fillna(0.5) * 100

        k = stoch_rsi.rolling(window=k_smooth).mean()
        d = k.rolling(window=d_smooth).mean()

        return float(k.iloc[-1]), float(d.iloc[-1])

    @staticmethod
    def calculate_atr(highs: pd.Series, lows: pd.Series, closes: pd.Series,
                      period: int = 14) -> float:
        """Calculate Average True Range (ATR)."""
        if len(closes) < period + 1:
            return 0.0

        high = highs.values.astype(float)
        low = lows.values.astype(float)
        close = closes.values.astype(float)

        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))

        # Wilder's smoothing
        atr = pd.Series(tr).ewm(alpha=1 / period, min_periods=period).mean()
        return float(atr.iloc[-1])

    @staticmethod
    def calculate_vwap(highs: pd.Series, lows: pd.Series, closes: pd.Series,
                       volumes: pd.Series) -> float:
        """Calculate VWAP from intraday or daily data.

        Uses typical price (H+L+C)/3 weighted by volume.
        """
        if len(closes) == 0 or volumes.sum() == 0:
            return 0.0

        typical_price = (highs + lows + closes) / 3
        cum_vol = volumes.cumsum()
        cum_tp_vol = (typical_price * volumes).cumsum()

        vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
        return float(vwap.iloc[-1]) if not np.isnan(vwap.iloc[-1]) else 0.0

    @staticmethod
    def calculate_fibonacci_levels(highs: pd.Series, lows: pd.Series,
                                    lookback: int = 50) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels from recent swing high/low.

        Returns dict with keys: '0%', '23.6%', '38.2%', '50%', '61.8%', '78.6%', '100%'
        """
        if len(highs) < lookback:
            lookback = len(highs)
        if lookback < 5:
            return {}

        recent_high = float(highs.tail(lookback).max())
        recent_low = float(lows.tail(lookback).min())
        diff = recent_high - recent_low

        if diff <= 0:
            return {}

        return {
            "0%": recent_high,
            "23.6%": recent_high - 0.236 * diff,
            "38.2%": recent_high - 0.382 * diff,
            "50%": recent_high - 0.5 * diff,
            "61.8%": recent_high - 0.618 * diff,
            "78.6%": recent_high - 0.786 * diff,
            "100%": recent_low,
        }

    @staticmethod
    def calculate_ichimoku(highs: pd.Series, lows: pd.Series,
                           tenkan_period: int = 9, kijun_period: int = 26,
                           senkou_b_period: int = 52) -> Tuple[float, float, float, float]:
        """Calculate Ichimoku Cloud components.

        Returns: (tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b)
        """
        if len(highs) < senkou_b_period:
            return 0.0, 0.0, 0.0, 0.0

        def midpoint(h, l, period):
            return (h.tail(period).max() + l.tail(period).min()) / 2

        tenkan = midpoint(highs, lows, tenkan_period)
        kijun = midpoint(highs, lows, kijun_period)
        senkou_a = (tenkan + kijun) / 2
        senkou_b = midpoint(highs, lows, senkou_b_period)

        return float(tenkan), float(kijun), float(senkou_a), float(senkou_b)

    @staticmethod
    def calculate_obv(closes: pd.Series, volumes: pd.Series) -> Tuple[float, float]:
        """Calculate On-Balance Volume and its 20-day SMA.

        Returns: (obv, obv_sma_20)
        """
        if len(closes) < 2:
            return 0.0, 0.0

        direction = np.sign(closes.diff().fillna(0).values)
        obv_values = (direction * volumes.values).cumsum()
        obv_series = pd.Series(obv_values)

        obv_current = float(obv_series.iloc[-1])
        obv_sma = float(obv_series.tail(20).mean()) if len(obv_series) >= 20 else float(obv_series.mean())

        return obv_current, obv_sma

    @staticmethod
    def calculate_pivot_points(high: float, low: float,
                                close: float) -> Dict[str, float]:
        """Calculate classic pivot points from previous day's HLC.

        Returns: dict with PP, R1, R2, R3, S1, S2, S3
        """
        pp = (high + low + close) / 3
        r1 = 2 * pp - low
        s1 = 2 * pp - high
        r2 = pp + (high - low)
        s2 = pp - (high - low)
        r3 = high + 2 * (pp - low)
        s3 = low - 2 * (high - pp)

        return {
            "PP": round(pp, 2),
            "R1": round(r1, 2), "R2": round(r2, 2), "R3": round(r3, 2),
            "S1": round(s1, 2), "S2": round(s2, 2), "S3": round(s3, 2),
        }

    @staticmethod
    def detect_candlestick_patterns(opens: pd.Series, highs: pd.Series,
                                     lows: pd.Series, closes: pd.Series) -> Tuple[str, str]:
        """Detect common candlestick patterns from the last few candles.

        Returns: (pattern_name, bias)
        - bias: "bullish" / "bearish" / "neutral"
        """
        if len(closes) < 3:
            return "none", "neutral"

        o1, o2 = float(opens.iloc[-2]), float(opens.iloc[-1])
        h1, h2 = float(highs.iloc[-2]), float(highs.iloc[-1])
        l1, l2 = float(lows.iloc[-2]), float(lows.iloc[-1])
        c1, c2 = float(closes.iloc[-2]), float(closes.iloc[-1])

        body2 = abs(c2 - o2)
        range2 = h2 - l2
        upper_shadow2 = h2 - max(o2, c2)
        lower_shadow2 = min(o2, c2) - l2

        body1 = abs(c1 - o1)

        # Doji: very small body relative to range
        if range2 > 0 and body2 / range2 < 0.1:
            return "doji", "neutral"

        # Hammer: small body at top, long lower shadow (bullish reversal)
        if range2 > 0 and lower_shadow2 > 2 * body2 and upper_shadow2 < body2 * 0.5:
            if c2 > o2:  # green hammer is stronger
                return "hammer", "bullish"
            return "hammer", "bullish"

        # Shooting Star: small body at bottom, long upper shadow (bearish reversal)
        if range2 > 0 and upper_shadow2 > 2 * body2 and lower_shadow2 < body2 * 0.5:
            return "shooting_star", "bearish"

        # Bullish Engulfing: red candle followed by larger green candle
        if c1 < o1 and c2 > o2 and o2 <= c1 and c2 >= o1:
            return "bullish_engulfing", "bullish"

        # Bearish Engulfing: green candle followed by larger red candle
        if c1 > o1 and c2 < o2 and o2 >= c1 and c2 <= o1:
            return "bearish_engulfing", "bearish"

        # Morning Star (3-candle): big red, small body, big green
        if len(closes) >= 3:
            o0, c0 = float(opens.iloc[-3]), float(closes.iloc[-3])
            body0 = abs(c0 - o0)
            if (c0 < o0 and body0 > 0 and  # first: big red
                body1 < body0 * 0.3 and      # second: small body
                c2 > o2 and body2 > body0 * 0.5 and  # third: big green
                c2 > (o0 + c0) / 2):          # closes above midpoint of first
                return "morning_star", "bullish"

        # Evening Star (3-candle): big green, small body, big red
        if len(closes) >= 3:
            o0, c0 = float(opens.iloc[-3]), float(closes.iloc[-3])
            body0 = abs(c0 - o0)
            if (c0 > o0 and body0 > 0 and  # first: big green
                body1 < body0 * 0.3 and      # second: small body
                c2 < o2 and body2 > body0 * 0.5 and  # third: big red
                c2 < (o0 + c0) / 2):          # closes below midpoint of first
                return "evening_star", "bearish"

        return "none", "neutral"

    @staticmethod
    def calculate_relative_strength(stock_closes: pd.Series,
                                     nifty_closes: pd.Series,
                                     period: int = 20) -> float:
        """Calculate relative strength of stock vs NIFTY over a period.

        Returns ratio where >1 means outperforming, <1 means underperforming.
        """
        if len(stock_closes) < period + 1 or len(nifty_closes) < period + 1:
            return 1.0

        stock_return = float(stock_closes.iloc[-1] / stock_closes.iloc[-period - 1])
        nifty_return = float(nifty_closes.iloc[-1] / nifty_closes.iloc[-period - 1])

        if nifty_return == 0:
            return 1.0

        return stock_return / nifty_return

    # -------------------------------------------------------------------------
    # COMPUTE SNAPSHOT (updated to include new indicators)
    # -------------------------------------------------------------------------

    @classmethod
    def compute_snapshot(
        cls,
        symbol: str,
        daily_closes: pd.Series,
        daily_highs: pd.Series,
        daily_lows: pd.Series,
        daily_volumes: pd.Series,
        current_price: float,
        current_volume: int,
        prev_close: float,
        daily_opens: Optional[pd.Series] = None,
        intraday_df: Optional[pd.DataFrame] = None,
        nifty_closes: Optional[pd.Series] = None,
    ) -> TechnicalSnapshot:
        """Compute a complete technical analysis snapshot for a symbol.

        Args:
            symbol: Stock symbol
            daily_closes: Series of daily close prices (up to 200 days)
            daily_highs: Series of daily highs
            daily_lows: Series of daily lows
            daily_volumes: Series of daily volumes
            current_price: Latest traded price
            current_volume: Today's traded volume
            prev_close: Previous day's closing price
            daily_opens: Series of daily open prices (optional, for candlestick patterns)
            intraday_df: Intraday DataFrame with OHLCV (optional, for VWAP)
            nifty_closes: NIFTY 50 daily closes (optional, for relative strength)
        """
        # --- EXISTING indicators ---

        # RSI
        rsi = cls.calculate_rsi(daily_closes)
        if rsi < 30:
            rsi_signal = "oversold"
        elif rsi > 70:
            rsi_signal = "overbought"
        else:
            rsi_signal = "neutral"

        # MACD
        macd_line, macd_signal_val, macd_hist = cls.calculate_macd(daily_closes)
        macd_crossover = cls.detect_macd_crossover(daily_closes)

        # SMAs
        sma_20 = cls.calculate_sma(daily_closes, 20)
        sma_50 = cls.calculate_sma(daily_closes, 50)
        sma_200 = cls.calculate_sma(daily_closes, 200)

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = cls.calculate_bollinger_bands(daily_closes)
        bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0.0
        if current_price > bb_upper:
            bb_position = "above_upper"
        elif current_price < bb_lower:
            bb_position = "below_lower"
        else:
            bb_position = "within"

        # Volume
        if len(daily_volumes) >= 20:
            avg_vol_20 = float(daily_volumes.tail(20).mean())
        elif len(daily_volumes) > 0:
            avg_vol_20 = float(daily_volumes.mean())
        else:
            avg_vol_20 = 0.0

        vol_ratio = current_volume / avg_vol_20 if avg_vol_20 > 0 else 1.0
        if vol_ratio > 2.0:
            vol_signal = "high"
        elif vol_ratio < 0.5:
            vol_signal = "low"
        else:
            vol_signal = "normal"

        # Support & Resistance
        support, resistance = cls.calculate_support_resistance(
            daily_highs, daily_lows, daily_closes
        )
        dist_support = (
            (current_price - support) / current_price * 100 if current_price > 0 else 0.0
        )
        dist_resistance = (
            (resistance - current_price) / current_price * 100 if current_price > 0 else 0.0
        )

        # Trend
        trend, trend_strength = cls.detect_trend(daily_closes, sma_20, sma_50, sma_200)

        # Golden/Death Cross detection
        golden_cross = False
        death_cross = False
        if len(daily_closes) >= 201:
            prev_sma50 = cls.calculate_sma(daily_closes.iloc[:-1], 50)
            prev_sma200 = cls.calculate_sma(daily_closes.iloc[:-1], 200)
            golden_cross = prev_sma50 <= prev_sma200 and sma_50 > sma_200
            death_cross = prev_sma50 >= prev_sma200 and sma_50 < sma_200

        # Price change
        change_pct = (
            (current_price - prev_close) / prev_close * 100 if prev_close > 0 else 0.0
        )

        # --- NEW indicators ---

        # ADX
        adx_val, plus_di, minus_di = cls.calculate_adx(daily_highs, daily_lows, daily_closes)
        if adx_val > 25:
            adx_signal = "strong_trend"
        elif adx_val > 15:
            adx_signal = "weak_trend"
        else:
            adx_signal = "no_trend"

        # Stochastic RSI
        stoch_k, stoch_d = cls.calculate_stochastic_rsi(daily_closes)
        if stoch_k < 20:
            stoch_signal = "oversold"
        elif stoch_k > 80:
            stoch_signal = "overbought"
        else:
            stoch_signal = "neutral"

        # ATR
        atr_val = cls.calculate_atr(daily_highs, daily_lows, daily_closes)
        atr_pct = (atr_val / current_price * 100) if current_price > 0 else 0.0

        # VWAP (from intraday data if available, else from daily)
        vwap_val = None
        price_vs_vwap = None
        if intraday_df is not None and len(intraday_df) > 0:
            vwap_val = cls.calculate_vwap(
                intraday_df["high"], intraday_df["low"],
                intraday_df["close"], intraday_df["volume"]
            )
        else:
            # Approximate from recent daily data
            vwap_val = cls.calculate_vwap(
                daily_highs.tail(5), daily_lows.tail(5),
                daily_closes.tail(5), daily_volumes.tail(5)
            )
        if vwap_val and vwap_val > 0:
            price_vs_vwap = "above" if current_price > vwap_val else "below"

        # Fibonacci
        fib_levels = cls.calculate_fibonacci_levels(daily_highs, daily_lows)
        nearest_fib_support = None
        nearest_fib_resistance = None
        if fib_levels:
            fib_values = sorted(fib_levels.values(), reverse=True)
            for fv in fib_values:
                if fv < current_price:
                    nearest_fib_support = fv
                    break
            for fv in sorted(fib_values):
                if fv > current_price:
                    nearest_fib_resistance = fv
                    break

        # Ichimoku
        tenkan, kijun, senkou_a, senkou_b = cls.calculate_ichimoku(daily_highs, daily_lows)
        ichimoku_signal = None
        if senkou_a > 0 and senkou_b > 0:
            cloud_top = max(senkou_a, senkou_b)
            cloud_bottom = min(senkou_a, senkou_b)
            if current_price > cloud_top:
                ichimoku_signal = "above_cloud"
            elif current_price < cloud_bottom:
                ichimoku_signal = "below_cloud"
            else:
                ichimoku_signal = "in_cloud"

        # OBV
        obv_val, obv_sma = cls.calculate_obv(daily_closes, daily_volumes)
        if obv_val > obv_sma * 1.05:
            obv_signal = "bullish"
        elif obv_val < obv_sma * 0.95:
            obv_signal = "bearish"
        else:
            obv_signal = "neutral"

        # Pivot Points (from previous day)
        pivot_data = {}
        if len(daily_highs) >= 2:
            prev_h = float(daily_highs.iloc[-2])
            prev_l = float(daily_lows.iloc[-2])
            prev_c = float(daily_closes.iloc[-2])
            pivot_data = cls.calculate_pivot_points(prev_h, prev_l, prev_c)

        # Candlestick Patterns
        candle_pattern = "none"
        candle_bias = "neutral"
        if daily_opens is not None and len(daily_opens) >= 3:
            candle_pattern, candle_bias = cls.detect_candlestick_patterns(
                daily_opens, daily_highs, daily_lows, daily_closes
            )

        # Relative Strength vs NIFTY
        rs_val = None
        rs_signal = None
        if nifty_closes is not None and len(nifty_closes) > 20:
            rs_val = cls.calculate_relative_strength(daily_closes, nifty_closes)
            if rs_val > 1.05:
                rs_signal = "outperforming"
            elif rs_val < 0.95:
                rs_signal = "underperforming"
            else:
                rs_signal = "neutral"

        # 52-week metrics
        high_52w = None
        low_52w = None
        pct_from_52w_high = None
        pct_from_52w_low = None
        if len(daily_highs) >= 50:  # use whatever history we have
            high_52w = float(daily_highs.max())
            low_52w = float(daily_lows.min())
            if high_52w > 0:
                pct_from_52w_high = (current_price - high_52w) / high_52w * 100
            if low_52w > 0:
                pct_from_52w_low = (current_price - low_52w) / low_52w * 100

        return TechnicalSnapshot(
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            current_price=current_price,
            prev_close=prev_close,
            change_percent=round(change_pct, 2),
            rsi_14=round(rsi, 2),
            rsi_signal=rsi_signal,
            macd_line=round(macd_line, 4),
            macd_signal=round(macd_signal_val, 4),
            macd_histogram=round(macd_hist, 4),
            macd_crossover=macd_crossover,
            sma_20=round(sma_20, 2),
            sma_50=round(sma_50, 2),
            sma_200=round(sma_200, 2),
            price_vs_sma20="above" if current_price > sma_20 else "below",
            price_vs_sma50="above" if current_price > sma_50 else "below",
            price_vs_sma200="above" if current_price > sma_200 else "below",
            golden_cross=golden_cross,
            death_cross=death_cross,
            bb_upper=round(bb_upper, 2),
            bb_middle=round(bb_middle, 2),
            bb_lower=round(bb_lower, 2),
            bb_width=round(bb_width, 4),
            bb_position=bb_position,
            current_volume=current_volume,
            avg_volume_20=round(avg_vol_20, 0),
            volume_ratio=round(vol_ratio, 2),
            volume_signal=vol_signal,
            support_level=round(support, 2),
            resistance_level=round(resistance, 2),
            distance_to_support=round(dist_support, 2),
            distance_to_resistance=round(dist_resistance, 2),
            trend=trend,
            trend_strength=trend_strength,
            # NEW fields
            adx=round(adx_val, 2),
            plus_di=round(plus_di, 2),
            minus_di=round(minus_di, 2),
            adx_signal=adx_signal,
            stoch_rsi_k=round(stoch_k, 2),
            stoch_rsi_d=round(stoch_d, 2),
            stoch_rsi_signal=stoch_signal,
            atr_14=round(atr_val, 2),
            atr_percent=round(atr_pct, 2),
            vwap=round(vwap_val, 2) if vwap_val else None,
            price_vs_vwap=price_vs_vwap,
            fib_levels=fib_levels,
            nearest_fib_support=round(nearest_fib_support, 2) if nearest_fib_support else None,
            nearest_fib_resistance=round(nearest_fib_resistance, 2) if nearest_fib_resistance else None,
            ichimoku_tenkan=round(tenkan, 2) if tenkan else None,
            ichimoku_kijun=round(kijun, 2) if kijun else None,
            ichimoku_senkou_a=round(senkou_a, 2) if senkou_a else None,
            ichimoku_senkou_b=round(senkou_b, 2) if senkou_b else None,
            ichimoku_signal=ichimoku_signal,
            obv=round(obv_val, 0),
            obv_sma_20=round(obv_sma, 0),
            obv_signal=obv_signal,
            pivot_point=pivot_data.get("PP"),
            pivot_r1=pivot_data.get("R1"),
            pivot_r2=pivot_data.get("R2"),
            pivot_r3=pivot_data.get("R3"),
            pivot_s1=pivot_data.get("S1"),
            pivot_s2=pivot_data.get("S2"),
            pivot_s3=pivot_data.get("S3"),
            candlestick_pattern=candle_pattern,
            candlestick_bias=candle_bias,
            rs_vs_nifty=round(rs_val, 4) if rs_val else None,
            rs_signal=rs_signal,
            high_52w=round(high_52w, 2) if high_52w else None,
            low_52w=round(low_52w, 2) if low_52w else None,
            pct_from_52w_high=round(pct_from_52w_high, 2) if pct_from_52w_high is not None else None,
            pct_from_52w_low=round(pct_from_52w_low, 2) if pct_from_52w_low is not None else None,
        )


def compute_live_analysis(symbol: str, upstox_client, instrument_resolver,
                          nifty_closes: Optional[pd.Series] = None) -> Optional[TechnicalSnapshot]:
    """On-demand full technical analysis for ANY stock.

    Fetches 200-day historical candles via the broker client, computes all
    indicators, and returns a TechnicalSnapshot. Works for any of the 2400+
    NSE stocks regardless of whether they're in the watchlist.

    Args:
        symbol: Trading symbol (e.g. "RELIANCE")
        upstox_client: UpstoxClient instance
        instrument_resolver: InstrumentResolver instance
        nifty_closes: Optional NIFTY 50 daily closes for relative strength

    Returns:
        TechnicalSnapshot or None on failure
    """
    try:
        # Fetch 200-day candles
        df = upstox_client.get_historical_candles(symbol, interval="day", days=200)
        if df is None or len(df) < 10:
            logging.error(f"Insufficient historical data for {symbol}")
            return None

        current_price = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-2]) if len(df) >= 2 else current_price
        current_volume = int(df["volume"].iloc[-1])

        # If no NIFTY closes provided, fetch them
        if nifty_closes is None:
            nifty_df = upstox_client.get_historical_candles("NIFTY50", interval="day", days=200)
            if nifty_df is not None and len(nifty_df) > 20:
                nifty_closes = nifty_df["close"].reset_index(drop=True)

        snapshot = TechnicalAnalysis.compute_snapshot(
            symbol=symbol,
            daily_closes=df["close"].reset_index(drop=True),
            daily_highs=df["high"].reset_index(drop=True),
            daily_lows=df["low"].reset_index(drop=True),
            daily_volumes=df["volume"].reset_index(drop=True),
            current_price=current_price,
            current_volume=current_volume,
            prev_close=prev_close,
            daily_opens=df["open"].reset_index(drop=True),
            nifty_closes=nifty_closes,
        )

        return snapshot

    except Exception as e:
        logging.error(f"compute_live_analysis failed for {symbol}: {e}")
        return None
