"""Market-wide scanner that periodically scans all NSE stocks.

Scans 2400+ stocks every 30 minutes (configurable) using 50-day historical
candles. Identifies stocks matching specific filter criteria and saves
results to the database + sends Telegram summaries.
"""

import logging
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np


class MarketRadar:
    """Periodic full-market scanner.

    Scan criteria:
    - Deeply oversold: RSI < 25
    - Deeply overbought: RSI > 80
    - Volume spike: > 5x 20-day average
    - Near 52-week high: within 5%
    - MACD bullish crossover with volume confirmation (> 1.5x)

    Results are saved to DB (radar_results table) and sent as Telegram summary.
    """

    SCAN_INTERVAL = 1800  # 30 minutes between full scans
    BATCH_SIZE = 100
    BATCH_DELAY = 0.3  # seconds between API calls within a batch
    CANDLE_DAYS = 50  # lighter than 200 for speed

    # Filter thresholds
    RSI_OVERSOLD = 25
    RSI_OVERBOUGHT = 80
    VOLUME_SPIKE = 5.0
    NEAR_52W_HIGH_PCT = 5.0
    MACD_VOLUME_CONFIRM = 1.5

    def __init__(self, upstox_client, instrument_resolver, database,
                 notification_manager):
        self.upstox_client = upstox_client
        self.instrument_resolver = instrument_resolver
        self.database = database
        self.notification_manager = notification_manager

        self._running = False
        self._thread = None
        self._last_scan_results: List[Dict] = []
        self._last_scan_time: Optional[datetime] = None
        self._pending_telegram: List[str] = []  # messages for Telegram

    def start(self):
        """Start the market radar in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._thread.start()
        logging.info("MarketRadar started")

    def stop(self):
        """Stop the radar."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=30)
        logging.info("MarketRadar stopped")

    def get_pending_telegram_messages(self) -> List[str]:
        """Get and clear pending Telegram messages (called by start_bot)."""
        msgs = self._pending_telegram.copy()
        self._pending_telegram.clear()
        return msgs

    def trigger_scan(self) -> List[Dict]:
        """Trigger an immediate scan (e.g. from /scan command). Returns results."""
        return self._run_full_scan()

    def get_latest_results(self) -> List[Dict]:
        """Get results from the most recent scan."""
        return self._last_scan_results

    def _scan_loop(self):
        """Main loop: run full market scan periodically."""
        # Wait 60s on startup before first scan
        for _ in range(60):
            if not self._running:
                return
            time.sleep(1)

        while self._running:
            try:
                self._run_full_scan()
            except Exception as e:
                logging.error(f"MarketRadar scan error: {e}")

            # Sleep in 1s increments for responsive shutdown
            for _ in range(self.SCAN_INTERVAL):
                if not self._running:
                    break
                time.sleep(1)

    def _run_full_scan(self) -> List[Dict]:
        """Scan all equity symbols and return flagged results."""
        all_symbols = self.instrument_resolver.get_all_equity_symbols()
        if not all_symbols:
            logging.warning("MarketRadar: no equity symbols available")
            return []

        logging.info(f"MarketRadar: scanning {len(all_symbols)} stocks...")
        results = []

        # Process in batches
        for i in range(0, len(all_symbols), self.BATCH_SIZE):
            if not self._running:
                break

            batch = all_symbols[i:i + self.BATCH_SIZE]
            for symbol in batch:
                if not self._running:
                    break
                try:
                    hits = self._scan_single(symbol)
                    if hits:
                        results.extend(hits)
                    time.sleep(self.BATCH_DELAY)
                except Exception as e:
                    logging.debug(f"Radar scan error for {symbol}: {e}")

            # Log progress
            progress = min(i + self.BATCH_SIZE, len(all_symbols))
            logging.info(f"MarketRadar progress: {progress}/{len(all_symbols)} ({len(results)} hits)")

        # Save and notify
        self._last_scan_results = results
        self._last_scan_time = datetime.now()

        for result in results:
            self.database.save_radar_result(result)

        if results:
            self._send_summary(results)
            logging.info(f"MarketRadar: scan complete, {len(results)} stocks flagged")
        else:
            logging.info("MarketRadar: scan complete, no stocks flagged")

        return results

    def _scan_single(self, symbol: str) -> List[Dict]:
        """Scan a single stock and return any matching filter hits."""
        df = self.upstox_client.get_historical_candles(
            symbol, interval="day", days=self.CANDLE_DAYS
        )
        if df is None or len(df) < 20:
            return []

        closes = df["close"].values.astype(float)
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        volumes = df["volume"].values.astype(float)
        current_price = float(closes[-1])
        current_volume = float(volumes[-1])

        hits = []

        # RSI (simple Wilder RSI)
        rsi = self._quick_rsi(closes)

        # Volume ratio
        avg_vol = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
        vol_ratio = current_volume / avg_vol if avg_vol > 0 else 1.0

        # 52-week high/low (from available data)
        high_52w = float(np.max(highs))
        low_52w = float(np.min(lows))
        pct_from_high = (current_price - high_52w) / high_52w * 100 if high_52w > 0 else -100

        # MACD crossover check
        macd_cross = self._quick_macd_crossover(closes)

        # Apply filters
        if rsi < self.RSI_OVERSOLD:
            hits.append({
                "symbol": symbol,
                "trigger_type": "deeply_oversold",
                "price": round(current_price, 2),
                "rsi": round(rsi, 1),
                "volume_ratio": round(vol_ratio, 1),
                "detail": f"RSI={rsi:.0f}, extremely oversold",
            })

        if rsi > self.RSI_OVERBOUGHT:
            hits.append({
                "symbol": symbol,
                "trigger_type": "deeply_overbought",
                "price": round(current_price, 2),
                "rsi": round(rsi, 1),
                "volume_ratio": round(vol_ratio, 1),
                "detail": f"RSI={rsi:.0f}, extremely overbought",
            })

        if vol_ratio > self.VOLUME_SPIKE:
            hits.append({
                "symbol": symbol,
                "trigger_type": "volume_spike",
                "price": round(current_price, 2),
                "rsi": round(rsi, 1),
                "volume_ratio": round(vol_ratio, 1),
                "detail": f"Volume {vol_ratio:.0f}x average",
            })

        if pct_from_high > -self.NEAR_52W_HIGH_PCT:
            hits.append({
                "symbol": symbol,
                "trigger_type": "near_52w_high",
                "price": round(current_price, 2),
                "rsi": round(rsi, 1),
                "volume_ratio": round(vol_ratio, 1),
                "detail": f"{pct_from_high:+.1f}% from {self.CANDLE_DAYS}d high Rs.{high_52w:.2f}",
            })

        if macd_cross and vol_ratio > self.MACD_VOLUME_CONFIRM:
            hits.append({
                "symbol": symbol,
                "trigger_type": "macd_bullish_crossover",
                "price": round(current_price, 2),
                "rsi": round(rsi, 1),
                "volume_ratio": round(vol_ratio, 1),
                "detail": f"MACD bullish crossover with {vol_ratio:.1f}x volume",
            })

        return hits

    def _quick_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """Fast RSI calculation for scanning."""
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _quick_macd_crossover(self, closes: np.ndarray) -> bool:
        """Quick MACD bullish crossover detection."""
        if len(closes) < 35:
            return False

        series = pd.Series(closes)
        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal

        if len(hist) < 2:
            return False

        return float(hist.iloc[-2]) <= 0 and float(hist.iloc[-1]) > 0

    def _send_summary(self, results: List[Dict]):
        """Send a Telegram summary notification of scan results."""
        if not self.notification_manager:
            return

        notif = self.notification_manager.create_radar_notification(results)
        self.notification_manager.add_notification(notif)

        # Build Telegram message
        by_type = {}
        for r in results:
            tt = r.get("trigger_type", "unknown")
            if tt not in by_type:
                by_type[tt] = []
            by_type[tt].append(r)

        lines = [f"<b>Market Radar: {len(results)} stocks flagged</b>\n"]
        for tt, items in by_type.items():
            lines.append(f"\n<b>{tt.replace('_', ' ').title()} ({len(items)}):</b>")
            for item in items[:5]:
                lines.append(
                    f"  {item['symbol']} Rs.{item['price']:.2f} "
                    f"RSI:{item.get('rsi', 0):.0f} Vol:{item.get('volume_ratio', 0):.1f}x"
                )
            if len(items) > 5:
                lines.append(f"  ... +{len(items) - 5} more")

        self._pending_telegram.append("\n".join(lines))
