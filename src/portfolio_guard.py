"""Continuous portfolio monitoring service.

Monitors held positions every 30 seconds and sends Telegram alerts
for significant events: large losses, trend deterioration, unusual volume,
RSI extremes (profit booking opportunities).
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict


class PortfolioGuard:
    """Watches held positions and sends alerts when action may be needed.

    Checks per holding (every 30s):
    - Loss from avg price > -3%  -> URGENT (cooldown 30 min)
    - Intraday P&L drop > -2%   -> WARNING (cooldown 15 min)
    - Trend deterioration        -> WARNING (cooldown 1 hour)
    - Unusual volume > 3x avg   -> INFO/URGENT (cooldown 30 min)
    - RSI overbought > 75       -> INFO (profit booking, cooldown 1 hour)
    """

    CHECK_INTERVAL = 30  # seconds between checks

    # Alert thresholds
    LOSS_FROM_AVG_PCT = -3.0
    INTRADAY_DROP_PCT = -2.0
    VOLUME_SPIKE_THRESHOLD = 3.0
    RSI_OVERBOUGHT = 75

    # Cooldowns in seconds per (symbol, check_type) pair
    COOLDOWNS = {
        "loss_from_avg": 1800,       # 30 min
        "intraday_drop": 900,        # 15 min
        "trend_deterioration": 3600, # 1 hour
        "volume_spike": 1800,        # 30 min
        "rsi_overbought": 3600,      # 1 hour
    }

    def __init__(self, upstox_client, notification_manager, instrument_resolver):
        self.upstox_client = upstox_client
        self.notification_manager = notification_manager
        self.instrument_resolver = instrument_resolver

        self._running = False
        self._thread = None
        self._last_alert_time: Dict[str, datetime] = defaultdict(lambda: datetime.min)
        self._prev_trend: Dict[str, str] = {}  # symbol -> last known trend
        self._pending_telegram: List[str] = []  # messages for Telegram

    def start(self):
        """Start portfolio guard in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._guard_loop, daemon=True)
        self._thread.start()
        logging.info("PortfolioGuard started")

    def stop(self):
        """Stop the guard."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logging.info("PortfolioGuard stopped")

    def get_pending_telegram_messages(self) -> List[str]:
        """Get and clear pending Telegram messages (called by start_bot)."""
        msgs = self._pending_telegram.copy()
        self._pending_telegram.clear()
        return msgs

    def _guard_loop(self):
        """Main loop: fetch portfolio, check each holding."""
        while self._running:
            try:
                portfolio = self.upstox_client.get_portfolio()
                if not portfolio or not portfolio.get("holdings"):
                    time.sleep(self.CHECK_INTERVAL)
                    continue

                holdings = portfolio["holdings"]
                for holding in holdings:
                    if not self._running:
                        break
                    self._check_holding(holding)

            except Exception as e:
                logging.error(f"PortfolioGuard error: {e}")

            # Sleep in 1s increments for responsive shutdown
            for _ in range(self.CHECK_INTERVAL):
                if not self._running:
                    break
                time.sleep(1)

    def _check_holding(self, holding: Dict):
        """Run all checks on a single holding."""
        symbol = holding.get("symbol", "").upper()
        if not symbol:
            return

        avg_price = holding.get("average_price", 0)
        current_price = holding.get("current_price", 0)
        pnl_percent = holding.get("pnl_percent", 0)

        if current_price <= 0 or avg_price <= 0:
            return

        # 1. Loss from average price
        if pnl_percent <= self.LOSS_FROM_AVG_PCT:
            self._send_alert(
                symbol=symbol,
                check_type="loss_from_avg",
                severity="URGENT",
                message=(
                    f"[URGENT] {symbol}: Down {pnl_percent:.1f}% from avg price "
                    f"(Avg: Rs.{avg_price:.2f}, Now: Rs.{current_price:.2f}). "
                    f"Consider reviewing position."
                ),
            )

        # 2. Try on-demand analysis for trend/volume/RSI checks
        try:
            from technical_analysis import compute_live_analysis
            snapshot = compute_live_analysis(symbol, self.upstox_client, self.instrument_resolver)
            if snapshot:
                self._check_with_snapshot(symbol, snapshot, current_price)
        except Exception as e:
            logging.debug(f"PortfolioGuard live analysis failed for {symbol}: {e}")

    def _check_with_snapshot(self, symbol: str, snapshot, current_price: float):
        """Run checks that need technical analysis data."""

        # 3. Trend deterioration
        prev_trend = self._prev_trend.get(symbol)
        current_trend = snapshot.trend
        self._prev_trend[symbol] = current_trend

        if prev_trend and self._trend_worsened(prev_trend, current_trend):
            self._send_alert(
                symbol=symbol,
                check_type="trend_deterioration",
                severity="WARNING",
                message=(
                    f"[WARNING] {symbol}: Trend deteriorated from {prev_trend} to "
                    f"{current_trend}. Price: Rs.{current_price:.2f}"
                ),
            )

        # 4. Unusual volume
        if snapshot.volume_ratio and snapshot.volume_ratio > self.VOLUME_SPIKE_THRESHOLD:
            severity = "URGENT" if snapshot.change_percent < -1 else "INFO"
            direction = "with selling" if snapshot.change_percent < 0 else "with buying"
            self._send_alert(
                symbol=symbol,
                check_type="volume_spike",
                severity=severity,
                message=(
                    f"[{severity}] {symbol}: Volume spike {snapshot.volume_ratio:.1f}x average "
                    f"{direction} pressure. Price: Rs.{current_price:.2f} "
                    f"({snapshot.change_percent:+.1f}%)"
                ),
            )

        # 5. RSI overbought (profit booking opportunity)
        if snapshot.rsi_14 > self.RSI_OVERBOUGHT:
            self._send_alert(
                symbol=symbol,
                check_type="rsi_overbought",
                severity="INFO",
                message=(
                    f"[INFO] {symbol}: RSI at {snapshot.rsi_14:.0f} (overbought). "
                    f"Consider booking partial profits. Price: Rs.{current_price:.2f}"
                ),
            )

    def _trend_worsened(self, prev: str, current: str) -> bool:
        """Check if trend has worsened (moved toward bearish)."""
        levels = ["strong_uptrend", "uptrend", "sideways", "downtrend", "strong_downtrend"]
        try:
            prev_idx = levels.index(prev)
            curr_idx = levels.index(current)
            return curr_idx > prev_idx  # higher index = more bearish
        except ValueError:
            return False

    def _send_alert(self, symbol: str, check_type: str, severity: str, message: str):
        """Send alert if cooldown has expired for this (symbol, check_type)."""
        key = f"{symbol}:{check_type}"
        cooldown = self.COOLDOWNS.get(check_type, 1800)
        now = datetime.now()

        if (now - self._last_alert_time[key]).total_seconds() < cooldown:
            return  # Still in cooldown

        self._last_alert_time[key] = now

        if self.notification_manager:
            notif = self.notification_manager.create_portfolio_guard_notification({
                "symbol": symbol,
                "check_type": check_type,
                "severity": severity,
                "message": message,
            })
            self.notification_manager.add_notification(notif)

        # Queue for Telegram
        self._pending_telegram.append(f"<b>{message}</b>")
        logging.info(f"PortfolioGuard alert: {message}")
