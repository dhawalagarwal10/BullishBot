import asyncio
import sqlite3
import logging

from telegram_bot.formatters import TelegramFormatter


class SignalMonitor:
    """Background task that polls the SQLite signals table and pushes new signals to Telegram.

    Self-contained: reads directly from the database instead of depending
    on dashboard services.
    """

    def __init__(self, bot, db_path: str, config: dict):
        self.bot = bot
        self.db_path = db_path
        self.config = config
        self.notification_config = config.get("notifications", {})
        self._last_signal_id = 0
        self._running = False
        self.formatter = TelegramFormatter()

    async def start(self):
        self._last_signal_id = self._get_max_signal_id()
        self._running = True
        asyncio.create_task(self._monitor_loop())
        logging.info("Telegram SignalMonitor started")

    async def stop(self):
        self._running = False

    def _get_max_signal_id(self) -> int:
        """Get the current maximum signal ID from the database."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(id) FROM signals")
            row = cursor.fetchone()
            conn.close()
            return row[0] if row and row[0] else 0
        except Exception as e:
            logging.error(f"SignalMonitor: failed to get max signal id: {e}")
            return 0

    def _get_signals_after(self, after_id: int):
        """Get signals with id > after_id, ordered by id."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM signals WHERE id > ? ORDER BY id ASC",
                (after_id,),
            )
            rows = [dict(r) for r in cursor.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            logging.error(f"SignalMonitor: failed to query signals: {e}")
            return []

    async def _monitor_loop(self):
        interval = self.config.get("polling_interval_seconds", 10)
        min_confidence = self.notification_config.get("min_signal_confidence", 50)
        signals_enabled = self.notification_config.get("signals", True)

        while self._running:
            try:
                if signals_enabled:
                    await self._check_new_signals(min_confidence)
            except Exception as e:
                logging.error(f"SignalMonitor error: {e}")

            await asyncio.sleep(interval)

    async def _check_new_signals(self, min_confidence: float):
        new_signals = self._get_signals_after(self._last_signal_id)

        for signal in new_signals:
            self._last_signal_id = signal["id"]

            if signal.get("confidence", 0) < min_confidence:
                continue

            signal_type = signal.get("signal_type", "")
            if signal_type not in ("BUY", "SELL"):
                continue

            text = self.formatter.format_signal_alert(signal)
            await self.bot.broadcast(text)
