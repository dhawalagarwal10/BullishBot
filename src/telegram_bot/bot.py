import os
import sys
import json
import sqlite3
import logging
from typing import Optional

from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from telegram_bot.formatters import TelegramFormatter


class TradingTelegramBot:
    def __init__(self, config: dict, db_path: str, upstox_client=None,
                 instrument_resolver=None, news_watcher=None, market_radar=None,
                 scanner=None):
        self.config = config
        self.bot_token = config["bot_token"]
        self.allowed_chat_ids = set(config.get("allowed_chat_ids", []))
        self.db_path = db_path
        self.upstox_client = upstox_client
        self.instrument_resolver = instrument_resolver
        self.news_watcher = news_watcher
        self.market_radar = market_radar
        self.scanner = scanner
        self.app: Optional[Application] = None
        self.formatter = TelegramFormatter()

    async def start(self):
        self.app = Application.builder().token(self.bot_token).build()

        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("help", self.cmd_start))
        self.app.add_handler(CommandHandler("portfolio", self.cmd_portfolio))
        self.app.add_handler(CommandHandler("quote", self.cmd_quote))
        self.app.add_handler(CommandHandler("signals", self.cmd_signals))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        # New commands
        self.app.add_handler(CommandHandler("analyze", self.cmd_analyze))
        self.app.add_handler(CommandHandler("scan", self.cmd_scan))
        self.app.add_handler(CommandHandler("watchlist", self.cmd_watchlist))
        self.app.add_handler(CommandHandler("news", self.cmd_news))
        self.app.add_handler(CommandHandler("alerts", self.cmd_alerts))

        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)
        logging.info("Telegram bot started polling")

    async def stop(self):
        if self.app:
            try:
                await self.app.updater.stop()
                await self.app.stop()
                await self.app.shutdown()
            except Exception as e:
                logging.warning(f"Telegram bot shutdown: {e}")

    def _is_authorized(self, chat_id: int) -> bool:
        if not self.allowed_chat_ids:
            return True
        return chat_id in self.allowed_chat_ids

    async def send_message(self, chat_id: int, text: str):
        """Send a message to a specific chat. Used by SignalMonitor."""
        if self.app and self.app.bot:
            try:
                # Telegram has a 4096 char limit per message
                if len(text) > 4000:
                    for i in range(0, len(text), 4000):
                        chunk = text[i:i + 4000]
                        await self.app.bot.send_message(
                            chat_id=chat_id, text=chunk, parse_mode="HTML"
                        )
                else:
                    await self.app.bot.send_message(
                        chat_id=chat_id, text=text, parse_mode="HTML"
                    )
            except Exception as e:
                logging.error(f"Failed to send Telegram message: {e}")

    async def broadcast(self, text: str):
        """Send message to all allowed chat IDs."""
        if not self.allowed_chat_ids:
            return
        for chat_id in self.allowed_chat_ids:
            await self.send_message(chat_id, text)

    # ---- Existing Commands ----

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_chat.id):
            await update.message.reply_text("Unauthorized.")
            return

        await update.message.reply_text(
            "<b>BullishBot Trading Assistant</b>\n\n"
            "Commands:\n"
            "/portfolio - View portfolio holdings\n"
            "/quote SYMBOL - Get live stock quote\n"
            "/analyze SYMBOL - Full technical analysis\n"
            "/signals - Recent trading signals\n"
            "/scan - Trigger market radar scan\n"
            "/watchlist - View/manage watchlist\n"
            "/news SYMBOL - Latest news for a stock\n"
            "/alerts - Show alert status\n"
            "/status - System status\n"
            "/help - Show this message\n\n"
            f"Your chat ID: <code>{update.effective_chat.id}</code>",
            parse_mode="HTML",
        )

    async def cmd_portfolio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_chat.id):
            return

        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("SELECT cash_balance, starting_balance FROM paper_account LIMIT 1")
            account = cursor.fetchone()

            cursor.execute("SELECT symbol, quantity, average_price FROM paper_holdings WHERE quantity > 0")
            holdings = cursor.fetchall()
            conn.close()

            if not account:
                await update.message.reply_text("No portfolio data available.")
                return

            text = self.formatter.format_portfolio(account, holdings, self.upstox_client)
            await update.message.reply_text(text, parse_mode="HTML")

        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def cmd_quote(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_chat.id):
            return

        if not context.args:
            await update.message.reply_text("Usage: /quote SYMBOL\nExample: /quote RELIANCE")
            return

        symbol = context.args[0].upper()

        if not self.upstox_client:
            await update.message.reply_text("Upstox client not available.")
            return

        quote = self.upstox_client.get_quote(symbol)
        if quote:
            text = self.formatter.format_quote(quote)
            await update.message.reply_text(text, parse_mode="HTML")
        else:
            await update.message.reply_text(f"Could not fetch quote for {symbol}")

    async def cmd_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_chat.id):
            return

        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM signals ORDER BY timestamp DESC LIMIT 5")
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                await update.message.reply_text("No recent signals. Scanner is monitoring the market.")
                return

            text = self.formatter.format_signals(rows)
            await update.message.reply_text(text, parse_mode="HTML")

        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_chat.id):
            return

        api_ok = False
        if self.upstox_client:
            test = self.upstox_client.get_quote("RELIANCE")
            api_ok = test is not None

        services = ["Telegram Bot: Running"]
        if self.scanner:
            status = self.scanner.get_scanner_status()
            services.append(f"Scanner: {'Running' if status.get('running') else 'Stopped'}")
            services.append(f"Market: {'Open' if status.get('market_open') else 'Closed'}")
            services.append(f"Watchlist: {status.get('watchlist_size', 0)} stocks")
        if self.market_radar:
            services.append("Market Radar: Running")
        if self.news_watcher:
            services.append("News Watcher: Running")

        text = (
            "<b>System Status</b>\n\n"
            f"API: {'Connected' if api_ok else 'Disconnected'}\n"
            + "\n".join(services)
        )
        await update.message.reply_text(text, parse_mode="HTML")

    # ---- New Commands ----

    async def cmd_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Full on-demand technical analysis for any stock."""
        if not self._is_authorized(update.effective_chat.id):
            return

        if not context.args:
            await update.message.reply_text("Usage: /analyze SYMBOL\nExample: /analyze RELIANCE")
            return

        symbol = context.args[0].upper()

        if not self.upstox_client or not self.instrument_resolver:
            await update.message.reply_text("Analysis service not available.")
            return

        await update.message.reply_text(f"Analyzing {symbol}... (fetching 200-day data)")

        try:
            from technical_analysis import compute_live_analysis
            snapshot = compute_live_analysis(symbol, self.upstox_client, self.instrument_resolver)

            if snapshot:
                text = self.formatter.format_full_analysis(snapshot)
                await update.message.reply_text(text, parse_mode="HTML")
            else:
                await update.message.reply_text(
                    f"Could not analyze {symbol}. Check if the symbol is valid."
                )
        except Exception as e:
            await update.message.reply_text(f"Analysis error: {e}")

    async def cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Trigger a market radar scan."""
        if not self._is_authorized(update.effective_chat.id):
            return

        if not self.market_radar:
            await update.message.reply_text("Market radar not available.")
            return

        await update.message.reply_text("Triggering market radar scan... This may take several minutes.")

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self.market_radar.trigger_scan)

            if results:
                lines = [f"<b>Market Radar: {len(results)} stocks flagged</b>\n"]
                by_type = {}
                for r in results:
                    tt = r.get("trigger_type", "unknown")
                    if tt not in by_type:
                        by_type[tt] = []
                    by_type[tt].append(r)

                for tt, items in by_type.items():
                    lines.append(f"\n<b>{tt.replace('_', ' ').title()} ({len(items)}):</b>")
                    for item in items[:5]:
                        lines.append(
                            f"  {item['symbol']} Rs.{item['price']:.2f} "
                            f"(RSI: {item.get('rsi', 0):.0f}, Vol: {item.get('volume_ratio', 0):.0f}x)"
                        )
                    if len(items) > 5:
                        lines.append(f"  ... +{len(items) - 5} more")

                text = "\n".join(lines)
                await update.message.reply_text(text, parse_mode="HTML")
            else:
                await update.message.reply_text("Scan complete. No stocks flagged.")
        except Exception as e:
            await update.message.reply_text(f"Scan error: {e}")

    async def cmd_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """View or manage the watchlist."""
        if not self._is_authorized(update.effective_chat.id):
            return

        if not self.scanner:
            await update.message.reply_text("Scanner not available.")
            return

        if not context.args:
            text = self.formatter.format_watchlist(self.scanner.watchlist)
            await update.message.reply_text(text, parse_mode="HTML")
            return

        action = context.args[0].lower()

        if action == "add" and len(context.args) >= 2:
            symbol = context.args[1].upper()
            success = self.scanner.add_to_watchlist(symbol)
            if success:
                await update.message.reply_text(f"Added {symbol} to watchlist.")
            else:
                await update.message.reply_text(f"{symbol} is already in the watchlist.")

        elif action == "remove" and len(context.args) >= 2:
            symbol = context.args[1].upper()
            success = self.scanner.remove_from_watchlist(symbol)
            if success:
                await update.message.reply_text(f"Removed {symbol} from watchlist.")
            else:
                await update.message.reply_text(f"{symbol} is not in the watchlist.")

        else:
            await update.message.reply_text(
                "Usage:\n"
                "/watchlist - View watchlist\n"
                "/watchlist add SYMBOL - Add stock\n"
                "/watchlist remove SYMBOL - Remove stock"
            )

    async def cmd_news(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get latest news for a stock."""
        if not self._is_authorized(update.effective_chat.id):
            return

        if not context.args:
            await update.message.reply_text("Usage: /news SYMBOL\nExample: /news RELIANCE")
            return

        symbol = context.args[0].upper()

        if not self.news_watcher:
            await update.message.reply_text("News service not available.")
            return

        await update.message.reply_text(f"Fetching news for {symbol}...")

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            items = await loop.run_in_executor(None, self.news_watcher.get_news, symbol, 5)
            text = self.formatter.format_news(symbol, items)
            await update.message.reply_text(text, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"Error fetching news: {e}")

    async def cmd_alerts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show alert configuration and recent alert counts."""
        if not self._is_authorized(update.effective_chat.id):
            return

        lines = ["<b>Alert Configuration</b>\n"]

        notifications = self.config.get("notifications", {})
        lines.append(f"Signals: {'Enabled' if notifications.get('signals', True) else 'Disabled'}")
        lines.append(f"Min confidence: {notifications.get('min_signal_confidence', 50)}%")
        lines.append(f"Portfolio alerts: {'Enabled' if notifications.get('portfolio_alerts', True) else 'Disabled'}")
        lines.append(f"System alerts: {'Enabled' if notifications.get('system_alerts', True) else 'Disabled'}")

        if self.scanner:
            lines.append(f"\nSignal cooldown: {self.scanner.signal_cooldown}s")
            lines.append(f"Analysis interval: {self.scanner.analysis_interval}s")
            if self.scanner.alert_count:
                lines.append("\nRecent alert counts:")
                for sym, count in self.scanner.alert_count.items():
                    lines.append(f"  {sym}: {count}")

        text = "\n".join(lines)
        await update.message.reply_text(text, parse_mode="HTML")
