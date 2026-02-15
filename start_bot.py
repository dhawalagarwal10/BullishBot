"""BullishBot Standalone Launcher.

Runs all background services in a single async event loop:
  1. Telegram Bot (command handlers + polling)
  2. Signal Monitor (polls DB for new signals -> pushes to Telegram)
  3. Portfolio Guard (monitors holdings -> sends Telegram alerts)
  4. Market Radar (scans all NSE stocks -> sends Telegram summaries)
  5. News Watcher (monitors news -> sends urgent alerts)

The MCP server (mcp_server.py) continues to run separately via Claude Desktop.

Usage:
    python start_bot.py
"""

import asyncio
import json
import logging
import os
import signal
import sqlite3
import sys

# Ensure src/ is on the path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
sys.path.insert(0, _SRC_DIR)

os.chdir(_PROJECT_ROOT)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/bot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("start_bot")


def load_config():
    """Load telegram and watchlist configs."""
    with open("config/telegram_config.json", "r") as f:
        telegram_config = json.load(f)

    with open("config/watchlist.json", "r") as f:
        watchlist_config = json.load(f)

    return telegram_config, watchlist_config


def get_held_symbols(db_path: str) -> list:
    """Read currently held symbols from the paper_holdings table."""
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        cursor = conn.cursor()
        cursor.execute("SELECT symbol FROM paper_holdings WHERE quantity > 0")
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        return symbols
    except Exception:
        return []


async def main():
    telegram_config, watchlist_config = load_config()

    # ---- Shared components ----
    from broker_client import UpstoxClient
    from instrument_resolver import InstrumentResolver
    from database import TradingDatabase
    from notification_manager import NotificationManager

    upstox_client = UpstoxClient()
    instrument_resolver = InstrumentResolver()
    database = TradingDatabase()
    notification_manager = NotificationManager()
    db_path = database.db_path

    watchlist_symbols = watchlist_config.get("stocks", [])
    held_symbols = get_held_symbols(db_path)

    # ---- News Watcher (threaded) ----
    from news_watcher import NewsWatcher

    news_watcher = NewsWatcher(notification_manager=notification_manager)
    news_watcher.start(holdings=held_symbols, watchlist=watchlist_symbols)
    logger.info("News Watcher started")

    # ---- Portfolio Guard (threaded) ----
    from portfolio_guard import PortfolioGuard

    portfolio_guard = PortfolioGuard(
        upstox_client=upstox_client,
        notification_manager=notification_manager,
        instrument_resolver=instrument_resolver,
    )
    portfolio_guard.start()
    logger.info("Portfolio Guard started")

    # ---- Market Radar (threaded) ----
    from market_radar import MarketRadar

    market_radar = MarketRadar(
        upstox_client=upstox_client,
        instrument_resolver=instrument_resolver,
        database=database,
        notification_manager=notification_manager,
    )
    market_radar.start()
    logger.info("Market Radar started")

    # ---- Opportunity Scanner (for /scan, /watchlist, /alerts) ----
    from opportunity_scanner import SmartOpportunityScanner
    from realtime_streamer import DataBuffer
    import queue

    data_buffer = DataBuffer()
    tick_queue = queue.Queue()

    scanner = SmartOpportunityScanner(
        data_buffer=data_buffer,
        database=database,
        notification_manager=notification_manager,
        news_analyzer=news_watcher,
    )
    scanner.tick_queue = tick_queue

    # ---- Telegram Bot ----
    from telegram_bot.bot import TradingTelegramBot

    bot = TradingTelegramBot(
        config=telegram_config,
        db_path=db_path,
        upstox_client=upstox_client,
        instrument_resolver=instrument_resolver,
        news_watcher=news_watcher,
        market_radar=market_radar,
        scanner=scanner,
    )
    await bot.start()
    logger.info("Telegram Bot started")

    # ---- Signal Monitor (async, polls DB -> pushes to Telegram) ----
    from telegram_bot.signal_monitor import SignalMonitor

    signal_monitor = SignalMonitor(
        bot=bot,
        db_path=db_path,
        config=telegram_config,
    )
    await signal_monitor.start()
    logger.info("Signal Monitor started")

    logger.info("=" * 50)
    logger.info("All services started. Press Ctrl+C to stop.")
    logger.info("=" * 50)

    # ---- Background task to forward pending messages to Telegram ----
    async def telegram_forwarder():
        """Poll services for pending messages and send them to Telegram."""
        while True:
            try:
                # Collect pending messages from all services
                messages = []
                messages.extend(news_watcher.get_pending_telegram_messages())
                messages.extend(portfolio_guard.get_pending_telegram_messages())
                messages.extend(market_radar.get_pending_telegram_messages())

                # Send each message
                for msg in messages:
                    await bot.broadcast(msg)

            except Exception as e:
                logger.error(f"Telegram forwarder error: {e}")

            await asyncio.sleep(2)  # Check every 2 seconds

    forwarder_task = asyncio.create_task(telegram_forwarder())

    # ---- Keep alive until shutdown ----
    stop_event = asyncio.Event()

    def _signal_handler():
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    # Windows doesn't support add_signal_handler for SIGINT; use fallback
    try:
        loop.add_signal_handler(signal.SIGINT, _signal_handler)
        loop.add_signal_handler(signal.SIGTERM, _signal_handler)
    except NotImplementedError:
        # Windows: Ctrl+C raises KeyboardInterrupt which we catch below
        pass

    try:
        await stop_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Interrupted, shutting down...")

    # Cancel the forwarder task
    forwarder_task.cancel()
    try:
        await forwarder_task
    except asyncio.CancelledError:
        pass

    # ---- Graceful shutdown ----
    logger.info("Stopping all services...")

    await signal_monitor.stop()
    logger.info("Signal Monitor stopped")

    await bot.stop()
    logger.info("Telegram Bot stopped")

    market_radar.stop()
    logger.info("Market Radar stopped")

    portfolio_guard.stop()
    logger.info("Portfolio Guard stopped")

    news_watcher.stop()
    logger.info("News Watcher stopped")

    logger.info("All services stopped. Goodbye.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
