import logging
import threading
import queue
import time
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Callable

import pandas as pd
import numpy as np
import requests

try:
    import upstox_client as upstox_sdk
    HAS_UPSTOX_SDK = True
except ImportError:
    HAS_UPSTOX_SDK = False


class DataBuffer:
    """Thread-safe in-memory buffer for OHLCV data per symbol.

    Holds:
    - daily_data: 200-day daily OHLCV DataFrames (historical + today updated live)
    - intraday_data: Today's 1-minute candles aggregated from ticks
    """

    def __init__(self):
        self.daily_data: Dict[str, pd.DataFrame] = {}
        self.intraday_data: Dict[str, pd.DataFrame] = {}
        self._lock = threading.Lock()

    def initialize_historical(self, symbol: str, daily_df: pd.DataFrame):
        """Load 200-day historical daily data for a symbol."""
        with self._lock:
            self.daily_data[symbol] = daily_df.copy()

    def update_from_tick(self, symbol: str, tick: dict):
        """Update both daily and intraday data from a real-time tick.

        Updates the last row of daily_data to reflect the current day's
        live OHLCV, and aggregates ticks into 1-minute intraday candles.
        """
        with self._lock:
            ltp = tick.get("ltp", 0)
            volume = tick.get("volume", 0)
            tick_time = tick.get("timestamp", datetime.now())

            if ltp <= 0:
                return

            # Update daily data: update today's candle
            if symbol in self.daily_data and len(self.daily_data[symbol]) > 0:
                df = self.daily_data[symbol]
                today = datetime.now().date()
                last_date = pd.Timestamp(df["timestamp"].iloc[-1]).date()

                if last_date == today:
                    # Update existing today row
                    idx = df.index[-1]
                    df.at[idx, "high"] = max(df.at[idx, "high"], ltp)
                    df.at[idx, "low"] = min(df.at[idx, "low"], ltp)
                    df.at[idx, "close"] = ltp
                    df.at[idx, "volume"] = volume
                else:
                    # Append a new row for today
                    new_row = pd.DataFrame([{
                        "timestamp": pd.Timestamp(today),
                        "open": ltp,
                        "high": ltp,
                        "low": ltp,
                        "close": ltp,
                        "volume": volume,
                        "oi": 0
                    }])
                    self.daily_data[symbol] = pd.concat(
                        [df, new_row], ignore_index=True
                    ).tail(250)  # Keep buffer slightly larger than 200

            # Update intraday 1-minute candles
            minute_key = tick_time.replace(second=0, microsecond=0) if isinstance(tick_time, datetime) else datetime.now().replace(second=0, microsecond=0)

            if symbol not in self.intraday_data or len(self.intraday_data[symbol]) == 0:
                self.intraday_data[symbol] = pd.DataFrame([{
                    "timestamp": minute_key,
                    "open": ltp,
                    "high": ltp,
                    "low": ltp,
                    "close": ltp,
                    "volume": volume
                }])
            else:
                idf = self.intraday_data[symbol]
                if idf["timestamp"].iloc[-1] == minute_key:
                    idx = idf.index[-1]
                    idf.at[idx, "high"] = max(idf.at[idx, "high"], ltp)
                    idf.at[idx, "low"] = min(idf.at[idx, "low"], ltp)
                    idf.at[idx, "close"] = ltp
                    idf.at[idx, "volume"] = volume
                else:
                    new_candle = pd.DataFrame([{
                        "timestamp": minute_key,
                        "open": ltp,
                        "high": ltp,
                        "low": ltp,
                        "close": ltp,
                        "volume": volume
                    }])
                    self.intraday_data[symbol] = pd.concat(
                        [idf, new_candle], ignore_index=True
                    ).tail(400)  # ~6.5 hours of 1-min candles

    def get_daily_series(self, symbol: str, field: str = "close", periods: int = 200) -> Optional[pd.Series]:
        """Get a time series of daily data for technical analysis."""
        with self._lock:
            if symbol in self.daily_data and len(self.daily_data[symbol]) > 0:
                return self.daily_data[symbol][field].tail(periods).reset_index(drop=True)
            return None

    def get_daily_dataframe(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get the full daily DataFrame for a symbol."""
        with self._lock:
            if symbol in self.daily_data:
                return self.daily_data[symbol].copy()
            return None

    def get_intraday_dataframe(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get the full intraday 1-minute DataFrame for a symbol."""
        with self._lock:
            if symbol in self.intraday_data and len(self.intraday_data[symbol]) > 0:
                return self.intraday_data[symbol].copy()
            return None

    def has_data(self, symbol: str) -> bool:
        """Check if historical data is loaded for a symbol."""
        with self._lock:
            return symbol in self.daily_data and len(self.daily_data[symbol]) > 0

    def get_symbols_with_data(self) -> List[str]:
        """Get list of symbols that have historical data loaded."""
        with self._lock:
            return [s for s in self.daily_data if len(self.daily_data[s]) > 0]


class RealtimeStreamer:
    """Manages the Upstox WebSocket connection for real-time market data.

    On startup:
    1. Fetches 200-day historical daily candles for all watchlist stocks
    2. Connects to Upstox WebSocket for real-time tick streaming
    3. On each tick, updates the DataBuffer and pushes to a queue for analysis
    """

    def __init__(
        self,
        access_token: str,
        instrument_resolver,
        data_buffer: DataBuffer,
        tick_queue: queue.Queue,
        watchlist: List[str],
    ):
        self.access_token = access_token
        self.instrument_resolver = instrument_resolver
        self.data_buffer = data_buffer
        self.tick_queue = tick_queue
        self.watchlist = watchlist

        self.market_timezone = timezone(timedelta(hours=5, minutes=30))
        self._running = False
        self._streamer = None
        self._ws_thread = None
        self._reconnect_delay = 5
        self._max_reconnect_delay = 300

    def start(self):
        """Fetch historical data then connect WebSocket."""
        self._running = True

        # Fetch historical data (blocking, done once at startup)
        self._fetch_all_historical()

        # Connect WebSocket in a background thread
        self._ws_thread = threading.Thread(target=self._connect_websocket, daemon=True)
        self._ws_thread.start()

    def stop(self):
        """Disconnect the WebSocket and stop streaming."""
        self._running = False
        if self._streamer:
            try:
                self._streamer.disconnect()
            except Exception:
                pass
        logging.info("Realtime streamer stopped")

    def _fetch_all_historical(self):
        """Fetch 200-day daily candle data for each watchlist stock + indices."""
        symbols_to_fetch = list(self.watchlist) + ["NIFTY50", "NIFTYBANK"]

        for symbol in symbols_to_fetch:
            try:
                instrument_key = self.instrument_resolver.get_instrument_key(symbol)
                if not instrument_key:
                    logging.warning(f"No instrument key for {symbol}, skipping historical fetch")
                    continue

                self._fetch_historical_for_symbol(symbol, instrument_key)
                time.sleep(0.3)  # Rate limiting

            except Exception as e:
                logging.error(f"Failed to fetch historical data for {symbol}: {e}")

        loaded = self.data_buffer.get_symbols_with_data()
        logging.info(f"Historical data loaded for {len(loaded)} symbols: {loaded}")

    def _fetch_historical_for_symbol(self, symbol: str, instrument_key: str):
        """Fetch historical daily candles for a single symbol via Upstox REST API."""
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=300)).strftime("%Y-%m-%d")

        # URL-encode the instrument key (pipes need encoding)
        encoded_key = requests.utils.quote(instrument_key, safe='')

        url = f"https://api.upstox.com/v2/historical-candle/{encoded_key}/day/{to_date}/{from_date}"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json"
        }

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        data = response.json()
        if data.get("status") == "success":
            candles = data.get("data", {}).get("candles", [])
            if not candles:
                logging.warning(f"No historical candles returned for {symbol}")
                return

            # Each candle: [timestamp, open, high, low, close, volume, oi]
            df = pd.DataFrame(
                candles,
                columns=["timestamp", "open", "high", "low", "close", "volume", "oi"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").tail(200).reset_index(drop=True)

            self.data_buffer.initialize_historical(symbol, df)
            logging.info(f"Loaded {len(df)} days of historical data for {symbol}")
        else:
            logging.warning(f"Historical API returned non-success for {symbol}: {data.get('message', '')}")

    def _connect_websocket(self):
        """Connect to Upstox WebSocket using the SDK's MarketDataStreamer."""
        if not self._running:
            return

        if not HAS_UPSTOX_SDK:
            logging.error("upstox_client SDK not available. WebSocket streaming disabled.")
            logging.info("Falling back to periodic REST polling.")
            self._fallback_polling_loop()
            return

        try:
            configuration = upstox_sdk.Configuration()
            configuration.access_token = self.access_token
            api_client = upstox_sdk.ApiClient(configuration)

            # Build instrument key list
            instrument_keys = self.instrument_resolver.get_instrument_keys_for_watchlist(self.watchlist)
            # Add index instruments
            nifty_key = self.instrument_resolver.get_instrument_key("NIFTY50")
            bank_key = self.instrument_resolver.get_instrument_key("NIFTYBANK")
            if nifty_key:
                instrument_keys.append(nifty_key)
            if bank_key:
                instrument_keys.append(bank_key)

            # Try MarketDataStreamerV3 first, fall back to V2
            streamer_class = getattr(upstox_sdk, "MarketDataStreamerV3", None)
            if streamer_class is None:
                streamer_class = getattr(upstox_sdk, "MarketDataStreamer", None)

            if streamer_class is None:
                logging.error("No MarketDataStreamer class found in SDK. Using REST fallback.")
                self._fallback_polling_loop()
                return

            self._streamer = streamer_class(api_client, instrument_keys, "full")

            self._streamer.on("open", self._on_open)
            self._streamer.on("message", self._on_message)
            self._streamer.on("error", self._on_error)
            self._streamer.on("close", self._on_close)

            logging.info(f"Connecting WebSocket for {len(instrument_keys)} instruments...")
            self._streamer.connect()

        except Exception as e:
            logging.error(f"WebSocket connection failed: {e}")
            self._schedule_reconnect()

    def _on_open(self):
        """WebSocket connection established."""
        logging.info("WebSocket connected to Upstox market data feed")
        self._reconnect_delay = 5

    def _on_message(self, message):
        """Process incoming tick data from WebSocket."""
        try:
            feeds = {}
            if isinstance(message, dict):
                feeds = message.get("feeds", message)
            else:
                return

            for instrument_key, feed_data in feeds.items():
                symbol = self.instrument_resolver.get_symbol(instrument_key)
                if not symbol:
                    continue

                # Extract tick data from the feed structure
                tick = self._parse_feed(symbol, instrument_key, feed_data)
                if tick and tick.get("ltp", 0) > 0:
                    self.data_buffer.update_from_tick(symbol, tick)
                    self.tick_queue.put(tick)

        except Exception as e:
            logging.error(f"Error processing tick message: {e}")

    def _parse_feed(self, symbol: str, instrument_key: str, feed_data: dict) -> Optional[dict]:
        """Parse the SDK feed data into a normalized tick dict."""
        try:
            # The feed structure varies between V2 and V3 SDK
            # Try V3 format first
            full_feed = feed_data.get("ff", feed_data.get("fullFeed", {}))
            market_data = full_feed.get("marketFF", full_feed.get("indexFF", {}))

            if not market_data:
                # Try flattened format
                market_data = feed_data

            ltpc = market_data.get("ltpc", {})
            ltp = ltpc.get("ltp", market_data.get("ltp", 0))
            cp = ltpc.get("cp", market_data.get("cp", 0))  # previous close

            tick = {
                "symbol": symbol,
                "instrument_key": instrument_key,
                "ltp": float(ltp) if ltp else 0,
                "cp": float(cp) if cp else 0,
                "volume": int(market_data.get("vtt", market_data.get("v", 0))),
                "timestamp": datetime.now(self.market_timezone),
            }

            # Extract OHLC from daily candle if available
            ohlc_data = market_data.get("marketOHLC", {}).get("ohlc", [])
            if isinstance(ohlc_data, list):
                for ohlc in ohlc_data:
                    if ohlc.get("interval") == "1d":
                        tick["open"] = float(ohlc.get("open", 0))
                        tick["high"] = float(ohlc.get("high", 0))
                        tick["low"] = float(ohlc.get("low", 0))
                        tick["close"] = float(ohlc.get("close", 0))
                        tick["day_volume"] = int(ohlc.get("volume", 0))
                        break

            return tick

        except Exception as e:
            logging.debug(f"Feed parse error for {symbol}: {e}")
            return None

    def _on_error(self, error):
        """WebSocket error handler."""
        logging.error(f"WebSocket error: {error}")

    def _on_close(self):
        """WebSocket connection closed."""
        logging.warning("WebSocket connection closed")
        if self._running:
            self._schedule_reconnect()

    def _schedule_reconnect(self):
        """Reconnect with exponential backoff."""
        if not self._running:
            return
        logging.info(f"Reconnecting WebSocket in {self._reconnect_delay}s...")
        time.sleep(self._reconnect_delay)
        self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)
        self._connect_websocket()

    def _fallback_polling_loop(self):
        """Fallback: poll quotes via REST API if WebSocket is unavailable."""
        logging.info("Starting REST polling fallback (30s interval)")

        while self._running:
            try:
                if not self.is_market_open():
                    time.sleep(60)
                    continue

                for symbol in self.watchlist:
                    if not self._running:
                        break

                    try:
                        instrument_key = self.instrument_resolver.get_instrument_key(symbol)
                        if not instrument_key:
                            continue

                        url = f"https://api.upstox.com/v2/market-quote/quotes"
                        headers = {
                            "Authorization": f"Bearer {self.access_token}",
                            "Accept": "application/json"
                        }
                        # Use the simple NSE:SYMBOL format for REST quotes
                        params = {"instrument_key": f"NSE_EQ:{symbol}"}

                        response = requests.get(url, headers=headers, params=params, timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            if data.get("status") == "success":
                                quote_data = data.get("data", {})
                                for key, qd in quote_data.items():
                                    tick = {
                                        "symbol": symbol,
                                        "instrument_key": instrument_key,
                                        "ltp": float(qd.get("last_price", 0)),
                                        "cp": float(qd.get("ohlc", {}).get("close", 0)),
                                        "volume": int(qd.get("volume", 0)),
                                        "timestamp": datetime.now(self.market_timezone),
                                    }
                                    if tick["ltp"] > 0:
                                        self.data_buffer.update_from_tick(symbol, tick)
                                        self.tick_queue.put(tick)

                        time.sleep(0.2)  # Rate limiting

                    except Exception as e:
                        logging.debug(f"Polling error for {symbol}: {e}")

                time.sleep(30)

            except Exception as e:
                logging.error(f"Fallback polling loop error: {e}")
                time.sleep(60)

    def is_market_open(self) -> bool:
        """Check if Indian equity market is currently open."""
        now = datetime.now(self.market_timezone)
        if now.weekday() >= 5:
            return False
        start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return start <= now <= end
