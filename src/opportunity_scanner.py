import json
import logging
import threading
import queue
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from technical_analysis import TechnicalAnalysis, TechnicalSnapshot
from database import TradingDatabase
from notification_manager import NotificationManager


class SignalEvaluator:
    """Multi-signal confluence engine that evaluates technical snapshots
    and generates BUY/SELL/HOLD signals with confidence scores.

    Each detected condition adds weighted points to either buy_score or sell_score.
    A signal is generated only when:
    - The dominant score >= MIN_CONFIDENCE (30)
    - The dominant score exceeds the other by >= MIN_MARGIN (15)
    """

    WEIGHTS = {
        "rsi_oversold_recovery": 15,
        "rsi_overbought_decline": 15,
        "macd_bullish_crossover": 20,
        "macd_bearish_crossover": 20,
        "price_above_sma200": 10,
        "price_below_sma200": 10,
        "golden_cross": 15,
        "death_cross": 15,
        "bollinger_below_lower": 10,
        "bollinger_above_upper": 10,
        "volume_spike": 10,
        "breaking_resistance": 15,
        "breaking_support": 15,
        "strong_uptrend": 10,
        "strong_downtrend": 10,
        # NEW indicators
        "adx_bullish": 15,
        "adx_bearish": 15,
        "stochrsi_oversold_cross": 12,
        "stochrsi_overbought_cross": 12,
        "ichimoku_bullish": 10,
        "ichimoku_bearish": 10,
        "obv_bullish": 8,
        "obv_bearish": 8,
        "candle_bullish": 10,
        "candle_bearish": 10,
        "rs_outperforming": 5,
        "rs_underperforming": 5,
    }

    MIN_CONFIDENCE = 30
    MIN_MARGIN = 15

    @classmethod
    def evaluate(
        cls,
        snapshot: TechnicalSnapshot,
        prev_snapshot: Optional[TechnicalSnapshot] = None,
    ) -> Tuple[str, float, List[str]]:
        """Evaluate a technical snapshot and return (signal_type, confidence, reasons).

        Returns:
            signal_type: "BUY", "SELL", or "HOLD"
            confidence: 0-100
            reasons: List of human-readable reason strings
        """
        buy_score = 0
        sell_score = 0
        reasons_buy = []
        reasons_sell = []

        # --- RSI Signals ---
        if snapshot.rsi_14 < 30:
            buy_score += cls.WEIGHTS["rsi_oversold_recovery"]
            reasons_buy.append(f"RSI oversold at {snapshot.rsi_14:.1f}")
        elif snapshot.rsi_14 < 40 and prev_snapshot and prev_snapshot.rsi_14 < 30:
            buy_score += cls.WEIGHTS["rsi_oversold_recovery"]
            reasons_buy.append(
                f"RSI recovering from oversold ({prev_snapshot.rsi_14:.1f} -> {snapshot.rsi_14:.1f})"
            )

        if snapshot.rsi_14 > 70:
            sell_score += cls.WEIGHTS["rsi_overbought_decline"]
            reasons_sell.append(f"RSI overbought at {snapshot.rsi_14:.1f}")
        elif snapshot.rsi_14 > 60 and prev_snapshot and prev_snapshot.rsi_14 > 70:
            sell_score += cls.WEIGHTS["rsi_overbought_decline"]
            reasons_sell.append(
                f"RSI declining from overbought ({prev_snapshot.rsi_14:.1f} -> {snapshot.rsi_14:.1f})"
            )

        # --- MACD Signals ---
        if snapshot.macd_crossover == "bullish_crossover":
            buy_score += cls.WEIGHTS["macd_bullish_crossover"]
            reasons_buy.append("Positive MACD crossover (bullish momentum)")
        elif snapshot.macd_crossover == "bearish_crossover":
            sell_score += cls.WEIGHTS["macd_bearish_crossover"]
            reasons_sell.append("Negative MACD crossover (bearish momentum)")

        # --- Moving Average Signals ---
        if snapshot.price_vs_sma200 == "above":
            buy_score += cls.WEIGHTS["price_above_sma200"]
            reasons_buy.append("Trading above 200-day SMA (long-term uptrend)")
        else:
            sell_score += cls.WEIGHTS["price_below_sma200"]
            reasons_sell.append("Trading below 200-day SMA (long-term downtrend)")

        if snapshot.golden_cross:
            buy_score += cls.WEIGHTS["golden_cross"]
            reasons_buy.append("Golden Cross detected (50 SMA crossed above 200 SMA)")
        if snapshot.death_cross:
            sell_score += cls.WEIGHTS["death_cross"]
            reasons_sell.append("Death Cross detected (50 SMA crossed below 200 SMA)")

        # --- Bollinger Band Signals ---
        if snapshot.bb_position == "below_lower":
            buy_score += cls.WEIGHTS["bollinger_below_lower"]
            reasons_buy.append("Price below lower Bollinger Band (potential bounce)")
        elif snapshot.bb_position == "above_upper":
            sell_score += cls.WEIGHTS["bollinger_above_upper"]
            reasons_sell.append("Price above upper Bollinger Band (potential pullback)")

        # --- Volume Signals ---
        if snapshot.volume_ratio > 2.0:
            if snapshot.change_percent > 0:
                buy_score += cls.WEIGHTS["volume_spike"]
                reasons_buy.append(
                    f"Unusual volume ({snapshot.volume_ratio:.1f}x avg) with positive price action"
                )
            else:
                sell_score += cls.WEIGHTS["volume_spike"]
                reasons_sell.append(
                    f"Unusual volume ({snapshot.volume_ratio:.1f}x avg) with negative price action"
                )

        # --- Support/Resistance Signals ---
        if snapshot.distance_to_resistance < 1.0 and snapshot.change_percent > 0.5:
            buy_score += cls.WEIGHTS["breaking_resistance"]
            reasons_buy.append(
                f"Breaking through resistance at Rs.{snapshot.resistance_level:.2f}"
            )
        if snapshot.distance_to_support < 1.0 and snapshot.change_percent < -0.5:
            sell_score += cls.WEIGHTS["breaking_support"]
            reasons_sell.append(
                f"Breaking below support at Rs.{snapshot.support_level:.2f}"
            )

        # --- Trend Signals ---
        if snapshot.trend == "strong_uptrend":
            buy_score += cls.WEIGHTS["strong_uptrend"]
            reasons_buy.append("Strong uptrend (all MAs aligned bullish)")
        elif snapshot.trend == "strong_downtrend":
            sell_score += cls.WEIGHTS["strong_downtrend"]
            reasons_sell.append("Strong downtrend (all MAs aligned bearish)")

        # --- ADX Signals ---
        if snapshot.adx is not None and snapshot.adx > 25:
            if snapshot.plus_di and snapshot.minus_di:
                if snapshot.plus_di > snapshot.minus_di:
                    buy_score += cls.WEIGHTS["adx_bullish"]
                    reasons_buy.append(f"ADX {snapshot.adx:.0f} with DI+ > DI- (strong bullish trend)")
                else:
                    sell_score += cls.WEIGHTS["adx_bearish"]
                    reasons_sell.append(f"ADX {snapshot.adx:.0f} with DI- > DI+ (strong bearish trend)")

        # --- Stochastic RSI Signals ---
        if snapshot.stoch_rsi_k is not None and snapshot.stoch_rsi_d is not None:
            if snapshot.stoch_rsi_k < 20 and snapshot.stoch_rsi_k > snapshot.stoch_rsi_d:
                buy_score += cls.WEIGHTS["stochrsi_oversold_cross"]
                reasons_buy.append(f"StochRSI oversold crossover (K={snapshot.stoch_rsi_k:.0f})")
            elif snapshot.stoch_rsi_k > 80 and snapshot.stoch_rsi_k < snapshot.stoch_rsi_d:
                sell_score += cls.WEIGHTS["stochrsi_overbought_cross"]
                reasons_sell.append(f"StochRSI overbought crossover (K={snapshot.stoch_rsi_k:.0f})")

        # --- Ichimoku Signals ---
        if snapshot.ichimoku_signal == "above_cloud":
            buy_score += cls.WEIGHTS["ichimoku_bullish"]
            reasons_buy.append("Price above Ichimoku cloud (bullish)")
        elif snapshot.ichimoku_signal == "below_cloud":
            sell_score += cls.WEIGHTS["ichimoku_bearish"]
            reasons_sell.append("Price below Ichimoku cloud (bearish)")

        # --- OBV Signals ---
        if snapshot.obv_signal == "bullish":
            buy_score += cls.WEIGHTS["obv_bullish"]
            reasons_buy.append("OBV confirms uptrend (volume backing price)")
        elif snapshot.obv_signal == "bearish":
            sell_score += cls.WEIGHTS["obv_bearish"]
            reasons_sell.append("OBV bearish divergence (volume not supporting price)")

        # --- Candlestick Pattern Signals ---
        if snapshot.candlestick_bias == "bullish" and snapshot.candlestick_pattern != "none":
            buy_score += cls.WEIGHTS["candle_bullish"]
            reasons_buy.append(f"Bullish candlestick: {snapshot.candlestick_pattern.replace('_', ' ')}")
        elif snapshot.candlestick_bias == "bearish" and snapshot.candlestick_pattern != "none":
            sell_score += cls.WEIGHTS["candle_bearish"]
            reasons_sell.append(f"Bearish candlestick: {snapshot.candlestick_pattern.replace('_', ' ')}")

        # --- Relative Strength vs NIFTY ---
        if snapshot.rs_vs_nifty is not None:
            if snapshot.rs_vs_nifty > 1.05:
                buy_score += cls.WEIGHTS["rs_outperforming"]
                reasons_buy.append(f"Outperforming NIFTY (RS={snapshot.rs_vs_nifty:.2f})")
            elif snapshot.rs_vs_nifty < 0.95:
                sell_score += cls.WEIGHTS["rs_underperforming"]
                reasons_sell.append(f"Underperforming NIFTY (RS={snapshot.rs_vs_nifty:.2f})")

        # --- Determine final signal ---
        buy_confidence = min(buy_score, 100)
        sell_confidence = min(sell_score, 100)

        if buy_confidence >= cls.MIN_CONFIDENCE and (buy_confidence - sell_confidence) >= cls.MIN_MARGIN:
            return "BUY", buy_confidence, reasons_buy
        elif sell_confidence >= cls.MIN_CONFIDENCE and (sell_confidence - buy_confidence) >= cls.MIN_MARGIN:
            return "SELL", sell_confidence, reasons_sell
        else:
            all_reasons = reasons_buy + reasons_sell
            return "HOLD", max(buy_confidence, sell_confidence), all_reasons if all_reasons else ["No strong signals detected"]


class SmartOpportunityScanner:
    """Event-driven opportunity scanner that consumes real-time ticks,
    runs technical analysis, and generates confluence-based trading signals.

    Replaces the old polling-based OpportunityScanner while preserving
    the same external interface (start_scanner, stop_scanner, add_to_watchlist,
    remove_from_watchlist, get_scanner_status).
    """

    def __init__(
        self,
        data_buffer,
        database: TradingDatabase,
        notification_manager: NotificationManager,
        news_analyzer=None,
        config_path: str = "config/watchlist.json",
    ):
        self.data_buffer = data_buffer
        self.database = database
        self.notification_manager = notification_manager
        self.news_analyzer = news_analyzer
        self.config = self._load_config(config_path)
        self.watchlist = self.config.get("stocks", [])
        self.alert_settings = self.config.get("alert_settings", {})

        self.tick_queue = queue.Queue()
        self.running = False
        self._analysis_thread = None
        self._news_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="news")

        # State tracking
        self._last_snapshots: Dict[str, TechnicalSnapshot] = {}
        self._last_analysis_time: Dict[str, datetime] = defaultdict(lambda: datetime.min)
        self._last_signal_time: Dict[str, datetime] = defaultdict(lambda: datetime.min)

        # Configurable thresholds (can be updated via set_alert_preferences)
        self.analysis_interval = self.alert_settings.get("analysis_interval_seconds", 60)
        self.signal_cooldown = self.alert_settings.get("signal_cooldown_seconds", 300)
        self.min_signal_confidence = self.alert_settings.get("min_signal_confidence", 40)
        self.max_alerts_per_hour = self.alert_settings.get("max_alerts_per_hour", 5)

        # Rate limiting
        self.alert_count = defaultdict(int)
        self.alert_reset_time = datetime.now()
        self.market_timezone = timezone(timedelta(hours=5, minutes=30))

        logging.info(
            f"Smart Opportunity Scanner initialized with {len(self.watchlist)} stocks"
        )

    def start_scanner(self):
        """Start the background analysis thread."""
        if self.running:
            logging.warning("Scanner is already running")
            return

        self.running = True
        self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self._analysis_thread.start()
        logging.info("Smart Opportunity Scanner started")

    def stop_scanner(self):
        """Stop the analysis thread."""
        if not self.running:
            logging.warning("Scanner is not running")
            return

        self.running = False
        self.tick_queue.put(None)  # Sentinel to unblock queue.get()
        if self._analysis_thread:
            self._analysis_thread.join(timeout=5)
        self._news_executor.shutdown(wait=False)
        logging.info("Smart Opportunity Scanner stopped")

    def _analysis_loop(self):
        """Main loop: consume ticks from the queue, periodically run TA + signal evaluation."""
        logging.info("Analysis loop started")

        while self.running:
            try:
                # Drain the tick queue with a timeout
                try:
                    tick = self.tick_queue.get(timeout=1.0)
                    if tick is None:  # Sentinel
                        break
                except queue.Empty:
                    continue

                symbol = tick.get("symbol")
                if not symbol or symbol not in self.watchlist:
                    # Still process index ticks for snapshots but don't generate signals
                    if symbol in ("NIFTY50", "NIFTYBANK"):
                        self._try_compute_snapshot(symbol, tick)
                    continue

                now = datetime.now()

                # Throttle: only run full TA once per analysis_interval per symbol
                elapsed = (now - self._last_analysis_time[symbol]).total_seconds()
                if elapsed < self.analysis_interval:
                    continue

                self._last_analysis_time[symbol] = now

                # Compute technical analysis
                snapshot = self._try_compute_snapshot(symbol, tick)
                if not snapshot:
                    continue

                # Save indicators to database
                self.database.save_technical_indicators(asdict(snapshot))

                # Get previous snapshot for crossover detection
                prev_snapshot = self._last_snapshots.get(symbol)
                self._last_snapshots[symbol] = snapshot

                # Evaluate signal
                signal_type, confidence, reasons = SignalEvaluator.evaluate(
                    snapshot, prev_snapshot
                )

                # Only alert on BUY/SELL with sufficient confidence
                if signal_type in ("BUY", "SELL") and confidence >= self.min_signal_confidence:
                    if self._should_alert(symbol):
                        self._generate_signal(symbol, signal_type, confidence, reasons, snapshot)

            except Exception as e:
                logging.error(f"Error in analysis loop: {e}")
                time.sleep(1)

    def _try_compute_snapshot(self, symbol: str, tick: dict) -> Optional[TechnicalSnapshot]:
        """Compute a full TechnicalSnapshot for a symbol from buffered data."""
        try:
            daily_closes = self.data_buffer.get_daily_series(symbol, "close", 200)
            daily_highs = self.data_buffer.get_daily_series(symbol, "high", 200)
            daily_lows = self.data_buffer.get_daily_series(symbol, "low", 200)
            daily_volumes = self.data_buffer.get_daily_series(symbol, "volume", 200)

            if daily_closes is None or len(daily_closes) < 20:
                return None

            # Get optional data for new indicators
            daily_opens = self.data_buffer.get_daily_series(symbol, "open", 200)
            nifty_closes = self.data_buffer.get_daily_series("NIFTY50", "close", 200)
            intraday_df = None
            if hasattr(self.data_buffer, 'get_intraday_dataframe'):
                intraday_df = self.data_buffer.get_intraday_dataframe(symbol)

            snapshot = TechnicalAnalysis.compute_snapshot(
                symbol=symbol,
                daily_closes=daily_closes,
                daily_highs=daily_highs,
                daily_lows=daily_lows,
                daily_volumes=daily_volumes,
                current_price=tick.get("ltp", 0),
                current_volume=tick.get("volume", 0),
                prev_close=tick.get("cp", 0),
                daily_opens=daily_opens,
                intraday_df=intraday_df,
                nifty_closes=nifty_closes,
            )

            self._last_snapshots[symbol] = snapshot
            return snapshot

        except Exception as e:
            logging.error(f"Failed to compute analysis for {symbol}: {e}")
            return None

    def _generate_signal(
        self,
        symbol: str,
        signal_type: str,
        confidence: float,
        reasons: List[str],
        snapshot: TechnicalSnapshot,
    ):
        """Generate a trading signal alert with optional news context."""
        # Fetch news asynchronously with a timeout
        news_context = ""
        if self.news_analyzer:
            try:
                future = self._news_executor.submit(
                    self.news_analyzer.get_trend_explanation, symbol
                )
                news_context = future.result(timeout=5.0)
            except Exception:
                news_context = "News context unavailable"

        # Save signal to database
        signal_data = {
            "symbol": symbol,
            "signal_type": signal_type,
            "confidence": confidence,
            "reasons": json.dumps(reasons),
            "technical_summary": json.dumps({
                "rsi": snapshot.rsi_14,
                "macd_histogram": snapshot.macd_histogram,
                "macd_crossover": snapshot.macd_crossover,
                "trend": snapshot.trend,
                "volume_ratio": snapshot.volume_ratio,
                "price": snapshot.current_price,
                "sma_20": snapshot.sma_20,
                "sma_50": snapshot.sma_50,
                "sma_200": snapshot.sma_200,
            }),
            "news_context": news_context,
        }
        self.database.save_signal(signal_data)

        # Create and send notification
        reasons_text = "; ".join(reasons[:3])
        message = (
            f"{signal_type} SIGNAL: {symbol} @ Rs.{snapshot.current_price:.2f} | "
            f"Confidence: {confidence:.0f}% | "
            f"Reasons: {reasons_text}"
        )
        if news_context and news_context != "News context unavailable":
            message += f"\nContext: {news_context}"

        notification = self.notification_manager.create_signal_notification({
            "symbol": symbol,
            "signal_type": signal_type,
            "confidence": confidence,
            "reasons": reasons,
            "message": message,
            "snapshot": asdict(snapshot),
            "news_context": news_context,
        })
        self.notification_manager.add_notification(notification)

        self._record_alert(symbol)
        logging.info(f"Signal generated: {signal_type} {symbol} @ {confidence:.0f}%")

    # --- Rate Limiting ---

    def _should_alert(self, symbol: str) -> bool:
        """Check if an alert should be sent for this symbol (rate limiting)."""
        now = datetime.now()

        # Reset hourly counters
        if now - self.alert_reset_time > timedelta(hours=1):
            self.alert_count.clear()
            self.alert_reset_time = now

        # Per-symbol cooldown
        if (now - self._last_signal_time[symbol]).total_seconds() < self.signal_cooldown:
            return False

        # Max alerts per hour per symbol
        if self.alert_count.get(symbol, 0) >= self.max_alerts_per_hour:
            return False

        return True

    def _record_alert(self, symbol: str):
        """Record that an alert was sent for rate limiting."""
        self._last_signal_time[symbol] = datetime.now()
        self.alert_count[symbol] = self.alert_count.get(symbol, 0) + 1

    # --- Watchlist Management (preserved interface) ---

    def add_to_watchlist(self, symbol: str) -> bool:
        """Add a stock to the watchlist."""
        try:
            symbol = symbol.upper()
            if symbol not in self.watchlist:
                self.watchlist.append(symbol)
                self.config["stocks"] = self.watchlist
                with open("config/watchlist.json", "w") as f:
                    json.dump(self.config, f, indent=2)
                logging.info(f"Added {symbol} to watchlist")
                return True
            else:
                logging.info(f"{symbol} already in watchlist")
                return False
        except Exception as e:
            logging.error(f"Error adding {symbol} to watchlist: {str(e)}")
            return False

    def remove_from_watchlist(self, symbol: str) -> bool:
        """Remove a stock from the watchlist."""
        try:
            symbol = symbol.upper()
            if symbol in self.watchlist:
                self.watchlist.remove(symbol)
                self.config["stocks"] = self.watchlist
                with open("config/watchlist.json", "w") as f:
                    json.dump(self.config, f, indent=2)
                logging.info(f"Removed {symbol} from watchlist")
                return True
            else:
                logging.info(f"{symbol} not in watchlist")
                return False
        except Exception as e:
            logging.error(f"Error removing {symbol} from watchlist: {str(e)}")
            return False

    def get_scanner_status(self) -> Dict:
        """Get current scanner status."""
        return {
            "running": self.running,
            "market_open": self._is_market_open(),
            "watchlist_size": len(self.watchlist),
            "recent_opportunities": len(self.database.get_recent_signals(10)),
            "alert_counts": dict(self.alert_count),
            "symbols_analyzed": len(self._last_snapshots),
            "mode": "realtime_streaming",
        }

    def get_technical_snapshot(self, symbol: str) -> Optional[TechnicalSnapshot]:
        """Get the latest cached TechnicalSnapshot for a symbol (used by MCP tools)."""
        return self._last_snapshots.get(symbol.upper())

    def _is_market_open(self) -> bool:
        """Check if Indian equity market is currently open."""
        try:
            now = datetime.now(self.market_timezone)
            if now.weekday() >= 5:
                return False
            start_time = datetime.strptime("09:15", "%H:%M").time()
            end_time = datetime.strptime("15:30", "%H:%M").time()
            return start_time <= now.time() <= end_time
        except Exception as e:
            logging.error(f"Error checking market hours: {str(e)}")
            return False

    def _load_config(self, config_path: str) -> Dict:
        """Load scanner configuration from JSON file."""
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"Watchlist config not found: {config_path}")
            return {"stocks": [], "alert_settings": {}}
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in watchlist config: {config_path}")
            return {"stocks": [], "alert_settings": {}}
