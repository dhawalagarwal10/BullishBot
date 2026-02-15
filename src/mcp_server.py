import logging
import json
import os
import sys
import queue
from typing import Dict, List, Optional
from datetime import datetime
# Ensure working directory is project root (parent of src/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from mcp.server.fastmcp import FastMCP
from broker_client import UpstoxClient
from database import TradingDatabase
from opportunity_scanner import SmartOpportunityScanner
from notification_manager import NotificationManager
from instrument_resolver import InstrumentResolver
from realtime_streamer import RealtimeStreamer, DataBuffer
from news_watcher import NewsWatcher
from paper_trader import PaperTrader
from technical_analysis import compute_live_analysis

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading.log'),
        logging.StreamHandler()
    ]
)

class UpstoxMCPServer:
    def __init__(self):
        self.mcp = FastMCP("upstox-trading")
        self.upstox_client = None
        self.database = None
        self.scanner = None
        self.notification_manager = None
        self.paper_trader = None
        self.paper_trading_enabled = False
        self.setup_components()
        self.setup_tools()
        
        logging.info("Upstox MCP Server initialized")
    
    def setup_components(self):
        """Initialize all components"""
        try:
            self.database = TradingDatabase()
            self.upstox_client = UpstoxClient()
            self.notification_manager = NotificationManager()

            # Instrument resolver for symbol <-> instrument key mapping
            self.instrument_resolver = InstrumentResolver()

            # Data buffer and tick queue for real-time streaming pipeline
            self.data_buffer = DataBuffer()
            self.tick_queue = queue.Queue()

            # News watcher for "why is it trending" context + continuous monitoring
            self.news_watcher = NewsWatcher(notification_manager=self.notification_manager)

            # Smart opportunity scanner (replaces old polling-based scanner)
            self.scanner = SmartOpportunityScanner(
                data_buffer=self.data_buffer,
                database=self.database,
                notification_manager=self.notification_manager,
                news_analyzer=self.news_watcher,
            )
            # Share the tick queue between streamer and scanner
            self.scanner.tick_queue = self.tick_queue

            # Real-time data streamer (WebSocket + historical data)
            access_token = os.getenv("UPSTOX_ACCESS_TOKEN", "")
            self.streamer = RealtimeStreamer(
                access_token=access_token,
                instrument_resolver=self.instrument_resolver,
                data_buffer=self.data_buffer,
                tick_queue=self.tick_queue,
                watchlist=self.scanner.watchlist,
            )

            # Paper trading setup
            try:
                with open("config/watchlist.json", "r") as f:
                    watchlist_config = json.load(f)
                paper_config = watchlist_config.get("paper_trading", {})
                self.paper_trading_enabled = paper_config.get("enabled", False)
            except Exception:
                self.paper_trading_enabled = False

            if self.paper_trading_enabled:
                starting_cash = paper_config.get("starting_cash", 1_000_000)
                self.paper_trader = PaperTrader(
                    db_path=self.database.db_path,
                    data_buffer=self.data_buffer,
                    upstox_client=self.upstox_client,
                )
                self.paper_trader.DEFAULT_STARTING_CASH = starting_cash
                self.paper_trader._ensure_account()
                logging.info(f"Paper trading ENABLED (starting cash: ₹{starting_cash:,.2f})")
            else:
                logging.info("Paper trading DISABLED - using real Upstox account")

            # Start background components
            self.scanner.start_scanner()
            self.streamer.start()

            mode_label = "PAPER TRADING" if self.paper_trading_enabled else "LIVE"
            self.notification_manager.add_notification(
                self.notification_manager.create_system_notification(
                    f"MCP Trading Server started [{mode_label}] with real-time streaming and smart analysis", "MEDIUM"
                )
            )

        except Exception as e:
            logging.error(f"Error setting up components: {str(e)}")
            raise
    
    def setup_tools(self):
        """Set up all MCP tools"""
        @self.mcp.tool()
        def get_quote(symbol: str, exchange: str = "NSE") -> str:
            """Get current price quote for a stock"""
            try:
                inst_key = self.instrument_resolver.get_instrument_key(symbol.upper())
                quote = self.upstox_client.get_quote(
                    symbol.upper(), exchange.upper(), instrument_key=inst_key
                )
                if quote:
                    return f"""📊 {symbol.upper()} Quote:
Price: ₹{quote['current_price']:.2f} ({quote['change_percent']:+.1f}%)
High: ₹{quote['high']:.2f} | Low: ₹{quote['low']:.2f}
Volume: {quote['volume']:,}
Updated: {quote['timestamp'][:19]}"""
                else:
                    return f"❌ Could not fetch quote for {symbol}. Please check symbol and try again."
            except Exception as e:
                logging.error(f"Error in get_quote: {str(e)}")
                return f"❌ Error fetching quote: {str(e)}"
        
        @self.mcp.tool()
        def place_order(symbol: str, quantity: int, action: str,
                       order_type: str = "MARKET", price: float = 0) -> str:
            """Place a buy or sell order. Routes through paper trader when paper trading is enabled."""
            try:
                if action.upper() not in ["BUY", "SELL"]:
                    return "❌ Action must be 'BUY' or 'SELL'"

                if quantity <= 0:
                    return "❌ Quantity must be positive"

                if self.paper_trading_enabled and self.paper_trader:
                    order_result = self.paper_trader.place_order(
                        symbol.upper(), quantity, action.upper(), order_type.upper(), price
                    )
                else:
                    if action.upper() == "BUY":
                        balance = self.upstox_client.get_balance()
                        if balance:
                            required_amount = quantity * price if price > 0 else quantity * 1000
                            if balance['available_margin'] < required_amount:
                                return f"❌ Insufficient funds. Available: ₹{balance['available_margin']:,.2f}"

                    order_result = self.upstox_client.place_order(
                        symbol.upper(), quantity, action.upper(), order_type.upper(), price
                    )

                if order_result:
                    if order_result.get("status") == "REJECTED":
                        error_msg = order_result.get("broker_response", {}).get("error", "Order rejected")
                        return f"❌ {error_msg}"

                    self.database.save_order(order_result)
                    self.notification_manager.add_notification(
                        self.notification_manager.create_order_notification(order_result)
                    )

                    mode_tag = " [PAPER]" if self.paper_trading_enabled else ""
                    fill_info = ""
                    if self.paper_trading_enabled and order_result.get("price"):
                        fill_info = f"\nFill Price: ₹{order_result['price']:.2f}"

                    return f"""✅ Order Placed Successfully!{mode_tag}
Order ID: {order_result['order_id']}
Stock: {symbol.upper()}
Action: {action.upper()} {quantity} shares
Type: {order_type.upper()}
{f'Price: ₹{price:.2f}' if price > 0 and not self.paper_trading_enabled else 'Price: Market'}{fill_info}
Status: {order_result['status']}"""
                else:
                    return f"❌ Order placement failed for {symbol}"

            except Exception as e:
                logging.error(f"Error in place_order: {str(e)}")
                return f"❌ Error placing order: {str(e)}"
        
        @self.mcp.tool()
        def get_portfolio() -> str:
            """Get current portfolio holdings. Shows paper portfolio when paper trading is enabled."""
            try:
                if self.paper_trading_enabled and self.paper_trader:
                    portfolio = self.paper_trader.get_portfolio()
                else:
                    portfolio = self.upstox_client.get_portfolio()

                if portfolio:
                    self.notification_manager.add_notification(
                        self.notification_manager.create_portfolio_notification(portfolio)
                    )
                    holdings_text = ""
                    for holding in portfolio['holdings']:
                        pnl_emoji = "📈" if holding['pnl'] >= 0 else "📉"
                        holdings_text += f"""
{holding['symbol']}: {holding['quantity']} shares
  Avg: ₹{holding['average_price']:.2f} | Current: ₹{holding['current_price']:.2f}
  Value: ₹{holding['current_value']:,.2f} | P&L: {pnl_emoji} ₹{holding['pnl']:,.2f} ({holding['pnl_percent']:+.1f}%)"""

                    total_pnl_emoji = "📈" if portfolio['total_pnl'] >= 0 else "📉"
                    mode_tag = " [PAPER]" if self.paper_trading_enabled else ""

                    return f"""💼 Portfolio Summary:{mode_tag}
Total Value: ₹{portfolio['total_value']:,.2f}
Total P&L: {total_pnl_emoji} ₹{portfolio['total_pnl']:,.2f} ({portfolio['total_pnl_percent']:+.1f}%)

Holdings:{holdings_text}"""
                else:
                    return "❌ Could not fetch portfolio data"

            except Exception as e:
                logging.error(f"Error in get_portfolio: {str(e)}")
                return f"❌ Error fetching portfolio: {str(e)}"
        
        @self.mcp.tool()
        def get_order_status(order_id: str) -> str:
            """Check the status of an order"""
            try:
                if self.paper_trading_enabled and self.paper_trader and order_id.startswith("PAPER-"):
                    paper_order = self.paper_trader.get_order_status(order_id)
                    if paper_order:
                        return f"""📋 Order Status [PAPER]:
Order ID: {order_id}
Stock: {paper_order['symbol']}
Status: {paper_order['status']}
Quantity: {paper_order['quantity']} (Filled: {paper_order['filled_quantity']})
Fill Price: ₹{paper_order['average_price']:.2f}
Time: {paper_order['timestamp']}"""
                    else:
                        return f"❌ Paper order {order_id} not found"

                db_order = self.database.get_order_status(order_id)
                broker_order = self.upstox_client.get_order_status(order_id)

                if broker_order:
                    if db_order and db_order['status'] != broker_order['status']:
                        db_order['status'] = broker_order['status']
                        db_order['broker_response'] = broker_order
                        self.database.save_order(db_order)

                    return f"""📋 Order Status:
Order ID: {order_id}
Stock: {broker_order['symbol']}
Status: {broker_order['status']}
Quantity: {broker_order['quantity']} (Filled: {broker_order['filled_quantity']})
Price: ₹{broker_order['price']:.2f}
{f"Avg Fill Price: ₹{broker_order['average_price']:.2f}" if broker_order['average_price'] > 0 else ""}
Time: {broker_order['timestamp']}"""

                elif db_order:
                    return f"""📋 Order Status (from database):
Order ID: {order_id}
Stock: {db_order['symbol']}
Status: {db_order['status']}
Quantity: {db_order['quantity']}
Type: {db_order['order_type']}
Time: {db_order['timestamp']}"""

                else:
                    return f"❌ Order {order_id} not found"

            except Exception as e:
                logging.error(f"Error in get_order_status: {str(e)}")
                return f"❌ Error checking order status: {str(e)}"
        
        @self.mcp.tool()
        def cancel_order(order_id: str) -> str:
            """Cancel a pending order"""
            try:
                success = self.upstox_client.cancel_order(order_id)
                if success:
                    order_data = self.database.get_order_status(order_id)
                    if order_data:
                        order_data['status'] = 'CANCELLED'
                        self.database.save_order(order_data)
                    
                    return f"✅ Order {order_id} cancelled successfully"
                else:
                    return f"❌ Failed to cancel order {order_id}"
                    
            except Exception as e:
                logging.error(f"Error in cancel_order: {str(e)}")
                return f"❌ Error cancelling order: {str(e)}"
        
        @self.mcp.tool()
        def get_balance() -> str:
            """Check account balance and available funds. Shows paper balance when paper trading is enabled."""
            try:
                if self.paper_trading_enabled and self.paper_trader:
                    balance = self.paper_trader.get_balance()
                else:
                    balance = self.upstox_client.get_balance()

                if balance:
                    mode_tag = " [PAPER]" if self.paper_trading_enabled else ""
                    return f"""💰 Account Balance:{mode_tag}
Available Margin: ₹{balance['available_margin']:,.2f}
Used Margin: ₹{balance['used_margin']:,.2f}
Total Margin: ₹{balance['total_margin']:,.2f}

P&L:
Unrealized: ₹{balance['unrealized_pnl']:,.2f}
Realized: ₹{balance['realized_pnl']:,.2f}"""
                else:
                    return "❌ Could not fetch balance information"

            except Exception as e:
                logging.error(f"Error in get_balance: {str(e)}")
                return f"❌ Error fetching balance: {str(e)}"
        
        @self.mcp.tool()
        def check_opportunities() -> str:
            """Get current trading signals detected by the smart scanner"""
            try:
                signals = self.database.get_recent_signals(5)
                if signals:
                    result = "🔍 Recent Trading Signals:\n\n"
                    for sig in reversed(signals):
                        timestamp = sig['timestamp'][:19] if sig.get('timestamp') else 'Unknown'
                        reasons = json.loads(sig.get('reasons', '[]'))
                        reasons_text = "; ".join(reasons[:2]) if reasons else "N/A"

                        emoji = "🟢" if sig['signal_type'] == "BUY" else "🔴" if sig['signal_type'] == "SELL" else "🟡"
                        result += f"""{emoji} {sig['signal_type']} | {sig['symbol']} | Confidence: {sig['confidence']:.0f}%
Reasons: {reasons_text}
Time: {timestamp}

"""
                    return result
                else:
                    return "📊 No recent signals detected. Smart scanner is monitoring market with real-time data..."

            except Exception as e:
                logging.error(f"Error in check_opportunities: {str(e)}")
                return f"❌ Error checking signals: {str(e)}"
        
        @self.mcp.tool()
        def manage_watchlist(action: str, symbol: Optional[str] = None) -> str:
            """Manage watchlist for opportunity scanner (add/remove/view)"""
            try:
                action = action.lower()
                
                if action == "view":
                    watchlist = self.scanner.watchlist
                    return f"📋 Current Watchlist ({len(watchlist)} stocks):\n" + ", ".join(watchlist)
                
                elif action == "add":
                    if not symbol:
                        return "❌ Symbol required for adding to watchlist"
                    
                    success = self.scanner.add_to_watchlist(symbol)
                    if success:
                        return f"✅ {symbol.upper()} added to watchlist"
                    else:
                        return f"❌ Failed to add {symbol.upper()} to watchlist"
                
                elif action == "remove":
                    if not symbol:
                        return "❌ Symbol required for removing from watchlist"
                    
                    success = self.scanner.remove_from_watchlist(symbol)
                    if success:
                        return f"✅ {symbol.upper()} removed from watchlist"
                    else:
                        return f"❌ Failed to remove {symbol.upper()} from watchlist"
                
                else:
                    return "❌ Action must be 'view', 'add', or 'remove'"
                    
            except Exception as e:
                logging.error(f"Error in manage_watchlist: {str(e)}")
                return f"❌ Error managing watchlist: {str(e)}"
        
        @self.mcp.tool()
        def get_scanner_status() -> str:
            """Get current status of the opportunity scanner"""
            try:
                status = self.scanner.get_scanner_status()
                market_status = "🟢 Open" if status['market_open'] else "🔴 Closed"
                scanner_status = "🟢 Running" if status['running'] else "🔴 Stopped"

                return f"""🔍 Scanner Status:
Market: {market_status}
Scanner: {scanner_status}
Mode: {status.get('mode', 'unknown')}
Watchlist: {status['watchlist_size']} stocks
Symbols Analyzed: {status.get('symbols_analyzed', 0)}
Recent Signals: {status['recent_opportunities']}
Alert Counts: {status['alert_counts']}"""

            except Exception as e:
                logging.error(f"Error in get_scanner_status: {str(e)}")
                return f"❌ Error getting scanner status: {str(e)}"
        
        @self.mcp.tool()
        def get_notifications() -> str:
            """Get recent notifications and alerts"""
            try:
                notifications = self.notification_manager.get_formatted_notifications(10)
                summary = self.notification_manager.get_notification_summary()
                
                return f"""📬 Notifications Summary:
Total: {summary['total_notifications']} | Unread: {summary['unread_notifications']}

Recent Notifications:
{notifications}"""
                
            except Exception as e:
                logging.error(f"Error in get_notifications: {str(e)}")
                return f"❌ Error getting notifications: {str(e)}"
        
        @self.mcp.tool()
        def mark_notifications_read() -> str:
            """Mark all notifications as read"""
            try:
                success = self.notification_manager.mark_all_as_read()
                if success:
                    return "✅ All notifications marked as read"
                else:
                    return "❌ Failed to mark notifications as read"
                    
            except Exception as e:
                logging.error(f"Error in mark_notifications_read: {str(e)}")
                return f"❌ Error marking notifications as read: {str(e)}"
        
        @self.mcp.tool()
        def get_paper_trade_history(limit: int = 20) -> str:
            """Get recent paper trade history. Only available when paper trading is enabled."""
            try:
                if not self.paper_trading_enabled or not self.paper_trader:
                    return "Paper trading is not enabled. Set paper_trading.enabled to true in config/watchlist.json."

                trades = self.paper_trader.get_trade_history(limit)
                if not trades:
                    return "No paper trades yet. Use place_order to simulate trades."

                result = f"📜 Paper Trade History (last {len(trades)}):\n\n"
                for t in trades:
                    emoji = "🟢" if t["action"] == "BUY" else "🔴"
                    result += f"""{emoji} {t['action']} {t['quantity']} {t['symbol']} @ ₹{t['fill_price']:.2f}
   Status: {t['status']} | Time: {t['created_at'][:19]}
"""
                return result

            except Exception as e:
                logging.error(f"Error in get_paper_trade_history: {str(e)}")
                return f"❌ Error fetching trade history: {str(e)}"

        @self.mcp.tool()
        def reset_paper_account(starting_cash: float = 1000000) -> str:
            """Reset paper trading account to starting state. Clears all holdings and trade history."""
            try:
                if not self.paper_trading_enabled or not self.paper_trader:
                    return "Paper trading is not enabled. Set paper_trading.enabled to true in config/watchlist.json."

                success = self.paper_trader.reset_account(starting_cash)
                if success:
                    return f"✅ Paper trading account reset. Starting cash: ₹{starting_cash:,.2f}"
                else:
                    return "❌ Failed to reset paper trading account"

            except Exception as e:
                logging.error(f"Error in reset_paper_account: {str(e)}")
                return f"❌ Error resetting account: {str(e)}"

        @self.mcp.tool()
        def get_system_status() -> str:
            """Get overall system status"""
            try:
                scanner_status = self.scanner.get_scanner_status()
                notification_summary = self.notification_manager.get_notification_summary()

                rel_key = self.instrument_resolver.get_instrument_key("RELIANCE")
                test_quote = self.upstox_client.get_quote("RELIANCE", instrument_key=rel_key)
                upstox_status = "🟢 Connected" if test_quote else "🔴 Disconnected"

                data_symbols = self.data_buffer.get_symbols_with_data()

                trading_mode = "📝 PAPER" if self.paper_trading_enabled else "💹 LIVE"

                return f"""🖥️ System Status:
Trading Mode: {trading_mode}
Upstox API: {upstox_status}
Scanner: {"🟢 Running" if scanner_status['running'] else "🔴 Stopped"}
Streaming Mode: {scanner_status.get('mode', 'unknown')}
Market: {"🟢 Open" if scanner_status['market_open'] else "🔴 Closed"}
Database: 🟢 Connected
Notifications: {notification_summary['unread_notifications']} unread

Watchlist: {scanner_status['watchlist_size']} stocks
Historical Data Loaded: {len(data_symbols)} symbols
Symbols Analyzed: {scanner_status.get('symbols_analyzed', 0)}
Recent Signals: {scanner_status['recent_opportunities']}"""

            except Exception as e:
                logging.error(f"Error in get_system_status: {str(e)}")
                return f"❌ Error getting system status: {str(e)}"

        @self.mcp.tool()
        def debug_api_connection() -> str:
            """Diagnose Upstox API connection issues.

            Checks: token status, instrument resolution, quote API, and historical API.
            Use this when get_quote or analyze_stock is failing.
            """
            try:
                results = []

                # 1. Token status
                token_info = self.upstox_client.get_token_info()
                results.append("=== Token Status ===")
                results.append(f"Token loaded: {token_info['token_loaded']}")
                results.append(f"Token length: {token_info['token_length']} chars")
                results.append(f"Token last 4: ...{token_info['token_last_4']}")
                results.append(f"API key loaded: {token_info['api_key_loaded']}")

                # 2. Try refreshing token from .env
                results.append("")
                results.append("=== Token Refresh ===")
                refreshed = self.upstox_client.refresh_token()
                if refreshed:
                    results.append("Token was refreshed from .env (new token detected)")
                    token_info = self.upstox_client.get_token_info()
                    results.append(f"New token last 4: ...{token_info['token_last_4']}")
                else:
                    results.append("Token unchanged (already current or .env not updated)")

                # 3. Instrument resolution test
                results.append("")
                results.append("=== Instrument Resolution ===")
                test_symbol = "RELIANCE"
                inst_key = self.instrument_resolver.get_instrument_key(test_symbol)
                if inst_key:
                    results.append(f"{test_symbol} -> {inst_key}")
                else:
                    results.append(f"FAILED to resolve {test_symbol}")

                # 4. Quote API test
                results.append("")
                results.append("=== Quote API Test ===")
                try:
                    quote = self.upstox_client.get_quote(test_symbol, instrument_key=inst_key)
                    if quote:
                        results.append(f"SUCCESS: {test_symbol} = Rs.{quote['current_price']:.2f}")
                    else:
                        results.append(f"FAILED: get_quote returned None")
                except Exception as e:
                    results.append(f"ERROR: {str(e)}")

                # 5. Historical API test
                results.append("")
                results.append("=== Historical API Test ===")
                try:
                    df = self.upstox_client.get_historical_candles(test_symbol, days=5, instrument_key=inst_key)
                    if df is not None and len(df) > 0:
                        last_close = df['close'].iloc[-1]
                        last_date = df['timestamp'].iloc[-1]
                        results.append(f"SUCCESS: Got {len(df)} candles")
                        results.append(f"Last candle: {last_date.strftime('%Y-%m-%d')} close=Rs.{last_close:.2f}")
                    else:
                        results.append("FAILED: No candles returned")
                except Exception as e:
                    results.append(f"ERROR: {str(e)}")

                return "\n".join(results)
            except Exception as e:
                logging.error(f"Error in debug_api_connection: {str(e)}")
                return f"Error running diagnostics: {str(e)}"

        # --- New Technical Analysis & Smart Alert Tools ---

        @self.mcp.tool()
        def get_technical_analysis(symbol: str) -> str:
            """Get full technical analysis breakdown for a stock including RSI, MACD, SMAs, Bollinger Bands, volume, support/resistance, and trend"""
            try:
                s = self.scanner.get_technical_snapshot(symbol.upper())
                if s:
                    cross_info = ""
                    if s.golden_cross:
                        cross_info = "\n  ** GOLDEN CROSS detected! **"
                    if s.death_cross:
                        cross_info = "\n  ** DEATH CROSS detected! **"

                    extra = ""
                    if s.adx is not None:
                        extra += f"\nADX: {s.adx:.1f} (DI+={s.plus_di:.1f} DI-={s.minus_di:.1f})"
                    if s.stoch_rsi_k is not None:
                        extra += f"\nStoch RSI: K={s.stoch_rsi_k:.1f} D={s.stoch_rsi_d:.1f}"
                    if s.atr_14 is not None:
                        extra += f"\nATR: Rs.{s.atr_14:.2f}"
                    if s.ichimoku_signal:
                        extra += f"\nIchimoku: {s.ichimoku_signal.replace('_', ' ').upper()}"
                    if s.candlestick_pattern:
                        extra += f"\nCandle Pattern: {s.candlestick_pattern.replace('_', ' ').title()}"
                    if s.rs_vs_nifty is not None:
                        extra += f"\nRS vs NIFTY: {s.rs_vs_nifty:.3f}"

                    return f"""Technical Analysis: {s.symbol}
Price: Rs.{s.current_price:.2f} ({s.change_percent:+.1f}%)

RSI (14): {s.rsi_14:.1f} [{s.rsi_signal.upper()}]
MACD: Line={s.macd_line:.2f} Signal={s.macd_signal:.2f} Hist={s.macd_histogram:.2f} [{s.macd_crossover.replace('_', ' ')}]

Moving Averages:
  SMA 20:  Rs.{s.sma_20:.2f} [Price {s.price_vs_sma20}]
  SMA 50:  Rs.{s.sma_50:.2f} [Price {s.price_vs_sma50}]
  SMA 200: Rs.{s.sma_200:.2f} [Price {s.price_vs_sma200}]{cross_info}

Bollinger Bands (20, 2):
  Upper: Rs.{s.bb_upper:.2f} | Middle: Rs.{s.bb_middle:.2f} | Lower: Rs.{s.bb_lower:.2f}
  Width: {s.bb_width:.3f} | Position: {s.bb_position.replace('_', ' ')}

Volume:
  Current: {s.current_volume:,} | Avg(20d): {s.avg_volume_20:,.0f}
  Ratio: {s.volume_ratio:.1f}x [{s.volume_signal.upper()}]

Support: Rs.{s.support_level:.2f} ({s.distance_to_support:.1f}% away)
Resistance: Rs.{s.resistance_level:.2f} ({s.distance_to_resistance:.1f}% away)

Trend: {s.trend.replace('_', ' ').upper()} (Strength: {s.trend_strength:.0f}/100){extra}"""

                # Fallback: fetch fresh data on-demand instead of stale DB
                s = compute_live_analysis(
                    symbol.upper(),
                    self.upstox_client,
                    self.instrument_resolver,
                )
                if not s:
                    return f"Could not fetch technical analysis for {symbol.upper()}. Check if the symbol is valid on NSE."

                cross_info = ""
                if s.golden_cross:
                    cross_info = "\n  ** GOLDEN CROSS detected! **"
                if s.death_cross:
                    cross_info = "\n  ** DEATH CROSS detected! **"

                extra = ""
                if s.adx is not None:
                    extra += f"\nADX: {s.adx:.1f} (DI+={s.plus_di:.1f} DI-={s.minus_di:.1f})"
                if s.stoch_rsi_k is not None:
                    extra += f"\nStoch RSI: K={s.stoch_rsi_k:.1f} D={s.stoch_rsi_d:.1f}"
                if s.atr_14 is not None:
                    extra += f"\nATR: Rs.{s.atr_14:.2f}"
                if s.ichimoku_signal:
                    extra += f"\nIchimoku: {s.ichimoku_signal.replace('_', ' ').upper()}"
                if s.candlestick_pattern:
                    extra += f"\nCandle Pattern: {s.candlestick_pattern.replace('_', ' ').title()}"
                if s.rs_vs_nifty is not None:
                    extra += f"\nRS vs NIFTY: {s.rs_vs_nifty:.3f}"

                return f"""Technical Analysis: {s.symbol} (on-demand)
Price: Rs.{s.current_price:.2f} ({s.change_percent:+.1f}%)

RSI (14): {s.rsi_14:.1f} [{s.rsi_signal.upper()}]
MACD: Line={s.macd_line:.2f} Signal={s.macd_signal:.2f} Hist={s.macd_histogram:.2f} [{s.macd_crossover.replace('_', ' ')}]

Moving Averages:
  SMA 20:  Rs.{s.sma_20:.2f} [Price {s.price_vs_sma20}]
  SMA 50:  Rs.{s.sma_50:.2f} [Price {s.price_vs_sma50}]
  SMA 200: Rs.{s.sma_200:.2f} [Price {s.price_vs_sma200}]{cross_info}

Bollinger Bands (20, 2):
  Upper: Rs.{s.bb_upper:.2f} | Middle: Rs.{s.bb_middle:.2f} | Lower: Rs.{s.bb_lower:.2f}
  Width: {s.bb_width:.3f} | Position: {s.bb_position.replace('_', ' ')}

Volume:
  Current: {s.current_volume:,} | Avg(20d): {s.avg_volume_20:,.0f}
  Ratio: {s.volume_ratio:.1f}x [{s.volume_signal.upper()}]

Support: Rs.{s.support_level:.2f} ({s.distance_to_support:.1f}% away)
Resistance: Rs.{s.resistance_level:.2f} ({s.distance_to_resistance:.1f}% away)

Trend: {s.trend.replace('_', ' ').upper()} (Strength: {s.trend_strength:.0f}/100){extra}"""
            except Exception as e:
                logging.error(f"Error in get_technical_analysis: {str(e)}")
                return f"Error fetching technical analysis: {str(e)}"

        @self.mcp.tool()
        def get_trend_explanation(symbol: str) -> str:
            """Get explanation of why a stock is trending, combining technical analysis with recent news"""
            try:
                snapshot = self.scanner.get_technical_snapshot(symbol.upper())
                tech_summary = ""
                if snapshot:
                    tech_summary = f"""Technical Summary for {symbol.upper()}:
- Price: Rs.{snapshot.current_price:.2f} ({snapshot.change_percent:+.1f}%)
- RSI: {snapshot.rsi_14:.1f} ({snapshot.rsi_signal})
- MACD: {'Bullish crossover' if snapshot.macd_crossover == 'bullish_crossover' else 'Bearish crossover' if snapshot.macd_crossover == 'bearish_crossover' else 'No crossover'}
- Trend: {snapshot.trend.replace('_', ' ')}
- Volume: {snapshot.volume_ratio:.1f}x average ({snapshot.volume_signal})
- Position: {snapshot.bb_position.replace('_', ' ')} Bollinger Bands
"""
                else:
                    # Fallback to database
                    import sqlite3
                    conn = sqlite3.connect(self.database.db_path, timeout=10)
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT * FROM technical_indicators
                        WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1
                    ''', (symbol.upper(),))
                    row = cursor.fetchone()
                    conn.close()
                    if row:
                        d = dict(row)
                        rsi = d.get('rsi_14', 0)
                        rsi_label = 'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'
                        tech_summary = f"""Technical Summary for {symbol.upper()} (from {d.get('timestamp', '')[:16]}):
- Price: Rs.{d.get('current_price', 0):.2f}
- RSI: {rsi:.1f} ({rsi_label})
- MACD: {(d.get('macd_crossover') or 'none').replace('_', ' ')}
- Trend: {(d.get('trend') or 'unknown').replace('_', ' ')}
- Volume Ratio: {d.get('volume_ratio', 0):.1f}x average
"""

                news = self.news_watcher.get_trend_explanation(symbol.upper())

                if not tech_summary and not news:
                    return f"No trend data available for {symbol.upper()}. Try adding it to the watchlist and running the scanner."

                return f"""Why is {symbol.upper()} Trending?

{tech_summary}
{news}

Note: This analysis is for informational purposes only. Always do your own research before making trading decisions."""
            except Exception as e:
                logging.error(f"Error in get_trend_explanation: {str(e)}")
                return f"Error fetching trend explanation: {str(e)}"

        @self.mcp.tool()
        def get_market_sentiment() -> str:
            """Get overall market sentiment based on NIFTY 50, BANK NIFTY, and watchlist breadth analysis"""
            try:
                nifty = self.scanner.get_technical_snapshot("NIFTY50")
                bank_nifty = self.scanner.get_technical_snapshot("NIFTYBANK")

                bullish_count = 0
                bearish_count = 0
                neutral_count = 0

                for sym in self.scanner.watchlist:
                    snap = self.scanner.get_technical_snapshot(sym)
                    if snap:
                        if snap.trend in ("strong_uptrend", "uptrend"):
                            bullish_count += 1
                        elif snap.trend in ("strong_downtrend", "downtrend"):
                            bearish_count += 1
                        else:
                            neutral_count += 1

                total = bullish_count + bearish_count + neutral_count

                if total > 0:
                    bullish_pct = (bullish_count / total) * 100
                    bearish_pct = (bearish_count / total) * 100
                    neutral_pct = (neutral_count / total) * 100
                    if bullish_pct > 60:
                        overall = "BULLISH"
                    elif bearish_pct > 60:
                        overall = "BEARISH"
                    else:
                        overall = "NEUTRAL"
                else:
                    overall = "INSUFFICIENT DATA"
                    bullish_pct = bearish_pct = neutral_pct = 0

                result = f"Market Sentiment: {overall}\n\n"

                if nifty:
                    result += f"""NIFTY 50: {nifty.current_price:.2f} ({nifty.change_percent:+.1f}%)
  RSI: {nifty.rsi_14:.1f} | Trend: {nifty.trend.replace('_', ' ')}
"""

                if bank_nifty:
                    result += f"""BANK NIFTY: {bank_nifty.current_price:.2f} ({bank_nifty.change_percent:+.1f}%)
  RSI: {bank_nifty.rsi_14:.1f} | Trend: {bank_nifty.trend.replace('_', ' ')}
"""

                if total > 0:
                    result += f"""
Watchlist Breadth ({total} stocks analyzed):
  Bullish: {bullish_count} ({bullish_pct:.0f}%)
  Bearish: {bearish_count} ({bearish_pct:.0f}%)
  Neutral: {neutral_count} ({neutral_pct:.0f}%)"""
                else:
                    result += "\nWatchlist breadth data not yet available. Scanner is still initializing."

                return result
            except Exception as e:
                logging.error(f"Error in get_market_sentiment: {str(e)}")
                return f"Error fetching market sentiment: {str(e)}"

        @self.mcp.tool()
        def analyze_stock(symbol: str) -> str:
            """Run full on-demand technical analysis for ANY NSE stock (not just watchlist).

            Fetches 200-day historical candles and computes all professional indicators:
            RSI, MACD, ADX, Stochastic RSI, ATR, VWAP, Fibonacci, Ichimoku Cloud,
            OBV, Pivot Points, Candlestick Patterns, Bollinger Bands, Support/Resistance,
            and Relative Strength vs NIFTY.

            Also fetches live quote to show current price alongside historical analysis.
            """
            try:
                s = compute_live_analysis(
                    symbol.upper(),
                    self.upstox_client,
                    self.instrument_resolver,
                )
                if not s:
                    return f"Could not analyze {symbol.upper()}. Check if the symbol is valid on NSE."

                # Try to get live quote for current price
                live_price = None
                live_change_pct = None
                price_source = "prev close"
                try:
                    inst_key = self.instrument_resolver.get_instrument_key(symbol.upper())
                    quote = self.upstox_client.get_quote(symbol.upper(), instrument_key=inst_key)
                    if quote and quote.get("current_price", 0) > 0:
                        live_price = quote["current_price"]
                        live_change_pct = quote.get("change_percent", 0)
                        price_source = "live"
                except Exception as e:
                    logging.debug(f"Could not fetch live quote for {symbol}: {e}")

                # Use live price if available, otherwise historical close
                display_price = live_price if live_price else s.current_price
                display_change = live_change_pct if live_change_pct is not None else s.change_percent

                lines = [
                    f"Full Analysis: {s.symbol}",
                    f"Price: Rs.{display_price:.2f} ({display_change:+.1f}%) [{price_source}]",
                    f"Trend: {s.trend.replace('_', ' ').upper()} (Strength: {s.trend_strength:.0f}/100)",
                    "",
                    "--- Momentum ---",
                    f"RSI (14): {s.rsi_14:.1f} [{s.rsi_signal.upper()}]",
                    f"MACD: Line={s.macd_line:.2f} Signal={s.macd_signal:.2f} Hist={s.macd_histogram:.2f} [{s.macd_crossover.replace('_', ' ')}]",
                ]

                if s.adx is not None:
                    lines.append(f"ADX: {s.adx:.1f} (DI+={s.plus_di:.1f} DI-={s.minus_di:.1f}) [{'Trending' if s.adx > 25 else 'Ranging'}]")
                if s.stoch_rsi_k is not None:
                    lines.append(f"Stoch RSI: K={s.stoch_rsi_k:.1f} D={s.stoch_rsi_d:.1f}")

                lines += [
                    "",
                    "--- Moving Averages ---",
                    f"SMA 20:  Rs.{s.sma_20:.2f} [Price {s.price_vs_sma20}]",
                    f"SMA 50:  Rs.{s.sma_50:.2f} [Price {s.price_vs_sma50}]",
                    f"SMA 200: Rs.{s.sma_200:.2f} [Price {s.price_vs_sma200}]",
                ]
                if s.golden_cross:
                    lines.append("  ** GOLDEN CROSS detected! **")
                if s.death_cross:
                    lines.append("  ** DEATH CROSS detected! **")

                lines += [
                    "",
                    "--- Bollinger Bands ---",
                    f"Upper: Rs.{s.bb_upper:.2f} | Middle: Rs.{s.bb_middle:.2f} | Lower: Rs.{s.bb_lower:.2f}",
                    f"Width: {s.bb_width:.3f} | Position: {s.bb_position.replace('_', ' ')}",
                ]

                if s.atr_14 is not None:
                    lines += ["", f"ATR (14): Rs.{s.atr_14:.2f}"]
                if s.vwap is not None:
                    lines.append(f"VWAP: Rs.{s.vwap:.2f}")

                lines += [
                    "",
                    "--- Volume ---",
                    f"Current: {s.current_volume:,} | Avg(20d): {s.avg_volume_20:,.0f}",
                    f"Ratio: {s.volume_ratio:.1f}x [{s.volume_signal.upper()}]",
                ]
                if s.obv is not None:
                    lines.append(f"OBV: {s.obv:,.0f} (SMA: {s.obv_sma_20:,.0f})")

                lines += [
                    "",
                    "--- Support / Resistance ---",
                    f"Support: Rs.{s.support_level:.2f} ({s.distance_to_support:.1f}% away)",
                    f"Resistance: Rs.{s.resistance_level:.2f} ({s.distance_to_resistance:.1f}% away)",
                ]

                if s.pivot_point is not None:
                    lines += [
                        "",
                        "--- Pivot Points ---",
                        f"PP: Rs.{s.pivot_point:.2f}",
                        f"S1: Rs.{s.pivot_s1:.2f} | S2: Rs.{s.pivot_s2:.2f} | S3: Rs.{s.pivot_s3:.2f}",
                        f"R1: Rs.{s.pivot_r1:.2f} | R2: Rs.{s.pivot_r2:.2f} | R3: Rs.{s.pivot_r3:.2f}",
                    ]

                if s.fib_levels:
                    lines += ["", "--- Fibonacci Retracement ---"]
                    for level, price in sorted(s.fib_levels.items(), key=lambda x: x[1], reverse=True):
                        lines.append(f"  {level}: Rs.{price:.2f}")

                if s.ichimoku_tenkan is not None:
                    lines += [
                        "",
                        "--- Ichimoku Cloud ---",
                        f"Tenkan: Rs.{s.ichimoku_tenkan:.2f} | Kijun: Rs.{s.ichimoku_kijun:.2f}",
                        f"Senkou A: Rs.{s.ichimoku_senkou_a:.2f} | Senkou B: Rs.{s.ichimoku_senkou_b:.2f}",
                        f"Signal: {s.ichimoku_signal.replace('_', ' ').upper() if s.ichimoku_signal else 'N/A'}",
                    ]

                if s.candlestick_pattern:
                    lines += ["", f"Candlestick Pattern: {s.candlestick_pattern.replace('_', ' ').title()}"]

                if s.rs_vs_nifty is not None:
                    label = "Outperforming" if s.rs_vs_nifty > 1.0 else "Underperforming"
                    lines.append(f"RS vs NIFTY (20d): {s.rs_vs_nifty:.3f} [{label}]")

                if s.high_52w is not None:
                    lines += [
                        "",
                        f"52-Week High: Rs.{s.high_52w:.2f} ({s.pct_from_52w_high:+.1f}%)",
                        f"52-Week Low:  Rs.{s.low_52w:.2f} ({s.pct_from_52w_low:+.1f}%)",
                    ]

                return "\n".join(lines)
            except Exception as e:
                logging.error(f"Error in analyze_stock: {str(e)}")
                return f"Error analyzing {symbol}: {str(e)}"

        @self.mcp.tool()
        def get_market_radar_results(limit: int = 30, trigger_type: Optional[str] = None) -> str:
            """Get results from the latest market radar scan.

            The market radar periodically scans all 2400+ NSE stocks for unusual activity.

            Args:
                limit: Maximum number of results to return (default 30).
                trigger_type: Filter by type - 'oversold', 'overbought', 'volume_spike',
                              'near_52w_high', or 'macd_bullish_crossover'. None for all.
            """
            try:
                results = self.database.get_recent_radar_results(
                    limit=limit, trigger_type=trigger_type
                )
                if not results:
                    msg = "No market radar results available."
                    if trigger_type:
                        msg += f" (filter: {trigger_type})"
                    msg += " The radar may not have run yet."
                    return msg

                by_type = {}
                for r in results:
                    tt = r.get("trigger_type", "unknown")
                    if tt not in by_type:
                        by_type[tt] = []
                    by_type[tt].append(r)

                lines = [f"Market Radar Results ({len(results)} stocks flagged):\n"]
                for tt, items in by_type.items():
                    lines.append(f"[{tt.replace('_', ' ').upper()}] ({len(items)} stocks)")
                    for item in items:
                        detail = item.get("detail", "")
                        lines.append(
                            f"  {item['symbol']} Rs.{item['price']:.2f} "
                            f"RSI:{item.get('rsi', 0):.0f} Vol:{item.get('volume_ratio', 0):.1f}x"
                            f"{' - ' + detail if detail else ''}"
                        )
                    lines.append("")

                scan_time = results[0].get("scan_time", "")
                if scan_time:
                    lines.append(f"Last scan: {scan_time[:19]}")

                return "\n".join(lines)
            except Exception as e:
                logging.error(f"Error in get_market_radar_results: {str(e)}")
                return f"Error fetching radar results: {str(e)}"

        @self.mcp.tool()
        def set_alert_preferences(
            min_confidence: int = 40,
            signal_cooldown: int = 300,
            analysis_interval: int = 60,
        ) -> str:
            """Configure alert sensitivity and frequency.

            Args:
                min_confidence: Minimum confidence (0-100) to trigger alerts. Default 40.
                signal_cooldown: Seconds between alerts for the same stock. Default 300.
                analysis_interval: Seconds between analysis runs per stock. Default 60.
            """
            try:
                changes = []

                if 0 <= min_confidence <= 100:
                    self.scanner.min_signal_confidence = min_confidence
                    self.database.set_alert_preference("min_confidence", str(min_confidence))
                    changes.append(f"Minimum confidence: {min_confidence}%")

                if signal_cooldown >= 60:
                    self.scanner.signal_cooldown = signal_cooldown
                    self.database.set_alert_preference("signal_cooldown", str(signal_cooldown))
                    changes.append(f"Signal cooldown: {signal_cooldown}s")

                if analysis_interval >= 10:
                    self.scanner.analysis_interval = analysis_interval
                    self.database.set_alert_preference("analysis_interval", str(analysis_interval))
                    changes.append(f"Analysis interval: {analysis_interval}s")

                if changes:
                    return "Alert preferences updated:\n" + "\n".join(f"  - {c}" for c in changes)
                else:
                    return "No valid preference changes provided."
            except Exception as e:
                logging.error(f"Error in set_alert_preferences: {str(e)}")
                return f"Error updating preferences: {str(e)}"

    def run(self):
        """Run the MCP server"""
        try:
            self.mcp.run()
        except KeyboardInterrupt:
            logging.info("Shutting down MCP server...")
            self.cleanup()
        except Exception as e:
            logging.error(f"Error running MCP server: {str(e)}")
            self.cleanup()
            raise
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'streamer') and self.streamer:
                self.streamer.stop()

            if self.scanner:
                self.scanner.stop_scanner()

            if self.notification_manager:
                self.notification_manager.add_notification(
                    self.notification_manager.create_system_notification(
                        "MCP Trading Server shutting down", "MEDIUM"
                    )
                )

            logging.info("Cleanup completed")

        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    try:
        server = UpstoxMCPServer()
        server.run()
        
    except Exception as e:
        logging.error(f"Failed to start MCP server: {str(e)}")
        sys.exit(1)