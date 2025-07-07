import logging
import json
import os
import sys
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server
from mcp.types import Resource, Tool
from upstox_client import UpstoxClient
from database import TradingDatabase
from opportunity_scanner import OpportunityScanner
from notification_manager import NotificationManager

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
        self.setup_components()
        self.setup_tools()
        
        logging.info("Upstox MCP Server initialized")
    
    def setup_components(self):
        """Initialize all components"""
        try:
            self.database = TradingDatabase()
            self.upstox_client = UpstoxClient()
            self.notification_manager = NotificationManager()
            self.scanner = OpportunityScanner(self.upstox_client, self.database)
            self.scanner.start_scanner()
            self.notification_manager.add_notification(
                self.notification_manager.create_system_notification(
                    "MCP Trading Server started successfully", "MEDIUM"
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
                quote = self.upstox_client.get_quote(symbol.upper(), exchange.upper())
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
            """Place a buy or sell order"""
            try:
                if action.upper() not in ["BUY", "SELL"]:
                    return "❌ Action must be 'BUY' or 'SELL'"
                
                if quantity <= 0:
                    return "❌ Quantity must be positive"
                
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
                    self.database.save_order(order_result)
                    self.notification_manager.add_notification(
                        self.notification_manager.create_order_notification(order_result)
                    )
                    
                    return f"""✅ Order Placed Successfully!
Order ID: {order_result['order_id']}
Stock: {symbol.upper()}
Action: {action.upper()} {quantity} shares
Type: {order_type.upper()}
{f'Price: ₹{price:.2f}' if price > 0 else 'Price: Market'}
Status: {order_result['status']}"""
                else:
                    return f"❌ Order placement failed for {symbol}"
                    
            except Exception as e:
                logging.error(f"Error in place_order: {str(e)}")
                return f"❌ Error placing order: {str(e)}"
        
        @self.mcp.tool()
        def get_portfolio() -> str:
            """Get current portfolio holdings"""
            try:
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
                    
                    return f"""💼 Portfolio Summary:
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
                db_order = self.database.get_order_status(order_id)
                broker_order = self.upstox_client.get_order_status(order_id)
                
                if broker_order:
                    if db_order and db_order['status'] != broker_order['status']:
                        broker_order['broker_response'] = broker_order
                        self.database.save_order(broker_order)
                    
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
            """Check account balance and available funds"""
            try:
                balance = self.upstox_client.get_balance()
                if balance:
                    return f"""💰 Account Balance:
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
            """Get current trading opportunities detected by scanner"""
            try:
                opportunities = self.database.get_recent_opportunities(5)
                if opportunities:
                    result = "🔍 Recent Opportunities:\n\n"
                    for opp in reversed(opportunities):  
                        timestamp = opp['timestamp'][:19] if opp['timestamp'] else 'Unknown'
                        result += f"""🚀 {opp['symbol']} - {opp['alert_type'].replace('_', ' ').title()}
Price: ₹{opp['price']:.2f} | Change: {opp['change_percent']:+.1f}%
Time: {timestamp}
{opp['message']}

"""
                    return result
                else:
                    return "📊 No recent opportunities detected. Scanner is monitoring market..."
                    
            except Exception as e:
                logging.error(f"Error in check_opportunities: {str(e)}")
                return f"❌ Error checking opportunities: {str(e)}"
        
        @self.mcp.tool()
        def manage_watchlist(action: str, symbol: str = None) -> str:
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
Watchlist: {status['watchlist_size']} stocks
Recent Opportunities: {status['recent_opportunities']}
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
        def get_system_status() -> str:
            """Get overall system status"""
            try:
                scanner_status = self.scanner.get_scanner_status()
                notification_summary = self.notification_manager.get_notification_summary()
                
                test_quote = self.upstox_client.get_quote("RELIANCE")
                upstox_status = "🟢 Connected" if test_quote else "🔴 Disconnected"
                
                return f"""🖥️ System Status:
Upstox API: {upstox_status}
Scanner: {"🟢 Running" if scanner_status['running'] else "🔴 Stopped"}
Market: {"🟢 Open" if scanner_status['market_open'] else "🔴 Closed"}
Database: 🟢 Connected
Notifications: {notification_summary['unread_notifications']} unread

Watchlist: {scanner_status['watchlist_size']} stocks
Recent Opportunities: {scanner_status['recent_opportunities']}"""
                
            except Exception as e:
                logging.error(f"Error in get_system_status: {str(e)}")
                return f"❌ Error getting system status: {str(e)}"

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
    os.makedirs("logs", exist_ok=True)
    
    try:
        server = UpstoxMCPServer()
        server.run()
        
    except Exception as e:
        logging.error(f"Failed to start MCP server: {str(e)}")
        sys.exit(1)