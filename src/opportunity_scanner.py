import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone  
from typing import Dict, List, Optional
from collections import defaultdict
import statistics

from upstox_client import UpstoxClient
from database import TradingDatabase

class OpportunityScanner:
    def __init__(self, 
                 upstox_client: UpstoxClient, 
                 database: TradingDatabase,
                 config_path: str = "config/watchlist.json"):
        self.upstox_client = upstox_client
        self.database = database
        self.config = self.load_config(config_path)
        self.watchlist = self.config.get("stocks", [])
        self.alert_settings = self.config.get("alert_settings", {})
        
        self.running = False
        self.scanner_thread = None
        self.last_alerts = defaultdict(lambda: datetime.min)
        self.stock_history = defaultdict(list) 
        self.volume_history = defaultdict(list)  
        self.market_timezone = timezone(timedelta(hours=5, minutes=30))  
        self.market_start = "09:15"
        self.market_end = "15:30"
        
        self.alert_count = defaultdict(int)
        self.alert_reset_time = datetime.now()
        
        logging.info(f"Opportunity Scanner initialized with {len(self.watchlist)} stocks")
    
    def load_config(self, config_path: str) -> Dict:
        """Load scanner configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"Watchlist config not found: {config_path}")
            return {"stocks": [], "alert_settings": {}}
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in watchlist config: {config_path}")
            return {"stocks": [], "alert_settings": {}}
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            now = datetime.now(self.market_timezone)
            current_time = now.time()
 
            if now.weekday() >= 5: 
                return False

            start_time = datetime.strptime(self.market_start, "%H:%M").time()
            end_time = datetime.strptime(self.market_end, "%H:%M").time()
            
            return start_time <= current_time <= end_time
        except Exception as e:
            logging.error(f"Error checking market hours: {str(e)}")
            return False
    
    def should_alert(self, symbol: str) -> bool:
        """Check if we should send alert for this symbol"""
        now = datetime.now()

        if now - self.alert_reset_time > timedelta(hours=1):
            self.alert_count.clear()
            self.alert_reset_time = now
  
        cooldown_minutes = self.alert_settings.get("alert_cooldown", 300) / 60
        if now - self.last_alerts[symbol] < timedelta(minutes=cooldown_minutes):
            return False

        max_alerts = self.alert_settings.get("max_alerts_per_hour", 5)
        if self.alert_count[symbol] >= max_alerts:
            return False
        
        return True
    
    def record_alert(self, symbol: str):
        """Record that an alert was sent"""
        now = datetime.now()
        self.last_alerts[symbol] = now
        self.alert_count[symbol] += 1
    
    def update_stock_history(self, symbol: str, quote_data: Dict):
        """Update historical data for a stock"""
        now = datetime.now()
 
        self.stock_history[symbol].append({
            "price": quote_data.get("current_price", 0),
            "volume": quote_data.get("volume", 0),
            "timestamp": now
        })

        if len(self.stock_history[symbol]) > 100:
            self.stock_history[symbol] = self.stock_history[symbol][-100:]
        
        self.volume_history[symbol].append({
            "volume": quote_data.get("volume", 0),
            "timestamp": now
        })

        if len(self.volume_history[symbol]) > 50:
            self.volume_history[symbol] = self.volume_history[symbol][-50:]
    
    def get_average_volume(self, symbol: str, minutes: int = 60) -> float:
        """Calculate average volume for the specified time period"""
        if symbol not in self.volume_history:
            return 0
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_volumes = [
            data["volume"] for data in self.volume_history[symbol]
            if data["timestamp"] > cutoff_time
        ]
        
        if not recent_volumes:
            return 0
        
        return statistics.mean(recent_volumes)
    
    def get_price_change_percent(self, symbol: str, minutes: int = 15) -> float:
        """Calculate price change percentage over specified minutes"""
        if symbol not in self.stock_history or len(self.stock_history[symbol]) < 2:
            return 0
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_prices = [
            data["price"] for data in self.stock_history[symbol]
            if data["timestamp"] > cutoff_time
        ]
        
        if len(recent_prices) < 2:
            return 0
        
        old_price = recent_prices[0]
        current_price = recent_prices[-1]
        
        if old_price == 0:
            return 0
        
        return ((current_price - old_price) / old_price) * 100
    
    def check_volume_spike(self, symbol: str, quote_data: Dict) -> bool:
        """Check if there's a volume spike"""
        current_volume = quote_data.get("volume", 0)
        avg_volume = self.get_average_volume(symbol, 60)  
        
        if avg_volume == 0:
            return False
        
        volume_threshold = self.alert_settings.get("volume_threshold", 3.0)
        volume_ratio = current_volume / avg_volume
        
        return volume_ratio >= volume_threshold
    
    def check_price_breakout(self, symbol: str, quote_data: Dict) -> bool:
        """Check if there's a price breakout"""
        if symbol not in self.stock_history or len(self.stock_history[symbol]) < 10:
            return False
        
        current_price = quote_data.get("current_price", 0)
        
        recent_prices = [data["price"] for data in self.stock_history[symbol][-20:]]
        
        if len(recent_prices) < 10:
            return False
        
        recent_high = max(recent_prices[:-1]) 
        recent_low = min(recent_prices[:-1])

        breakout_threshold = 0.01  
        
        high_breakout = current_price > recent_high * (1 + breakout_threshold)
        low_breakout = current_price < recent_low * (1 - breakout_threshold)
        
        return high_breakout or low_breakout
    
    def check_momentum_surge(self, symbol: str, quote_data: Dict) -> bool:
        """Check if there's a momentum surge"""
        momentum_window = self.alert_settings.get("momentum_window", 15)
        price_change_threshold = self.alert_settings.get("price_change_threshold", 5.0)
        
        price_change = abs(self.get_price_change_percent(symbol, momentum_window))
        
        return price_change >= price_change_threshold
    
    def generate_alert_message(self, symbol: str, alert_type: str, quote_data: Dict) -> str:
        """Generate alert message"""
        current_price = quote_data.get("current_price", 0)
        change_percent = quote_data.get("change_percent", 0)
        volume = quote_data.get("volume", 0)
        
        base_message = f"🚀 OPPORTUNITY: {symbol} @ ₹{current_price:.2f} ({change_percent:+.1f}%)"
        
        if alert_type == "volume_spike":
            avg_volume = self.get_average_volume(symbol, 60)
            volume_ratio = volume / avg_volume if avg_volume > 0 else 0
            return f"{base_message} | Volume Spike: {volume_ratio:.1f}x average"
        
        elif alert_type == "price_breakout":
            return f"{base_message} | Price Breakout Detected"
        
        elif alert_type == "momentum_surge":
            momentum_window = self.alert_settings.get("momentum_window", 15)
            price_change = self.get_price_change_percent(symbol, momentum_window)
            return f"{base_message} | Momentum Surge: {price_change:+.1f}% in {momentum_window}min"
        
        return base_message
    
    def scan_stock(self, symbol: str) -> Optional[Dict]:
        """Scan a single stock for opportunities"""
        try:
            quote_data = self.upstox_client.get_quote(symbol)
            if not quote_data:
                return None

            self.update_stock_history(symbol, quote_data)

            self.database.update_stock_data(symbol, quote_data)

            alert_triggered = False
            alert_type = None

            if self.check_volume_spike(symbol, quote_data):
                alert_type = "volume_spike"
                alert_triggered = True

            elif self.check_price_breakout(symbol, quote_data):
                alert_type = "price_breakout"
                alert_triggered = True

            elif self.check_momentum_surge(symbol, quote_data):
                alert_type = "momentum_surge"
                alert_triggered = True

            if alert_triggered and self.should_alert(symbol):
                alert_message = self.generate_alert_message(symbol, alert_type, quote_data)
                
                opportunity_data = {
                    "symbol": symbol,
                    "alert_type": alert_type,
                    "price": quote_data.get("current_price", 0),
                    "volume": quote_data.get("volume", 0),
                    "change_percent": quote_data.get("change_percent", 0),
                    "message": alert_message
                }
                
                self.database.save_opportunity(opportunity_data)
                self.record_alert(symbol)
                
                return {
                    "symbol": symbol,
                    "alert_type": alert_type,
                    "message": alert_message,
                    "quote_data": quote_data
                }
            
            return None
            
        except Exception as e:
            logging.error(f"Error scanning {symbol}: {str(e)}")
            return None
    
    def scan_all_stocks(self) -> List[Dict]:
        """Scan all stocks in watchlist"""
        opportunities = []
        
        for symbol in self.watchlist:
            try:
                opportunity = self.scan_stock(symbol)
                if opportunity:
                    opportunities.append(opportunity)
                    logging.info(f"Opportunity detected: {opportunity['message']}")

                time.sleep(0.1)
                
            except Exception as e:
                logging.error(f"Error scanning {symbol}: {str(e)}")
                continue
        
        return opportunities
    
    def scanner_loop(self):
        """Main scanner loop"""
        logging.info("Opportunity scanner started")
        
        while self.running:
            try:
                if not self.is_market_open():
                    logging.info("Market closed, scanner paused")
                    time.sleep(300)  
                    continue

                opportunities = self.scan_all_stocks()
                
                if opportunities:
                    logging.info(f"Found {len(opportunities)} opportunities")
                else:
                    logging.debug("No opportunities found in this scan")
                
                time.sleep(30)  
                
            except Exception as e:
                logging.error(f"Error in scanner loop: {str(e)}")
                time.sleep(60) 
    
    def start_scanner(self):
        """Start the background scanner"""
        if self.running:
            logging.warning("Scanner is already running")
            return
        
        self.running = True
        self.scanner_thread = threading.Thread(target=self.scanner_loop, daemon=True)
        self.scanner_thread.start()
        logging.info("Background scanner started")
    
    def stop_scanner(self):
        """Stop the background scanner"""
        if not self.running:
            logging.warning("Scanner is not running")
            return
        
        self.running = False
        if self.scanner_thread:
            self.scanner_thread.join(timeout=5)
        logging.info("Background scanner stopped")
    
    def get_scanner_status(self) -> Dict:
        """Get current scanner status"""
        return {
            "running": self.running,
            "market_open": self.is_market_open(),
            "watchlist_size": len(self.watchlist),
            "recent_opportunities": len(self.database.get_recent_opportunities(10)),
            "alert_counts": dict(self.alert_count)
        }
    
    def add_to_watchlist(self, symbol: str) -> bool:
        """Add stock to watchlist"""
        try:
            symbol = symbol.upper()
            if symbol not in self.watchlist:
                self.watchlist.append(symbol)

                self.config["stocks"] = self.watchlist
                with open("config/watchlist.json", 'w') as f:
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
        """Remove stock from watchlist"""
        try:
            symbol = symbol.upper()
            if symbol in self.watchlist:
                self.watchlist.remove(symbol)

                self.config["stocks"] = self.watchlist
                with open("config/watchlist.json", 'w') as f:
                    json.dump(self.config, f, indent=2)
                
                logging.info(f"Removed {symbol} from watchlist")
                return True
            else:
                logging.info(f"{symbol} not in watchlist")
                return False
        except Exception as e:
            logging.error(f"Error removing {symbol} from watchlist: {str(e)}")
            return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        from upstox_client import UpstoxClient
        from database import TradingDatabase

        upstox_client = UpstoxClient()
        database = TradingDatabase()
        scanner = OpportunityScanner(upstox_client, database)

        print("Testing opportunity scanner...")
        opportunities = scanner.scan_all_stocks()
        print(f"Found {len(opportunities)} opportunities")

        status = scanner.get_scanner_status()
        print(f"Scanner status: {status}")
        
    except Exception as e:
        print(f"Scanner test failed: {str(e)}")