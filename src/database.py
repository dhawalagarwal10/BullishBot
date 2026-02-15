import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
import os

class TradingDatabase:
    def __init__(self, db_path: str = "data/trades.db"):
        self.db_path = db_path
        self.setup_database()
        
    def setup_database(self):
        """Create database tables if they don't exist"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS orders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        order_id TEXT UNIQUE NOT NULL,
                        symbol TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        action TEXT NOT NULL,
                        order_type TEXT NOT NULL,
                        price REAL,
                        status TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        broker_response TEXT
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS opportunities (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        alert_type TEXT NOT NULL,
                        price REAL NOT NULL,
                        volume INTEGER,
                        change_percent REAL,
                        message TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        acted_upon BOOLEAN DEFAULT FALSE
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        total_value REAL,
                        day_pnl REAL,
                        holdings TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS stock_data (
                        symbol TEXT PRIMARY KEY,
                        current_price REAL,
                        volume INTEGER,
                        change_percent REAL,
                        high REAL,
                        low REAL,
                        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS technical_indicators (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        current_price REAL,
                        rsi_14 REAL,
                        macd_line REAL,
                        macd_signal REAL,
                        macd_histogram REAL,
                        macd_crossover TEXT,
                        sma_20 REAL,
                        sma_50 REAL,
                        sma_200 REAL,
                        bb_upper REAL,
                        bb_middle REAL,
                        bb_lower REAL,
                        volume_ratio REAL,
                        support_level REAL,
                        resistance_level REAL,
                        trend TEXT,
                        trend_strength REAL
                    )
                ''')

                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_ti_symbol_time
                    ON technical_indicators(symbol, timestamp DESC)
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        reasons TEXT,
                        technical_summary TEXT,
                        news_context TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        acknowledged BOOLEAN DEFAULT FALSE
                    )
                ''')

                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_signals_symbol_time
                    ON signals(symbol, timestamp DESC)
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alert_preferences (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        setting_key TEXT UNIQUE NOT NULL,
                        setting_value TEXT NOT NULL,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS radar_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        trigger_type TEXT NOT NULL,
                        price REAL,
                        rsi REAL,
                        volume_ratio REAL,
                        detail TEXT,
                        scan_time DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_radar_scan_time
                    ON radar_results(scan_time DESC)
                ''')

                conn.commit()
                logging.info("Database setup completed successfully")
                
        except Exception as e:
            logging.error(f"Database setup failed: {str(e)}")
            raise
    
    def save_order(self, order_data: Dict) -> bool:
        """Save order to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO orders 
                    (order_id, symbol, quantity, action, order_type, price, status, broker_response)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    order_data.get('order_id'),
                    order_data.get('symbol'),
                    order_data.get('quantity'),
                    order_data.get('action'),
                    order_data.get('order_type'),
                    order_data.get('price'),
                    order_data.get('status'),
                    json.dumps(order_data.get('broker_response', {}))
                ))
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"Failed to save order: {str(e)}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get order status from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM orders WHERE order_id = ?
                ''', (order_id,))
                row = cursor.fetchone()
                
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, row))
                return None
        except Exception as e:
            logging.error(f"Failed to get order status: {str(e)}")
            return None
    
    def save_opportunity(self, opportunity_data: Dict) -> bool:
        """Save opportunity alert to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO opportunities 
                    (symbol, alert_type, price, volume, change_percent, message)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    opportunity_data.get('symbol'),
                    opportunity_data.get('alert_type'),
                    opportunity_data.get('price'),
                    opportunity_data.get('volume'),
                    opportunity_data.get('change_percent'),
                    opportunity_data.get('message')
                ))
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"Failed to save opportunity: {str(e)}")
            return False
    
    def get_recent_opportunities(self, limit: int = 10) -> List[Dict]:
        """Get recent opportunities"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM opportunities 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                rows = cursor.fetchall()
                
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logging.error(f"Failed to get opportunities: {str(e)}")
            return []
    
    def update_stock_data(self, symbol: str, data: Dict) -> bool:
        """Update stock data cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO stock_data 
                    (symbol, current_price, volume, change_percent, high, low, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    data.get('current_price'),
                    data.get('volume'),
                    data.get('change_percent'),
                    data.get('high'),
                    data.get('low'),
                    datetime.now()
                ))
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"Failed to update stock data: {str(e)}")
            return False
    
    def get_stock_data(self, symbol: str) -> Optional[Dict]:
        """Get cached stock data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM stock_data WHERE symbol = ?
                ''', (symbol,))
                row = cursor.fetchone()
                
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, row))
                return None
        except Exception as e:
            logging.error(f"Failed to get stock data: {str(e)}")
            return None
    
    def save_technical_indicators(self, snapshot: Dict) -> bool:
        """Save a technical analysis snapshot to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO technical_indicators
                    (symbol, current_price, rsi_14, macd_line, macd_signal, macd_histogram,
                     macd_crossover, sma_20, sma_50, sma_200, bb_upper, bb_middle, bb_lower,
                     volume_ratio, support_level, resistance_level, trend, trend_strength)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    snapshot.get('symbol'),
                    snapshot.get('current_price'),
                    snapshot.get('rsi_14'),
                    snapshot.get('macd_line'),
                    snapshot.get('macd_signal'),
                    snapshot.get('macd_histogram'),
                    snapshot.get('macd_crossover'),
                    snapshot.get('sma_20'),
                    snapshot.get('sma_50'),
                    snapshot.get('sma_200'),
                    snapshot.get('bb_upper'),
                    snapshot.get('bb_middle'),
                    snapshot.get('bb_lower'),
                    snapshot.get('volume_ratio'),
                    snapshot.get('support_level'),
                    snapshot.get('resistance_level'),
                    snapshot.get('trend'),
                    snapshot.get('trend_strength')
                ))
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"Failed to save technical indicators: {str(e)}")
            return False

    def get_latest_indicators(self, symbol: str) -> Optional[Dict]:
        """Get the most recent technical indicators for a symbol"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM technical_indicators
                    WHERE symbol = ?
                    ORDER BY timestamp DESC LIMIT 1
                ''', (symbol,))
                row = cursor.fetchone()

                if row:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, row))
                return None
        except Exception as e:
            logging.error(f"Failed to get latest indicators: {str(e)}")
            return None

    def save_signal(self, signal_data: Dict) -> bool:
        """Save a trading signal to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO signals
                    (symbol, signal_type, confidence, reasons, technical_summary, news_context)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    signal_data.get('symbol'),
                    signal_data.get('signal_type'),
                    signal_data.get('confidence'),
                    signal_data.get('reasons', '[]'),
                    signal_data.get('technical_summary', '{}'),
                    signal_data.get('news_context', '')
                ))
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"Failed to save signal: {str(e)}")
            return False

    def get_recent_signals(self, limit: int = 10, symbol: str = None) -> List[Dict]:
        """Get recent trading signals, optionally filtered by symbol"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if symbol:
                    cursor.execute('''
                        SELECT * FROM signals
                        WHERE symbol = ?
                        ORDER BY timestamp DESC LIMIT ?
                    ''', (symbol, limit))
                else:
                    cursor.execute('''
                        SELECT * FROM signals
                        ORDER BY timestamp DESC LIMIT ?
                    ''', (limit,))
                rows = cursor.fetchall()

                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logging.error(f"Failed to get recent signals: {str(e)}")
            return []

    def get_alert_preference(self, key: str, default: str = None) -> Optional[str]:
        """Get an alert preference value"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT setting_value FROM alert_preferences
                    WHERE setting_key = ?
                ''', (key,))
                row = cursor.fetchone()
                return row[0] if row else default
        except Exception as e:
            logging.error(f"Failed to get alert preference: {str(e)}")
            return default

    def set_alert_preference(self, key: str, value: str) -> bool:
        """Set an alert preference value"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO alert_preferences
                    (setting_key, setting_value, updated_at)
                    VALUES (?, ?, ?)
                ''', (key, value, datetime.now()))
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"Failed to set alert preference: {str(e)}")
            return False

    def save_radar_result(self, result: Dict) -> bool:
        """Save a market radar scan result"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO radar_results
                    (symbol, trigger_type, price, rsi, volume_ratio, detail)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    result.get('symbol'),
                    result.get('trigger_type'),
                    result.get('price'),
                    result.get('rsi'),
                    result.get('volume_ratio'),
                    result.get('detail', '')
                ))
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"Failed to save radar result: {str(e)}")
            return False

    def get_recent_radar_results(self, limit: int = 50, trigger_type: str = None) -> List[Dict]:
        """Get recent market radar results, optionally filtered by trigger type"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if trigger_type:
                    cursor.execute('''
                        SELECT * FROM radar_results
                        WHERE trigger_type = ?
                        ORDER BY scan_time DESC LIMIT ?
                    ''', (trigger_type, limit))
                else:
                    cursor.execute('''
                        SELECT * FROM radar_results
                        ORDER BY scan_time DESC LIMIT ?
                    ''', (limit,))
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logging.error(f"Failed to get radar results: {str(e)}")
            return []

    def cleanup_old_data(self, days: int = 30) -> bool:
        """Clean up old data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM opportunities 
                    WHERE timestamp < datetime('now', '-{} days')
                '''.format(days))
                
                cursor.execute('''
                    DELETE FROM portfolio_snapshots
                    WHERE timestamp < datetime('now', '-{} days')
                '''.format(days))

                cursor.execute('''
                    DELETE FROM technical_indicators
                    WHERE timestamp < datetime('now', '-7 days')
                ''')

                cursor.execute('''
                    DELETE FROM signals
                    WHERE timestamp < datetime('now', '-{} days')
                '''.format(days))

                cursor.execute('''
                    DELETE FROM radar_results
                    WHERE scan_time < datetime('now', '-3 days')
                ''')

                conn.commit()
                return True
        except Exception as e:
            logging.error(f"Failed to cleanup old data: {str(e)}")
            return False

if __name__ == "__main__":
    db = TradingDatabase()
    print("Database initialized successfully!")