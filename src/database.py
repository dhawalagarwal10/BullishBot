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
                
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"Failed to cleanup old data: {str(e)}")
            return False

if __name__ == "__main__":
    db = TradingDatabase()
    print("Database initialized successfully!")