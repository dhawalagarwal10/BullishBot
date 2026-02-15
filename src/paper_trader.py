import logging
import json
import uuid
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional


class PaperTrader:
    """Local paper trading simulator that uses real market data with virtual cash.

    Mirrors the UpstoxClient interface (place_order, get_portfolio, get_balance,
    get_order_status, cancel_order) so the MCP server can swap between real and
    paper trading transparently.

    All state is persisted to SQLite so it survives restarts.
    """

    DEFAULT_STARTING_CASH = 1_000_000.0  # ₹10,00,000

    def __init__(self, db_path: str = "data/trades.db", data_buffer=None, upstox_client=None):
        self.db_path = db_path
        self.data_buffer = data_buffer
        self.upstox_client = upstox_client  # For price lookups when DataBuffer has no data
        self._setup_tables()
        self._ensure_account()

    # ------------------------------------------------------------------ #
    #  Database setup
    # ------------------------------------------------------------------ #

    def _setup_tables(self):
        """Create paper trading tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_account (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    cash_balance REAL NOT NULL,
                    starting_balance REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_holdings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    quantity INTEGER NOT NULL DEFAULT 0,
                    average_price REAL NOT NULL DEFAULT 0,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    requested_price REAL,
                    fill_price REAL,
                    status TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    filled_at DATETIME
                )
            ''')

            conn.commit()

    def _ensure_account(self):
        """Create the paper account row if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM paper_account WHERE id = 1")
            if cursor.fetchone() is None:
                cursor.execute(
                    "INSERT INTO paper_account (id, cash_balance, starting_balance) VALUES (1, ?, ?)",
                    (self.DEFAULT_STARTING_CASH, self.DEFAULT_STARTING_CASH)
                )
                conn.commit()
                logging.info(f"Paper trading account created with ₹{self.DEFAULT_STARTING_CASH:,.2f}")

    # ------------------------------------------------------------------ #
    #  Price lookup
    # ------------------------------------------------------------------ #

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get the current market price for a symbol.

        Tries DataBuffer first (live ticks), then falls back to the Upstox
        REST quote API.
        """
        # Try DataBuffer (real-time)
        if self.data_buffer and self.data_buffer.has_data(symbol):
            closes = self.data_buffer.get_daily_series(symbol, "close", 1)
            if closes is not None and len(closes) > 0:
                price = float(closes.iloc[-1])
                if price > 0:
                    return price

        # Fallback to Upstox REST API
        if self.upstox_client:
            try:
                quote = self.upstox_client.get_quote(symbol)
                if quote and quote.get("current_price", 0) > 0:
                    return float(quote["current_price"])
            except Exception as e:
                logging.debug(f"Paper trader price lookup failed for {symbol}: {e}")

        return None

    # ------------------------------------------------------------------ #
    #  Public interface (mirrors UpstoxClient)
    # ------------------------------------------------------------------ #

    def place_order(self, symbol: str, quantity: int, action: str,
                    order_type: str = "MARKET", price: float = 0,
                    exchange: str = "NSE") -> Optional[Dict]:
        """Simulate placing an order. Market orders fill immediately at live price."""
        try:
            action = action.upper()
            order_type = order_type.upper()
            symbol = symbol.upper()

            if action not in ("BUY", "SELL"):
                logging.error("Paper order: action must be BUY or SELL")
                return None

            if quantity <= 0:
                logging.error("Paper order: quantity must be positive")
                return None

            # Determine fill price
            if order_type == "MARKET":
                fill_price = self._get_current_price(symbol)
                if fill_price is None:
                    logging.error(f"Paper order: unable to get market price for {symbol}")
                    return None
            else:
                # LIMIT order: use requested price, but still need market price for validation
                if price <= 0:
                    logging.error("Paper order: price required for LIMIT orders")
                    return None
                fill_price = price

            order_id = f"PAPER-{uuid.uuid4().hex[:12].upper()}"
            total_cost = fill_price * quantity

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if action == "BUY":
                    # Check sufficient funds
                    cursor.execute("SELECT cash_balance FROM paper_account WHERE id = 1")
                    cash = cursor.fetchone()[0]
                    if cash < total_cost:
                        return {
                            "order_id": order_id,
                            "symbol": symbol,
                            "quantity": quantity,
                            "action": action,
                            "order_type": order_type,
                            "price": fill_price,
                            "status": "REJECTED",
                            "broker_response": {"error": f"Insufficient funds. Available: ₹{cash:,.2f}, Required: ₹{total_cost:,.2f}"}
                        }

                    # Deduct cash
                    cursor.execute(
                        "UPDATE paper_account SET cash_balance = cash_balance - ? WHERE id = 1",
                        (total_cost,)
                    )

                    # Update holdings
                    cursor.execute("SELECT quantity, average_price FROM paper_holdings WHERE symbol = ?", (symbol,))
                    row = cursor.fetchone()
                    if row:
                        old_qty, old_avg = row
                        new_qty = old_qty + quantity
                        new_avg = ((old_avg * old_qty) + (fill_price * quantity)) / new_qty
                        cursor.execute(
                            "UPDATE paper_holdings SET quantity = ?, average_price = ?, updated_at = ? WHERE symbol = ?",
                            (new_qty, new_avg, datetime.now(), symbol)
                        )
                    else:
                        cursor.execute(
                            "INSERT INTO paper_holdings (symbol, quantity, average_price, updated_at) VALUES (?, ?, ?, ?)",
                            (symbol, quantity, fill_price, datetime.now())
                        )

                elif action == "SELL":
                    # Check sufficient holdings
                    cursor.execute("SELECT quantity, average_price FROM paper_holdings WHERE symbol = ?", (symbol,))
                    row = cursor.fetchone()
                    if not row or row[0] < quantity:
                        available = row[0] if row else 0
                        return {
                            "order_id": order_id,
                            "symbol": symbol,
                            "quantity": quantity,
                            "action": action,
                            "order_type": order_type,
                            "price": fill_price,
                            "status": "REJECTED",
                            "broker_response": {"error": f"Insufficient holdings. Available: {available}, Requested: {quantity}"}
                        }

                    # Add cash from sale
                    cursor.execute(
                        "UPDATE paper_account SET cash_balance = cash_balance + ? WHERE id = 1",
                        (total_cost,)
                    )

                    # Update holdings
                    new_qty = row[0] - quantity
                    if new_qty == 0:
                        cursor.execute("DELETE FROM paper_holdings WHERE symbol = ?", (symbol,))
                    else:
                        cursor.execute(
                            "UPDATE paper_holdings SET quantity = ?, updated_at = ? WHERE symbol = ?",
                            (new_qty, datetime.now(), symbol)
                        )

                # Record the order
                cursor.execute('''
                    INSERT INTO paper_orders
                    (order_id, symbol, quantity, action, order_type, requested_price, fill_price, status, filled_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (order_id, symbol, quantity, action, order_type, price, fill_price, "COMPLETE", datetime.now()))

                conn.commit()

            logging.info(f"Paper order filled: {action} {quantity} {symbol} @ ₹{fill_price:.2f}")

            return {
                "order_id": order_id,
                "symbol": symbol,
                "quantity": quantity,
                "action": action,
                "order_type": order_type,
                "price": fill_price,
                "status": "COMPLETE",
                "broker_response": {"message": "Paper trade executed successfully", "fill_price": fill_price}
            }

        except Exception as e:
            logging.error(f"Paper order error: {e}")
            return None

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get status of a paper order."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM paper_orders WHERE order_id = ?", (order_id,))
                row = cursor.fetchone()
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    order = dict(zip(columns, row))
                    return {
                        "order_id": order["order_id"],
                        "status": order["status"],
                        "symbol": order["symbol"],
                        "quantity": order["quantity"],
                        "filled_quantity": order["quantity"] if order["status"] == "COMPLETE" else 0,
                        "price": order["requested_price"] or 0,
                        "average_price": order["fill_price"] or 0,
                        "timestamp": order["filled_at"] or order["created_at"],
                    }
                return None
        except Exception as e:
            logging.error(f"Paper get_order_status error: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a paper order. Since paper orders fill instantly, this always returns False."""
        logging.info(f"Paper cancel_order: {order_id} - paper orders fill instantly and cannot be cancelled")
        return False

    def get_portfolio(self) -> Optional[Dict]:
        """Get current paper portfolio holdings with live P&L."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT symbol, quantity, average_price FROM paper_holdings WHERE quantity > 0")
                rows = cursor.fetchall()

            if not rows:
                return {
                    "total_value": 0,
                    "total_investment": 0,
                    "total_pnl": 0,
                    "total_pnl_percent": 0,
                    "holdings": []
                }

            holdings = []
            total_value = 0
            total_investment = 0

            for symbol, quantity, avg_price in rows:
                current_price = self._get_current_price(symbol)
                if current_price is None:
                    current_price = avg_price  # Use avg if no live price

                current_value = current_price * quantity
                investment_value = avg_price * quantity
                pnl = current_value - investment_value
                pnl_pct = (pnl / investment_value * 100) if investment_value > 0 else 0

                holdings.append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "average_price": avg_price,
                    "current_price": current_price,
                    "current_value": current_value,
                    "pnl": pnl,
                    "pnl_percent": pnl_pct,
                })

                total_value += current_value
                total_investment += investment_value

            total_pnl = total_value - total_investment
            total_pnl_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0

            return {
                "total_value": total_value,
                "total_investment": total_investment,
                "total_pnl": total_pnl,
                "total_pnl_percent": total_pnl_pct,
                "holdings": holdings,
            }

        except Exception as e:
            logging.error(f"Paper get_portfolio error: {e}")
            return None

    def get_balance(self) -> Optional[Dict]:
        """Get paper account balance and unrealized P&L."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT cash_balance, starting_balance FROM paper_account WHERE id = 1")
                row = cursor.fetchone()
                if not row:
                    return None
                cash, starting = row

            portfolio = self.get_portfolio()
            holdings_value = portfolio["total_value"] if portfolio else 0
            unrealized_pnl = portfolio["total_pnl"] if portfolio else 0
            total_value = cash + holdings_value
            realized_pnl = total_value - starting - unrealized_pnl

            return {
                "available_margin": cash,
                "used_margin": holdings_value,
                "total_margin": total_value,
                "unrealized_pnl": unrealized_pnl,
                "realized_pnl": realized_pnl,
            }

        except Exception as e:
            logging.error(f"Paper get_balance error: {e}")
            return None

    def get_trade_history(self, limit: int = 20) -> List[Dict]:
        """Get recent paper trade history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT order_id, symbol, quantity, action, order_type,
                           fill_price, status, created_at
                    FROM paper_orders
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (limit,))
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logging.error(f"Paper get_trade_history error: {e}")
            return []

    def reset_account(self, starting_cash: float = None) -> bool:
        """Reset the paper trading account to starting state."""
        try:
            cash = starting_cash or self.DEFAULT_STARTING_CASH
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM paper_holdings")
                cursor.execute("DELETE FROM paper_orders")
                cursor.execute(
                    "UPDATE paper_account SET cash_balance = ?, starting_balance = ?, created_at = ? WHERE id = 1",
                    (cash, cash, datetime.now())
                )
                conn.commit()
            logging.info(f"Paper trading account reset with ₹{cash:,.2f}")
            return True
        except Exception as e:
            logging.error(f"Paper reset_account error: {e}")
            return False
