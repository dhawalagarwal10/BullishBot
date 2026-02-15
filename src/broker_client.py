import json
import logging
import requests
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv


class UpstoxClient:
    def __init__(self, config_path: str = "config/upstox_config.json"):
        self.config = self.load_config(config_path)
        self.base_url = self.config.get("base_url", "https://api.upstox.com")

        # Force reload .env to get fresh token (override=True ensures we pick up changes)
        load_dotenv(override=True)

        self.api_key = os.getenv("UPSTOX_API_KEY") or self.config.get("api_key")
        self.access_token = os.getenv("UPSTOX_ACCESS_TOKEN")
        self._update_headers()

        # Lazy-loaded instrument resolver for symbol -> instrument_key mapping
        self._instrument_resolver = None

        if not self.api_key:
            raise ValueError("Upstox API key not found in config or environment")
        if not self.access_token:
            logging.warning("Access token not found. Manual login required.")

    def _update_headers(self):
        """Update authorization headers with current token."""
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

    def refresh_token(self) -> bool:
        """Reload the access token from .env file. Call this if token was updated."""
        try:
            load_dotenv(override=True)
            new_token = os.getenv("UPSTOX_ACCESS_TOKEN")
            if new_token and new_token != self.access_token:
                self.access_token = new_token
                self._update_headers()
                logging.info("Access token refreshed from .env")
                return True
            return False
        except Exception as e:
            logging.error(f"Failed to refresh token: {e}")
            return False

    def get_token_info(self) -> Dict:
        """Get info about the current token for diagnostics."""
        token = self.access_token or ""
        return {
            "token_loaded": bool(token),
            "token_length": len(token),
            "token_last_4": token[-4:] if len(token) >= 4 else "N/A",
            "api_key_loaded": bool(self.api_key),
        }

    def _get_resolver(self):
        """Lazy-load InstrumentResolver to avoid import issues."""
        if self._instrument_resolver is None:
            try:
                from instrument_resolver import InstrumentResolver
                self._instrument_resolver = InstrumentResolver()
            except Exception as e:
                logging.debug(f"InstrumentResolver not available: {e}")
        return self._instrument_resolver

    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"Config file not found: {config_path}")
            return {}
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in config file: {config_path}")
            return {}

    def get_quote(self, symbol: str, exchange: str = "NSE", instrument_key: str = None) -> Optional[Dict]:
        """Get current quote for a symbol. Auto-resolves instrument key via InstrumentResolver."""
        try:
            if not instrument_key:
                resolver = self._get_resolver()
                if resolver:
                    instrument_key = resolver.get_instrument_key(symbol.upper())
                if not instrument_key:
                    instrument_key = f"{exchange}_EQ|{symbol}"  # last-resort fallback

            url = f"{self.base_url}/v2/market-quote/quotes"
            params = {"instrument_key": instrument_key}

            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()

            data = response.json()
            if data.get("status") == "success":
                # Response keys use format like "NSE_EQ:SYMBOL", not the instrument_key
                resp_data = data.get("data", {})
                quote_data = resp_data.get(instrument_key, {})
                if not quote_data:
                    # Try first available key in response
                    for _, val in resp_data.items():
                        quote_data = val
                        break

                formatted_quote = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "current_price": quote_data.get("last_price", 0),
                    "change": quote_data.get("net_change", 0),
                    "change_percent": quote_data.get("percent_change", 0),
                    "volume": quote_data.get("volume", 0),
                    "high": quote_data.get("ohlc", {}).get("high", 0),
                    "low": quote_data.get("ohlc", {}).get("low", 0),
                    "open": quote_data.get("ohlc", {}).get("open", 0),
                    "close": quote_data.get("ohlc", {}).get("close", 0),
                    "timestamp": datetime.now().isoformat()
                }

                return formatted_quote
            else:
                logging.error(f"API error: {data.get('message', 'Unknown error')}")
                return None

        except requests.RequestException as e:
            logging.error(f"Request failed for {symbol}: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error getting quote for {symbol}: {str(e)}")
            return None

    def get_historical_candles(self, symbol: str, interval: str = "day",
                               days: int = 200, instrument_key: str = None) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV candles for any stock on demand.

        Args:
            symbol: Trading symbol (e.g. "RELIANCE")
            interval: Candle interval - "day", "1minute", "30minute", etc.
            days: Number of days of history to fetch
            instrument_key: Optional pre-resolved instrument key

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, oi
            or None on failure.
        """
        try:
            if not instrument_key:
                resolver = self._get_resolver()
                if resolver:
                    instrument_key = resolver.get_instrument_key(symbol.upper())
                if not instrument_key:
                    logging.error(f"Cannot resolve instrument key for {symbol}")
                    return None

            to_date = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=days + 100)).strftime("%Y-%m-%d")

            encoded_key = requests.utils.quote(instrument_key, safe='')
            url = f"{self.base_url}/v2/historical-candle/{encoded_key}/{interval}/{to_date}/{from_date}"

            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()

            data = response.json()
            if data.get("status") == "success":
                candles = data.get("data", {}).get("candles", [])
                if not candles:
                    logging.warning(f"No historical candles returned for {symbol}")
                    return None

                # Each candle: [timestamp, open, high, low, close, volume, oi]
                df = pd.DataFrame(
                    candles,
                    columns=["timestamp", "open", "high", "low", "close", "volume", "oi"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp").tail(days).reset_index(drop=True)
                return df
            else:
                logging.error(f"Historical API error for {symbol}: {data.get('message', '')}")
                return None

        except requests.RequestException as e:
            logging.error(f"Request failed for {symbol} historical candles: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error fetching candles for {symbol}: {e}")
            return None

    def place_order(self, symbol: str, quantity: int, action: str,
                   order_type: str = "MARKET", price: float = 0,
                   exchange: str = "NSE") -> Optional[Dict]:
        """Place an order"""
        try:
            if action.upper() not in ["BUY", "SELL"]:
                raise ValueError("Action must be 'BUY' or 'SELL'")

            if order_type.upper() not in ["MARKET", "LIMIT"]:
                raise ValueError("Order type must be 'MARKET' or 'LIMIT'")

            if order_type.upper() == "LIMIT" and price <= 0:
                raise ValueError("Price must be specified for LIMIT orders")

            order_data = {
                "instrument_token": f"{exchange}:{symbol}",
                "quantity": quantity,
                "product": "I",
                "validity": "DAY",
                "price": price if order_type.upper() == "LIMIT" else 0,
                "tag": "MCP_TRADING",
                "order_type": order_type.upper(),
                "transaction_type": action.upper()
            }

            url = f"{self.base_url}/v2/order/place"

            response = requests.post(url, headers=self.headers, json=order_data)
            response.raise_for_status()

            data = response.json()
            if data.get("status") == "success":
                order_id = data.get("data", {}).get("order_id")

                return {
                    "order_id": order_id,
                    "symbol": symbol,
                    "quantity": quantity,
                    "action": action,
                    "order_type": order_type,
                    "price": price,
                    "status": "PLACED",
                    "broker_response": data
                }
            else:
                logging.error(f"Order placement failed: {data.get('message', 'Unknown error')}")
                return None

        except requests.RequestException as e:
            logging.error(f"Request failed for order placement: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error placing order: {str(e)}")
            return None

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get order status"""
        try:
            url = f"{self.base_url}/v2/order/details"
            params = {"order_id": order_id}

            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()

            data = response.json()
            if data.get("status") == "success":
                order_data = data.get("data", {})

                return {
                    "order_id": order_id,
                    "status": order_data.get("order_status", "UNKNOWN"),
                    "symbol": order_data.get("instrument_token", "").split(":")[-1],
                    "quantity": order_data.get("quantity", 0),
                    "filled_quantity": order_data.get("filled_quantity", 0),
                    "price": order_data.get("price", 0),
                    "average_price": order_data.get("average_price", 0),
                    "timestamp": order_data.get("order_timestamp", "")
                }
            else:
                logging.error(f"Failed to get order status: {data.get('message', 'Unknown error')}")
                return None

        except requests.RequestException as e:
            logging.error(f"Request failed for order status: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error getting order status: {str(e)}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            url = f"{self.base_url}/v2/order/cancel"
            data = {"order_id": order_id}

            response = requests.delete(url, headers=self.headers, json=data)
            response.raise_for_status()

            result = response.json()
            if result.get("status") == "success":
                return True
            else:
                logging.error(f"Failed to cancel order: {result.get('message', 'Unknown error')}")
                return False

        except requests.RequestException as e:
            logging.error(f"Request failed for order cancellation: {str(e)}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error cancelling order: {str(e)}")
            return False

    def get_portfolio(self) -> Optional[Dict]:
        """Get portfolio holdings"""
        try:
            url = f"{self.base_url}/v2/portfolio/long-term-holdings"

            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            data = response.json()
            if data.get("status") == "success":
                holdings = data.get("data", [])

                total_value = 0
                total_investment = 0

                formatted_holdings = []
                for holding in holdings:
                    current_value = holding.get("last_price", 0) * holding.get("quantity", 0)
                    investment_value = holding.get("average_price", 0) * holding.get("quantity", 0)
                    pnl = current_value - investment_value

                    formatted_holdings.append({
                        "symbol": holding.get("instrument_token", "").split(":")[-1],
                        "quantity": holding.get("quantity", 0),
                        "average_price": holding.get("average_price", 0),
                        "current_price": holding.get("last_price", 0),
                        "current_value": current_value,
                        "pnl": pnl,
                        "pnl_percent": (pnl / investment_value * 100) if investment_value > 0 else 0
                    })

                    total_value += current_value
                    total_investment += investment_value

                return {
                    "total_value": total_value,
                    "total_investment": total_investment,
                    "total_pnl": total_value - total_investment,
                    "total_pnl_percent": ((total_value - total_investment) / total_investment * 100) if total_investment > 0 else 0,
                    "holdings": formatted_holdings
                }
            else:
                logging.error(f"Failed to get portfolio: {data.get('message', 'Unknown error')}")
                return None

        except requests.RequestException as e:
            logging.error(f"Request failed for portfolio: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error getting portfolio: {str(e)}")
            return None

    def get_balance(self) -> Optional[Dict]:
        """Get account balance"""
        try:
            url = f"{self.base_url}/v2/user/get-funds-and-margin"

            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            data = response.json()
            if data.get("status") == "success":
                equity = data.get("data", {}).get("equity", {})

                return {
                    "available_margin": equity.get("available_margin", 0),
                    "used_margin": equity.get("used_margin", 0),
                    "total_margin": equity.get("available_margin", 0) + equity.get("used_margin", 0),
                    "unrealized_pnl": equity.get("unrealized_pnl", 0),
                    "realized_pnl": equity.get("realized_pnl", 0)
                }
            else:
                logging.error(f"Failed to get balance: {data.get('message', 'Unknown error')}")
                return None

        except requests.RequestException as e:
            logging.error(f"Request failed for balance: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error getting balance: {str(e)}")
            return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        client = UpstoxClient()

        quote = client.get_quote("RELIANCE")
        if quote:
            print(f"Quote test successful: {quote}")
        else:
            print("Quote test failed")

    except Exception as e:
        print(f"Client initialization failed: {str(e)}")
        print("Please check your API credentials and configuration")
