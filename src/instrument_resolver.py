import json
import logging
import os
import gzip
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import requests


class InstrumentResolver:
    """Maps trading symbols (e.g. RELIANCE) to Upstox instrument keys (e.g. NSE_EQ|INE002A01018).

    The Upstox WebSocket and Historical Candle APIs require instrument keys,
    not plain trading symbols. This class resolves the mapping by:
    1. Trying to download the full Upstox instrument master file
    2. Falling back to a hardcoded config/instrument_keys.json
    """

    INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    CACHE_PATH = "config/instruments_cache.json"
    FALLBACK_PATH = "config/instrument_keys.json"
    CACHE_MAX_AGE_HOURS = 24

    def __init__(self):
        self._symbol_to_key: Dict[str, str] = {}
        self._key_to_symbol: Dict[str, str] = {}
        self._load()

    def _load(self):
        """Load instrument mappings from cache, download, or fallback."""
        if self._load_from_cache():
            logging.info(f"Loaded {len(self._symbol_to_key)} instrument mappings from cache")
            return

        if self._download_and_parse():
            logging.info(f"Downloaded {len(self._symbol_to_key)} instrument mappings from Upstox")
            return

        self._load_from_fallback()
        logging.info(f"Loaded {len(self._symbol_to_key)} instrument mappings from fallback")

    def _load_from_cache(self) -> bool:
        """Load from local cache if it exists and is fresh."""
        try:
            if not os.path.exists(self.CACHE_PATH):
                return False

            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(self.CACHE_PATH))
            if file_age > timedelta(hours=self.CACHE_MAX_AGE_HOURS):
                return False

            with open(self.CACHE_PATH, 'r') as f:
                data = json.load(f)

            self._symbol_to_key = data.get("symbol_to_key", {})
            self._key_to_symbol = data.get("key_to_symbol", {})
            return len(self._symbol_to_key) > 0

        except Exception as e:
            logging.debug(f"Cache load failed: {e}")
            return False

    def _download_and_parse(self) -> bool:
        """Download the Upstox instrument master file and build mappings."""
        try:
            response = requests.get(self.INSTRUMENTS_URL, timeout=30)
            response.raise_for_status()

            raw = gzip.decompress(response.content)
            instruments = json.loads(raw)

            for inst in instruments:
                segment = inst.get("segment", "")
                instrument_type = inst.get("instrument_type", "")
                trading_symbol = inst.get("trading_symbol", "")
                instrument_key = inst.get("instrument_key", "")

                if not trading_symbol or not instrument_key:
                    continue

                # Equity stocks
                if segment == "NSE_EQ" and instrument_type == "EQ":
                    self._symbol_to_key[trading_symbol.upper()] = instrument_key
                    self._key_to_symbol[instrument_key] = trading_symbol.upper()

                # Indices
                if segment == "NSE_INDEX":
                    name = trading_symbol.upper().replace(" ", "")
                    self._symbol_to_key[name] = instrument_key
                    self._key_to_symbol[instrument_key] = name

            self._save_cache()
            return len(self._symbol_to_key) > 0

        except Exception as e:
            logging.warning(f"Instrument download failed: {e}")
            return False

    def _load_from_fallback(self):
        """Load from the hardcoded fallback file."""
        try:
            with open(self.FALLBACK_PATH, 'r') as f:
                data = json.load(f)

            for segment, mappings in data.items():
                for symbol, instrument_key in mappings.items():
                    self._symbol_to_key[symbol.upper()] = instrument_key
                    self._key_to_symbol[instrument_key] = symbol.upper()

        except Exception as e:
            logging.error(f"Fallback instrument load failed: {e}")

    def _save_cache(self):
        """Save current mappings to cache file."""
        try:
            os.makedirs(os.path.dirname(self.CACHE_PATH), exist_ok=True)
            with open(self.CACHE_PATH, 'w') as f:
                json.dump({
                    "symbol_to_key": self._symbol_to_key,
                    "key_to_symbol": self._key_to_symbol,
                    "updated_at": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save instrument cache: {e}")

    # Common symbol aliases (user-friendly name -> actual trading symbol)
    SYMBOL_ALIASES = {
        "NIFTY50": "NIFTY",
        "NIFTY 50": "NIFTY",
        "NIFTY_50": "NIFTY",
        "BANKNIFTY": "BANKNIFTY",  # Already exists
        "SENSEX": "SENSEX",
    }

    def get_instrument_key(self, symbol: str, exchange: str = "NSE") -> Optional[str]:
        """Convert a trading symbol to its Upstox instrument key."""
        normalized = symbol.upper()
        # Check aliases first
        if normalized in self.SYMBOL_ALIASES:
            normalized = self.SYMBOL_ALIASES[normalized]
        return self._symbol_to_key.get(normalized)

    def get_symbol(self, instrument_key: str) -> Optional[str]:
        """Convert an instrument key back to its trading symbol."""
        return self._key_to_symbol.get(instrument_key)

    def get_instrument_keys_for_watchlist(self, symbols: List[str]) -> List[str]:
        """Batch convert a list of trading symbols to instrument keys."""
        keys = []
        for symbol in symbols:
            key = self.get_instrument_key(symbol)
            if key:
                keys.append(key)
            else:
                logging.warning(f"No instrument key found for {symbol}")
        return keys

    def get_all_equity_symbols(self) -> List[str]:
        """Return all known equity trading symbols (NSE_EQ segment)."""
        return [
            sym for sym, key in self._symbol_to_key.items()
            if key.startswith("NSE_EQ|")
        ]
