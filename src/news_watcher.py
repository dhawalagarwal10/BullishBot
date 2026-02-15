"""Enhanced news monitoring service.

Replaces news_analyzer.py with continuous monitoring, urgency detection,
and immediate Telegram alerts for critical news affecting held stocks.
Preserves the get_trend_explanation() interface for backward compat.
"""

import logging
import time
import threading
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Set
from collections import OrderedDict

import requests


# Keywords that indicate urgent/market-moving news
URGENT_KEYWORDS = [
    "crash", "fraud", "sebi", "earnings", "results", "dividend", "split",
    "buyback", "downgrade", "upgrade", "merger", "acquisition", "ban",
    "default", "scam", "bankruptcy", "delisting", "halt", "circuit",
    "warning", "probe", "investigation", "penalty", "loss", "profit",
    "guidance", "outlook", "rating", "target", "block deal", "stake",
]


class NewsWatcher:
    """Continuous news monitoring service with urgency detection.

    Polls Google News RSS for:
    - Each held stock (from portfolio)
    - Each watchlist stock
    - Broad market terms (NIFTY, Sensex, RBI)

    Urgent news for HELD stocks -> immediate URGENT alert via notification_manager.
    Normal news -> batched for hourly digest.

    Also provides get_trend_explanation(symbol) for the scanner (backward compat).
    """

    CACHE_TTL = 600  # 10 min cache for trend explanations
    MAX_CACHE_SIZE = 100
    POLL_INTERVAL = 600  # 10 minutes between full poll cycles
    RATE_LIMIT_DELAY = 2.0  # seconds between RSS requests

    def __init__(self, notification_manager=None, telegram_callback=None):
        self.notification_manager = notification_manager
        self._telegram_callback = telegram_callback  # async callable(text) to send to Telegram
        self._cache: OrderedDict[str, Dict] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._seen_headlines: Set[str] = set()  # dedup within session
        self._last_search_time = datetime.min
        self._holdings: List[str] = []
        self._watchlist: List[str] = []
        self._running = False
        self._thread = None
        self._pending_news: List[Dict] = []  # for hourly digest
        self._last_digest_time = datetime.now()
        self._pending_telegram: List[str] = []  # messages to send via Telegram

    def start(self, holdings: List[str] = None, watchlist: List[str] = None):
        """Start continuous news monitoring in a background thread."""
        self._holdings = [s.upper() for s in (holdings or [])]
        self._watchlist = [s.upper() for s in (watchlist or [])]
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logging.info(f"NewsWatcher started: {len(self._holdings)} holdings, {len(self._watchlist)} watchlist")

    def stop(self):
        """Stop the news monitoring thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logging.info("NewsWatcher stopped")

    def update_holdings(self, holdings: List[str]):
        """Update the list of held stocks (called when portfolio changes)."""
        self._holdings = [s.upper() for s in holdings]

    def update_watchlist(self, watchlist: List[str]):
        """Update the watchlist."""
        self._watchlist = [s.upper() for s in watchlist]

    def get_pending_telegram_messages(self) -> List[str]:
        """Get and clear pending Telegram messages (called by start_bot)."""
        msgs = self._pending_telegram.copy()
        self._pending_telegram.clear()
        return msgs

    # ---- Backward-compat interface (used by scanner) ----

    def get_trend_explanation(self, symbol: str) -> str:
        """Get recent news headlines for a stock. Cached, rate-limited."""
        cached = self._get_cached(symbol)
        if cached:
            return cached

        self._wait_for_rate_limit()

        try:
            news_items = self._search_news(symbol)
            if news_items:
                explanation = self._format_explanation(symbol, news_items)
            else:
                explanation = f"No recent news found for {symbol}. Signal is based on technical analysis only."

            self._set_cached(symbol, explanation)
            return explanation
        except Exception as e:
            logging.error(f"Error fetching news for {symbol}: {e}")
            return "Unable to fetch news context."

    def get_news(self, symbol: str, count: int = 5) -> List[Dict]:
        """Get recent news items for a symbol (for /news command)."""
        self._wait_for_rate_limit()
        try:
            return self._search_news(symbol, max_items=count)
        except Exception as e:
            logging.error(f"Error fetching news for {symbol}: {e}")
            return []

    # ---- Internal polling loop ----

    def _poll_loop(self):
        """Main loop: poll news for all monitored symbols."""
        while self._running:
            try:
                # Combine holdings + watchlist + market terms
                symbols = list(set(self._holdings + self._watchlist))
                market_terms = ["NIFTY Sensex India market", "RBI monetary policy India"]

                # Poll each symbol
                for symbol in symbols:
                    if not self._running:
                        break
                    try:
                        self._check_symbol_news(symbol)
                        time.sleep(self.RATE_LIMIT_DELAY)
                    except Exception as e:
                        logging.debug(f"News poll error for {symbol}: {e}")

                # Poll market-wide
                for term in market_terms:
                    if not self._running:
                        break
                    try:
                        self._check_market_news(term)
                        time.sleep(self.RATE_LIMIT_DELAY)
                    except Exception as e:
                        logging.debug(f"Market news poll error: {e}")

                # Send hourly digest if enough time has passed
                if (datetime.now() - self._last_digest_time).total_seconds() > 3600:
                    self._send_digest()

                # Sleep until next poll cycle
                for _ in range(int(self.POLL_INTERVAL)):
                    if not self._running:
                        break
                    time.sleep(1)

            except Exception as e:
                logging.error(f"NewsWatcher poll loop error: {e}")
                time.sleep(60)

    def _check_symbol_news(self, symbol: str):
        """Check news for a single symbol and handle urgency."""
        items = self._search_news(symbol, max_items=5)
        is_held = symbol in self._holdings

        for item in items:
            headline = item.get("title", "")
            # Dedup
            if headline in self._seen_headlines:
                continue
            self._seen_headlines.add(headline)

            # Check urgency
            urgent = self._is_urgent(headline)

            if urgent and is_held and self.notification_manager:
                # Immediate alert for urgent news on held stocks
                notif = self.notification_manager.create_news_notification({
                    "symbol": symbol,
                    "headline": headline,
                    "source": item.get("source", ""),
                    "urgent": True,
                })
                self.notification_manager.add_notification(notif)
                logging.info(f"URGENT news alert for held stock {symbol}: {headline[:80]}")
                # Queue for Telegram
                source = item.get("source", "")
                msg = f"<b>[URGENT NEWS] {symbol}</b>\n{headline}"
                if source:
                    msg += f"\n<i>Source: {source}</i>"
                self._pending_telegram.append(msg)
            else:
                # Queue for digest
                self._pending_news.append({
                    "symbol": symbol,
                    "headline": headline,
                    "source": item.get("source", ""),
                    "urgent": urgent,
                    "time": datetime.now().isoformat(),
                })

        # Update cache for trend explanation
        if items:
            explanation = self._format_explanation(symbol, items)
            self._set_cached(symbol, explanation)

    def _check_market_news(self, query: str):
        """Check broad market news."""
        items = self._fetch_google_news_rss(query, max_items=3)
        for item in items:
            headline = item.get("title", "")
            if headline in self._seen_headlines:
                continue
            self._seen_headlines.add(headline)

            urgent = self._is_urgent(headline)
            if urgent and self.notification_manager:
                notif = self.notification_manager.create_news_notification({
                    "symbol": "MARKET",
                    "headline": headline,
                    "source": item.get("source", ""),
                    "urgent": True,
                })
                self.notification_manager.add_notification(notif)

    def _send_digest(self):
        """Send an hourly digest of non-urgent news."""
        self._last_digest_time = datetime.now()
        if not self._pending_news or not self.notification_manager:
            self._pending_news.clear()
            return

        # Group by symbol, take top 2 per symbol
        by_symbol: Dict[str, List] = {}
        for item in self._pending_news:
            sym = item["symbol"]
            if sym not in by_symbol:
                by_symbol[sym] = []
            if len(by_symbol[sym]) < 2:
                by_symbol[sym].append(item)

        if not by_symbol:
            self._pending_news.clear()
            return

        digest_lines = ["Hourly News Digest:"]
        for sym, items in by_symbol.items():
            digest_lines.append(f"\n{sym}:")
            for item in items:
                digest_lines.append(f"  - {item['headline'][:80]}")

        notif = self.notification_manager.create_news_notification({
            "symbol": "DIGEST",
            "headline": f"News digest: {len(by_symbol)} stocks",
            "source": "",
            "urgent": False,
        })
        notif["message"] = "\n".join(digest_lines)
        self.notification_manager.add_notification(notif)

        self._pending_news.clear()
        # Trim seen headlines to prevent unbounded growth
        if len(self._seen_headlines) > 1000:
            self._seen_headlines = set(list(self._seen_headlines)[-500:])

    # ---- Urgency detection ----

    @staticmethod
    def _is_urgent(headline: str) -> bool:
        """Check if a headline contains urgent/market-moving keywords."""
        lower = headline.lower()
        return any(kw in lower for kw in URGENT_KEYWORDS)

    # ---- News fetching (same as original news_analyzer) ----

    def _search_news(self, symbol: str, max_items: int = 5) -> List[Dict]:
        """Search for recent news about a stock via Google News RSS."""
        queries = [
            f"{symbol} stock NSE India",
            f"{symbol} share price India",
        ]
        for query in queries:
            try:
                items = self._fetch_google_news_rss(query, max_items)
                if items:
                    return items
            except Exception as e:
                logging.debug(f"News search failed for '{query}': {e}")
        return []

    def _fetch_google_news_rss(self, query: str, max_items: int = 5) -> List[Dict]:
        """Fetch news from Google News RSS feed."""
        url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"

        response = requests.get(
            url,
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        if response.status_code != 200:
            return []

        root = ET.fromstring(response.content)
        items = []
        for item in root.findall(".//item")[:max_items]:
            title_el = item.find("title")
            pub_date_el = item.find("pubDate")
            source_el = item.find("source")

            if title_el is not None and title_el.text:
                items.append({
                    "title": title_el.text.strip(),
                    "date": pub_date_el.text.strip() if pub_date_el is not None and pub_date_el.text else "",
                    "source": source_el.text.strip() if source_el is not None and source_el.text else "",
                })
        return items

    def _format_explanation(self, symbol: str, news_items: List[Dict]) -> str:
        """Format news items into a concise explanation string."""
        if not news_items:
            return ""
        headlines = []
        for item in news_items[:3]:
            title = item["title"]
            source = item.get("source", "")
            if source:
                headlines.append(f"- {title} ({source})")
            else:
                headlines.append(f"- {title}")
        return "Recent news:\n" + "\n".join(headlines)

    # ---- Caching ----

    def _get_cached(self, symbol: str) -> Optional[str]:
        with self._cache_lock:
            if symbol in self._cache:
                entry = self._cache[symbol]
                if (datetime.now() - entry["time"]).total_seconds() < self.CACHE_TTL:
                    return entry["text"]
                else:
                    del self._cache[symbol]
        return None

    def _set_cached(self, symbol: str, text: str):
        with self._cache_lock:
            self._cache[symbol] = {"text": text, "time": datetime.now()}
            while len(self._cache) > self.MAX_CACHE_SIZE:
                self._cache.popitem(last=False)

    def _wait_for_rate_limit(self):
        now = datetime.now()
        elapsed = (now - self._last_search_time).total_seconds()
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_search_time = datetime.now()
