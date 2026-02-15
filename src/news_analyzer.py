import logging
import time
import threading
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from collections import OrderedDict

import requests


class NewsAnalyzer:
    """Fetches recent news about stocks to explain why they are trending.

    Uses Google News RSS (no API key required) to find recent headlines
    about Indian stocks. Results are cached to avoid excessive requests.
    """

    CACHE_TTL = 600  # 10 minutes
    MAX_CACHE_SIZE = 50

    def __init__(self):
        self._cache: OrderedDict[str, Dict] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._last_search_time = datetime.min
        self._min_search_interval = 2.0  # seconds between searches

    def get_trend_explanation(self, symbol: str) -> str:
        """Get a human-readable explanation of why a stock might be trending.

        Returns recent news headlines for the stock. Called from the
        SmartOpportunityScanner in a thread pool with a timeout.
        """
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

    def _search_news(self, symbol: str) -> List[Dict]:
        """Search for recent news about a stock via Google News RSS.

        Falls back to an alternative search query if the first fails.
        """
        queries = [
            f"{symbol} stock NSE India",
            f"{symbol} share price India",
        ]

        for query in queries:
            try:
                items = self._fetch_google_news_rss(query)
                if items:
                    return items
            except Exception as e:
                logging.debug(f"News search failed for query '{query}': {e}")

        return []

    def _fetch_google_news_rss(self, query: str) -> List[Dict]:
        """Fetch news from Google News RSS feed."""
        url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"

        response = requests.get(
            url,
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )

        if response.status_code != 200:
            return []

        root = ET.fromstring(response.content)
        items = []

        for item in root.findall(".//item")[:5]:
            title_el = item.find("title")
            pub_date_el = item.find("pubDate")
            source_el = item.find("source")

            if title_el is not None and title_el.text:
                news_item = {
                    "title": title_el.text.strip(),
                    "date": pub_date_el.text.strip() if pub_date_el is not None and pub_date_el.text else "",
                    "source": source_el.text.strip() if source_el is not None and source_el.text else "",
                }
                items.append(news_item)

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

    def _get_cached(self, symbol: str) -> Optional[str]:
        """Get cached news for a symbol if still fresh."""
        with self._cache_lock:
            if symbol in self._cache:
                entry = self._cache[symbol]
                if (datetime.now() - entry["time"]).total_seconds() < self.CACHE_TTL:
                    return entry["text"]
                else:
                    del self._cache[symbol]
        return None

    def _set_cached(self, symbol: str, text: str):
        """Cache news text for a symbol."""
        with self._cache_lock:
            self._cache[symbol] = {"text": text, "time": datetime.now()}
            while len(self._cache) > self.MAX_CACHE_SIZE:
                self._cache.popitem(last=False)

    def _wait_for_rate_limit(self):
        """Enforce minimum interval between news searches."""
        now = datetime.now()
        elapsed = (now - self._last_search_time).total_seconds()
        if elapsed < self._min_search_interval:
            time.sleep(self._min_search_interval - elapsed)
        self._last_search_time = datetime.now()
