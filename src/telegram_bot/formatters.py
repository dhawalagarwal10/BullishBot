import json


class TelegramFormatter:
    """Formats trading data as clean Telegram messages using HTML."""

    @staticmethod
    def format_quote(quote: dict) -> str:
        symbol = quote.get("symbol", "?")
        price = quote.get("current_price", 0)
        change = quote.get("change_percent", 0)
        volume = quote.get("volume", 0)
        high = quote.get("high", 0)
        low = quote.get("low", 0)

        arrow = "+" if change >= 0 else ""

        return (
            f"<b>{symbol}</b>\n\n"
            f"Price: Rs.{price:,.2f} ({arrow}{change:.2f}%)\n"
            f"High: Rs.{high:,.2f} | Low: Rs.{low:,.2f}\n"
            f"Volume: {volume:,}"
        )

    @staticmethod
    def format_portfolio(account, holdings, upstox_client=None) -> str:
        cash = account["cash_balance"]
        starting = account["starting_balance"]

        lines = ["<b>Portfolio [PAPER]</b>\n"]

        total_current = 0
        total_invested = 0
        holding_lines = []

        for h in holdings:
            symbol = h["symbol"]
            qty = h["quantity"]
            avg = h["average_price"]
            invested = avg * qty

            # Try to get live price
            current_price = avg
            if upstox_client:
                quote = upstox_client.get_quote(symbol)
                if quote:
                    current_price = quote.get("current_price", avg)

            current_val = current_price * qty
            pnl = current_val - invested
            pnl_pct = (pnl / invested * 100) if invested > 0 else 0

            total_current += current_val
            total_invested += invested

            arrow = "+" if pnl >= 0 else ""
            holding_lines.append(
                f"  {symbol}: {qty} shares\n"
                f"    Avg: Rs.{avg:,.2f} | Now: Rs.{current_price:,.2f}\n"
                f"    P&L: {arrow}Rs.{pnl:,.2f} ({arrow}{pnl_pct:.1f}%)"
            )

        total_pnl = total_current - total_invested
        total_value = cash + total_current
        overall_pnl = total_value - starting
        overall_pct = (overall_pnl / starting * 100) if starting > 0 else 0

        lines.append(f"Total Value: Rs.{total_value:,.2f}")
        lines.append(f"Cash: Rs.{cash:,.2f}")
        arrow = "+" if overall_pnl >= 0 else ""
        lines.append(f"Overall P&L: {arrow}Rs.{overall_pnl:,.2f} ({arrow}{overall_pct:.1f}%)\n")
        lines.append("<b>Holdings:</b>")
        lines.extend(holding_lines)

        return "\n".join(lines)

    @staticmethod
    def format_signals(rows) -> str:
        lines = ["<b>Recent Trading Signals</b>\n"]

        for row in rows:
            signal_type = row["signal_type"]
            symbol = row["symbol"]
            confidence = row["confidence"]
            timestamp = (row.get("timestamp") or "")[:16]

            reasons_raw = row.get("reasons", "[]")
            try:
                reasons = json.loads(reasons_raw) if isinstance(reasons_raw, str) else reasons_raw
            except Exception:
                reasons = []

            reasons_text = "; ".join(reasons[:2]) if reasons else "N/A"

            lines.append(
                f"{signal_type} | {symbol} | {confidence:.0f}%\n"
                f"  {reasons_text}\n"
                f"  {timestamp}\n"
            )

        return "\n".join(lines)

    @staticmethod
    def format_signal_alert(signal: dict) -> str:
        """Format a single signal for push notification."""
        signal_type = signal.get("signal_type", "?")
        symbol = signal.get("symbol", "?")
        confidence = signal.get("confidence", 0)
        reasons = signal.get("reasons", [])

        if isinstance(reasons, str):
            try:
                reasons = json.loads(reasons)
            except Exception:
                reasons = []

        header = f"<b>{signal_type} Signal: {symbol}</b>"
        body = f"Confidence: {confidence:.0f}%\n"

        if reasons:
            body += "\nReasons:\n"
            for r in reasons[:3]:
                body += f"  - {r}\n"

        return f"{header}\n{body}"

    @staticmethod
    def format_full_analysis(snapshot) -> str:
        """Format a complete TechnicalSnapshot for /analyze command."""
        s = snapshot
        lines = [f"<b>Full Analysis: {s.symbol}</b>\n"]
        lines.append(f"Price: Rs.{s.current_price:,.2f} ({s.change_percent:+.1f}%)")

        # Trend
        lines.append(f"\n<b>Trend:</b> {s.trend.replace('_', ' ').upper()} (Strength: {s.trend_strength:.0f}/100)")

        # RSI
        lines.append(f"\n<b>RSI (14):</b> {s.rsi_14:.1f} [{s.rsi_signal.upper()}]")

        # Stochastic RSI
        if s.stoch_rsi_k is not None:
            lines.append(f"<b>StochRSI:</b> K={s.stoch_rsi_k:.1f} D={s.stoch_rsi_d:.1f} [{(s.stoch_rsi_signal or 'neutral').upper()}]")

        # MACD
        lines.append(f"\n<b>MACD:</b> {s.macd_line:.2f} / {s.macd_signal:.2f} (Hist: {s.macd_histogram:.2f})")
        lines.append(f"  Crossover: {s.macd_crossover.replace('_', ' ')}")

        # ADX
        if s.adx is not None:
            lines.append(f"\n<b>ADX:</b> {s.adx:.1f} [{(s.adx_signal or 'N/A').replace('_', ' ').upper()}]")
            lines.append(f"  DI+: {s.plus_di:.1f} | DI-: {s.minus_di:.1f}")

        # Moving Averages
        lines.append(f"\n<b>Moving Averages:</b>")
        lines.append(f"  SMA 20: Rs.{s.sma_20:,.2f} [{s.price_vs_sma20}]")
        lines.append(f"  SMA 50: Rs.{s.sma_50:,.2f} [{s.price_vs_sma50}]")
        lines.append(f"  SMA 200: Rs.{s.sma_200:,.2f} [{s.price_vs_sma200}]")
        if s.golden_cross:
            lines.append("  ** GOLDEN CROSS **")
        if s.death_cross:
            lines.append("  ** DEATH CROSS **")

        # Bollinger Bands
        lines.append(f"\n<b>Bollinger Bands:</b>")
        lines.append(f"  Upper: Rs.{s.bb_upper:,.2f} | Mid: Rs.{s.bb_middle:,.2f} | Lower: Rs.{s.bb_lower:,.2f}")
        lines.append(f"  Position: {s.bb_position.replace('_', ' ')}")

        # Ichimoku
        if s.ichimoku_tenkan is not None:
            lines.append(f"\n<b>Ichimoku Cloud:</b>")
            lines.append(f"  Tenkan: {s.ichimoku_tenkan:.2f} | Kijun: {s.ichimoku_kijun:.2f}")
            lines.append(f"  Senkou A: {s.ichimoku_senkou_a:.2f} | B: {s.ichimoku_senkou_b:.2f}")
            lines.append(f"  Signal: {(s.ichimoku_signal or 'N/A').replace('_', ' ').upper()}")

        # ATR
        if s.atr_14 is not None:
            lines.append(f"\n<b>ATR (14):</b> Rs.{s.atr_14:.2f} ({s.atr_percent:.1f}% volatility)")

        # VWAP
        if s.vwap is not None:
            lines.append(f"<b>VWAP:</b> Rs.{s.vwap:,.2f} [Price {s.price_vs_vwap}]")

        # Volume & OBV
        lines.append(f"\n<b>Volume:</b> {s.current_volume:,} ({s.volume_ratio:.1f}x avg) [{s.volume_signal.upper()}]")
        if s.obv is not None:
            lines.append(f"<b>OBV:</b> {s.obv:,.0f} [{(s.obv_signal or 'neutral').upper()}]")

        # Support & Resistance
        lines.append(f"\n<b>Support:</b> Rs.{s.support_level:,.2f} ({s.distance_to_support:.1f}% away)")
        lines.append(f"<b>Resistance:</b> Rs.{s.resistance_level:,.2f} ({s.distance_to_resistance:.1f}% away)")

        # Fibonacci
        if s.nearest_fib_support or s.nearest_fib_resistance:
            lines.append(f"\n<b>Fibonacci:</b>")
            if s.nearest_fib_support:
                lines.append(f"  Nearest Fib Support: Rs.{s.nearest_fib_support:,.2f}")
            if s.nearest_fib_resistance:
                lines.append(f"  Nearest Fib Resistance: Rs.{s.nearest_fib_resistance:,.2f}")

        # Pivot Points
        if s.pivot_point is not None:
            lines.append(f"\n<b>Pivot Points:</b>")
            lines.append(f"  R3: {s.pivot_r3} | R2: {s.pivot_r2} | R1: {s.pivot_r1}")
            lines.append(f"  PP: {s.pivot_point}")
            lines.append(f"  S1: {s.pivot_s1} | S2: {s.pivot_s2} | S3: {s.pivot_s3}")

        # Candlestick
        if s.candlestick_pattern and s.candlestick_pattern != "none":
            lines.append(f"\n<b>Candlestick:</b> {s.candlestick_pattern.replace('_', ' ')} ({s.candlestick_bias})")

        # RS vs NIFTY
        if s.rs_vs_nifty is not None:
            lines.append(f"\n<b>RS vs NIFTY:</b> {s.rs_vs_nifty:.3f} [{(s.rs_signal or 'neutral').upper()}]")

        # 52-week
        if s.high_52w is not None:
            lines.append(f"\n<b>52-Week Range:</b>")
            lines.append(f"  High: Rs.{s.high_52w:,.2f} ({s.pct_from_52w_high:+.1f}%)")
            lines.append(f"  Low: Rs.{s.low_52w:,.2f} ({s.pct_from_52w_low:+.1f}%)")

        return "\n".join(lines)

    @staticmethod
    def format_news(symbol: str, items: list) -> str:
        """Format news items for /news command."""
        if not items:
            return f"No recent news found for {symbol}."

        lines = [f"<b>News: {symbol}</b>\n"]
        for item in items[:5]:
            title = item.get("title", "")
            source = item.get("source", "")
            date = item.get("date", "")
            line = f"- {title}"
            if source:
                line += f" <i>({source})</i>"
            if date:
                # Trim the date for readability
                line += f"\n  {date[:22]}"
            lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def format_watchlist(stocks: list) -> str:
        """Format watchlist for /watchlist command."""
        if not stocks:
            return "Watchlist is empty. Use /watchlist add SYMBOL to add stocks."

        lines = [f"<b>Watchlist ({len(stocks)} stocks)</b>\n"]
        for i, sym in enumerate(stocks, 1):
            lines.append(f"  {i}. {sym}")

        lines.append("\nManage: /watchlist add SYMBOL | /watchlist remove SYMBOL")
        return "\n".join(lines)
