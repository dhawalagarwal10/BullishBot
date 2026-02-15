# BullishBot

a trading assistant that connects Claude Desktop and Telegram with Upstox. you can trade, monitor your portfolio, get alerts, and scan for opportunities - all through chat.

> always optimistic about your losses

## what it does

- **Claude Desktop integration** - talk to Claude in natural language and it places orders, checks prices, analyzes stocks through MCP tools
- **Telegram bot** - get real-time alerts, check portfolio, scan markets, all from your phone
- **market radar** - scans NSE stocks in the background and flags unusual volume/price moves
- **portfolio guard** - monitors your holdings and alerts you on big P&L changes
- **news watcher** - tracks financial news for your watchlist and holdings, sends urgent alerts
- **opportunity scanner** - finds trading setups based on technical signals
- **paper trading** - test strategies without risking real money
- **technical analysis** - RSI, MACD, moving averages, trend detection, all built in

## how it works

there are two main pieces:

1. **MCP Server** (`src/mcp_server.py`) - runs inside Claude Desktop. this is how Claude talks to Upstox. you ask Claude to buy something, it calls the MCP tools, order gets placed.

2. **Telegram Bot** (`start_bot.py`) - runs separately. starts up all the background services (market radar, portfolio guard, news watcher, signal monitor) and connects them to Telegram so you get alerts on your phone.

```
Claude Desktop  --->  MCP Server  --->  Upstox API
                                            ^
Telegram Bot  ---+---> Signal Monitor       |
                 +---> Portfolio Guard  -----+
                 +---> Market Radar    -----+
                 +---> News Watcher
                 +---> Opportunity Scanner
```

## setup

### prerequisites

- python 3.8+
- an Upstox account with API access
- Claude Desktop (for the MCP integration)
- a Telegram bot token (talk to @BotFather)

### installation

```bash
git clone https://github.com/dhawalagarwal10/BullishBot.git
cd BullishBot
pip install -r requirements.txt
```

or run the setup script which creates everything for you:

```bash
python setup.py
```

### configuration

1. create a `.env` file in the root:

```
UPSTOX_API_KEY=your_api_key
UPSTOX_API_SECRET=your_secret
UPSTOX_ACCESS_TOKEN=your_access_token
LOG_LEVEL=INFO
ENVIRONMENT=development
```

2. create `config/upstox_config.json`:

```json
{
  "api_key": "your_api_key",
  "api_secret": "your_secret",
  "redirect_uri": "http://localhost:8080",
  "base_url": "https://api.upstox.com",
  "max_order_value": 50000,
  "confirm_threshold": 10000,
  "auto_login": true,
  "market_hours": {
    "start": "09:15",
    "end": "15:30",
    "timezone": "Asia/Kolkata"
  }
}
```

3. create `config/telegram_config.json`:

```json
{
  "bot_token": "your_telegram_bot_token",
  "allowed_chat_ids": [your_chat_id],
  "notifications": {
    "signals": true,
    "min_signal_confidence": 50,
    "portfolio_alerts": true,
    "pnl_change_threshold_percent": 5.0,
    "system_alerts": true
  },
  "polling_interval_seconds": 10
}
```

4. edit `config/watchlist.json` to add the stocks you want to track

### setting up Claude Desktop

copy the contents of `config/claude_desktop_config.json` into your Claude Desktop config:

- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Mac: `~/Library/Application Support/Claude/claude_desktop_config.json`

then restart Claude Desktop.

### running

**start the telegram bot and all background services:**

```bash
python start_bot.py
```

**the MCP server** starts automatically when Claude Desktop launches (if you set up the config correctly).

## telegram commands

| command             | what it does                      |
| ------------------- | --------------------------------- |
| `/start`            | shows help and available commands |
| `/portfolio`        | your current holdings and P&L     |
| `/quote RELIANCE`   | get current price for a stock     |
| `/signals`          | recent trading signals            |
| `/status`           | system status and uptime          |
| `/analyze RELIANCE` | technical analysis for a stock    |
| `/scan`             | scan for trading opportunities    |
| `/watchlist`        | manage your watchlist             |
| `/news`             | latest financial news             |
| `/alerts`           | your alert settings               |

## claude desktop commands

just talk naturally. some examples:

- "what's the price of RELIANCE?"
- "buy 10 shares of INFY"
- "show my portfolio"
- "analyze TCS for me"
- "what's the market sentiment today?"
- "check for trading opportunities"
- "show me paper trade history"

## project structure

```
BullishBot/
├── start_bot.py              # launches telegram bot + all services
├── setup.py                  # first-time setup script
├── requirements.txt
├── config/
│   ├── upstox_config.json    # upstox API config
│   ├── telegram_config.json  # telegram bot config
│   ├── watchlist.json        # stocks to monitor
│   └── instrument_keys.json  # NSE instrument mappings
├── src/
│   ├── mcp_server.py         # Claude Desktop MCP integration
│   ├── broker_client.py      # Upstox API wrapper
│   ├── technical_analysis.py # RSI, MACD, moving averages etc
│   ├── market_radar.py       # background market scanner
│   ├── portfolio_guard.py    # holdings monitor + alerts
│   ├── news_watcher.py       # financial news tracker
│   ├── news_analyzer.py      # news sentiment analysis
│   ├── opportunity_scanner.py # trading setup detection
│   ├── paper_trader.py       # paper trading engine
│   ├── realtime_streamer.py  # live market data buffer
│   ├── database.py           # SQLite database layer
│   ├── instrument_resolver.py # stock symbol resolution
│   ├── notification_manager.py # notification routing
│   └── telegram_bot/
│       ├── bot.py            # telegram command handlers
│       ├── signal_monitor.py # polls DB, pushes signals to telegram
│       └── formatters.py     # message formatting
├── data/                     # databases (gitignored)
└── logs/                     # log files (gitignored)
```

## disclaimer

this is a personal project. trading involves risk. don't blame the bot if you lose money - it's called BullishBot, not ProfitableBot.

not a financial advicer. use at your own risk.

---

made by a trader who got tired of clicking buttons.

[star this repo](https://github.com/dhawalagarwal10/BullishBot) if you find it useful · [report a bug](https://github.com/dhawalagarwal10/BullishBot/issues) · [request a feature](https://github.com/dhawalagarwal10/BullishBot/issues)

_the market can stay irrational longer than you can stay solvent. trade responsibly._
