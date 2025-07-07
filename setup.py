import os
import sys
import json
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class UpstoxMCPSetup:
    def __init__(self):
        self.project_root = Path.cwd()
        self.config_dir = self.project_root / "config"
        self.data_dir = self.project_root / "data"
        self.logs_dir = self.project_root / "logs"
        self.src_dir = self.project_root / "src"
        
    def create_directory_structure(self):
        """Create all necessary directories"""
        logging.info("Creating directory structure...")
        
        directories = [
            self.config_dir,
            self.data_dir,
            self.logs_dir,
            self.src_dir
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)
            logging.info(f"Created directory: {directory}")
    
    def setup_virtual_environment(self):
        """Create and activate virtual environment"""
        logging.info("Setting up virtual environment...")
        
        venv_path = self.project_root / "venv"
        
        if not venv_path.exists():
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
            logging.info("Virtual environment created")

        if sys.platform == "win32":
            python_path = venv_path / "Scripts" / "python.exe"
            pip_path = venv_path / "Scripts" / "pip.exe"
        else:
            python_path = venv_path / "bin" / "python"
            pip_path = venv_path / "bin" / "pip"

        requirements = [
            "mcp==1.0.0",
            "upstox-python-sdk==2.8.0",
            "requests==2.31.0",
            "websocket-client==1.6.4",
            "pandas==2.1.4",
            "numpy==1.24.3",
            "python-dotenv==1.0.0",
            "schedule==1.2.0",
            "pytz==2023.3"
        ]
        
        logging.info("Installing Python packages...")
        for requirement in requirements:
            subprocess.run([str(pip_path), "install", requirement], check=True)
            logging.info(f"Installed: {requirement}")
        
        return python_path
    
    def create_config_files(self):
        """Create configuration files"""
        logging.info("Creating configuration files...")

        upstox_config = {
            "api_key": "YOUR_UPSTOX_API_KEY",
            "api_secret": "YOUR_UPSTOX_SECRET",
            "redirect_uri": "http://localhost:8080",
            "base_url": "https://api.upstox.com",
            "max_order_value": 50000,
            "confirm_threshold": 10000,
            "auto_login": True,
            "market_hours": {
                "start": "09:15",
                "end": "15:30",
                "timezone": "Asia/Kolkata"
            }
        }
        
        with open(self.config_dir / "upstox_config.json", "w") as f:
            json.dump(upstox_config, f, indent=2)

        watchlist_config = {
            "stocks": [
                "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR",
                "ICICIBANK", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
                "LT", "ASIANPAINT", "AXISBANK", "MARUTI", "TITAN"
            ],
            "alert_settings": {
                "volume_threshold": 3.0,
                "price_change_threshold": 5.0,
                "momentum_window": 15,
                "alert_cooldown": 300,
                "max_alerts_per_hour": 5
            },
            "exchanges": ["NSE", "BSE"],
            "default_exchange": "NSE"
        }
        
        with open(self.config_dir / "watchlist.json", "w") as f:
            json.dump(watchlist_config, f, indent=2)

        env_content = """# Upstox API Credentials
UPSTOX_API_KEY=your_actual_api_key_here
UPSTOX_SECRET=your_actual_secret_here
UPSTOX_ACCESS_TOKEN=your_access_token_here

# Logging
LOG_LEVEL=INFO
ENVIRONMENT=development
"""
        
        with open(self.project_root / ".env", "w") as f:
            f.write(env_content)
        
        logging.info("Configuration files created")
    
    def create_claude_desktop_config(self, python_path):
        """Create Claude Desktop configuration"""
        logging.info("Creating Claude Desktop configuration...")

        python_path_str = str(python_path.resolve())
        mcp_server_path = str((self.src_dir / "mcp_server.py").resolve())
        project_root_str = str(self.project_root.resolve())
        
        claude_config = {
            "mcpServers": {
                "upstox-trading": {
                    "command": python_path_str,
                    "args": [mcp_server_path],
                    "env": {
                        "PYTHONPATH": project_root_str
                    }
                }
            }
        }
        
        with open(self.config_dir / "claude_desktop_config.json", "w") as f:
            json.dump(claude_config, f, indent=2)
        
        # Instructions for user
        if sys.platform == "win32":
            claude_config_dir = os.path.expanduser("~\\AppData\\Roaming\\Claude")
        else:
            claude_config_dir = os.path.expanduser("~/.config/claude")
        
        logging.info(f"Claude Desktop config created at: {self.config_dir / 'claude_desktop_config.json'}")
        logging.info(f"Please copy this file to: {claude_config_dir}")
        
        return claude_config_dir
    
    def initialize_database(self):
        """Initialize the database"""
        logging.info("Initializing database...")
        
        # Create empty database file
        db_path = self.data_dir / "trades.db"
        db_path.touch()
        
        # Import and initialize database
        sys.path.append(str(self.src_dir))
        from src.database import TradingDatabase
        
        db = TradingDatabase(str(db_path))
        logging.info("Database initialized")
    
    def create_startup_scripts(self, python_path):
        """Create startup scripts"""
        logging.info("Creating startup scripts...")

        batch_content = f"""@echo off
echo Starting Upstox MCP Trading Assistant...
cd /d "{self.project_root}"
"{python_path}" src\\mcp_server.py
pause
"""
        
        with open(self.project_root / "start_trading.bat", "w") as f:
            f.write(batch_content)

        startup_content = f"""#!/usr/bin/env python3
import sys
import os
sys.path.append('{self.src_dir}')
os.chdir('{self.project_root}')

from src.mcp_server import UpstoxMCPServer

if __name__ == "__main__":
    server = UpstoxMCPServer()
    server.run()
"""
        
        with open(self.project_root / "start_trading.py", "w") as f:
            f.write(startup_content)
        
        logging.info("Startup scripts created")
    
    def create_readme(self):
        """Create README file with instructions"""
        logging.info("Creating README file...")
        
        readme_content = """# Upstox MCP Trading Assistant

A conversational trading assistant that connects Claude Desktop with Upstox for seamless trading operations.

## Features

- **Quick Price Checks**: Get real-time stock quotes
- **Order Management**: Place, track, and cancel orders
- **Portfolio Monitoring**: View holdings and P&L
- **Smart Alerts**: Automatic opportunity detection
- **Natural Language Interface**: Trade using conversational commands

## Setup Instructions

### 1. Configure Upstox API

1. Open `config/upstox_config.json`
2. Add your Upstox API credentials:
   - `api_key`: Your Upstox API key
   - `api_secret`: Your Upstox API secret

3. Update `.env` file with your credentials:
   ```
   UPSTOX_API_KEY=your_actual_api_key_here
   UPSTOX_SECRET=your_actual_secret_here
   UPSTOX_ACCESS_TOKEN=your_access_token_here
   ```

### 2. Configure Claude Desktop

1. Copy `config/claude_desktop_config.json` to your Claude Desktop config directory:
   - Windows: `%APPDATA%\\Claude\\claude_desktop_config.json`
   - Mac: `~/Library/Application Support/Claude/claude_desktop_config.json`

2. Restart Claude Desktop

### 3. Start the MCP Server

Run one of these commands:

**Windows:**
```cmd
start_trading.bat
```

**Python:**
```cmd
python start_trading.py
```

### 4. Test the System

In Claude Desktop, try these commands:

- "What's the current price of RELIANCE?"
- "Show me my portfolio"
- "Check for trading opportunities"
- "Get system status"

## Available Commands

### Trading Commands
- `get_quote(symbol)` - Get current stock price
- `place_order(symbol, quantity, action)` - Place buy/sell order
- `get_portfolio()` - View current holdings
- `get_order_status(order_id)` - Check order status
- `cancel_order(order_id)` - Cancel pending order
- `get_balance()` - Check account balance

### Opportunity Scanner
- `check_opportunities()` - View recent opportunities
- `manage_watchlist(action, symbol)` - Manage watchlist
- `get_scanner_status()` - Check scanner status

### System Commands
- `get_notifications()` - View recent alerts
- `mark_notifications_read()` - Mark notifications as read
- `get_system_status()` - Overall system health

## Configuration

### Watchlist Settings
Edit `config/watchlist.json` to:
- Add/remove stocks to monitor
- Adjust alert sensitivity
- Configure volume and price thresholds

### Risk Management
- `max_order_value`: Maximum order size (₹50,000 default)
- `confirm_threshold`: Confirmation required for orders above this amount
- Position limits and drawdown protection

## Troubleshooting

1. **Connection Issues**: Check API credentials in config files
2. **Scanner Not Working**: Ensure market hours are correct
3. **Orders Failing**: Verify account balance and limits
4. **Claude Desktop Not Responding**: Restart Claude Desktop

## Logs

Check `logs/trading.log` for detailed system logs and error messages.

## Support

For issues and questions, check the logs directory for error details.
"""
        
        with open(self.project_root / "README.md", "w") as f:
            f.write(readme_content)
        
        logging.info("README file created")
    
    def run_setup(self):
        """Run the complete setup process"""
        logging.info("Starting Upstox MCP Trading Assistant setup...")
        
        try:
            self.create_directory_structure()

            python_path = self.setup_virtual_environment()

            self.create_config_files()

            claude_config_dir = self.create_claude_desktop_config(python_path)

            self.initialize_database()

            self.create_startup_scripts(python_path)

            self.create_readme()
            
            logging.info("Setup completed successfully!")
            
            print("\n" + "="*50)
            print("🚀 UPSTOX MCP TRADING ASSISTANT SETUP COMPLETE!")
            print("="*50)
            print("\nNext Steps:")
            print("1. Edit config/upstox_config.json with your API credentials")
            print("2. Update .env file with your tokens")
            print(f"3. Copy config/claude_desktop_config.json to {claude_config_dir}")
            print("4. Restart Claude Desktop")
            print("5. Run: start_trading.bat (Windows) or python start_trading.py")
            print("\nTest with: 'What's the current price of RELIANCE?'")
            print("="*50)
            
        except Exception as e:
            logging.error(f"Setup failed: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    setup = UpstoxMCPSetup()
    setup.run_setup()