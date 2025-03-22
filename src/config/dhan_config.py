import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
CLIENT_ID = os.getenv('DHAN_CLIENT_ID')
ACCESS_TOKEN = os.getenv('DHAN_ACCESS_TOKEN')

if not CLIENT_ID or not ACCESS_TOKEN:
    raise ValueError("DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN must be set in .env file")

# Exchange constants
EXCHANGE_NSE = 'NSE_EQ'  # National Stock Exchange Equity
EXCHANGE_BSE = 'BSE_EQ'  # Bombay Stock Exchange Equity 