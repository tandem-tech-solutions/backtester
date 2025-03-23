"""
Dhan API configuration
Get your client ID and access token from Dhan
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")

if not CLIENT_ID or not ACCESS_TOKEN:
    raise ValueError(
        "Please set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN environment variables. "
        "Create a .env file with these variables or set them in your environment."
    )

# Market segments
EXCHANGE_NSE = "NSE_EQ"
EXCHANGE_BSE = "BSE_EQ" 