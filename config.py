import os
import logging
from dotenv import load_dotenv

# Load local environment variables if a .env file exists
load_dotenv()

# Setup professional application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("travel_agent")

# Fetch API Keys with defaults
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY", "")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# Verify key availability and print warning status (not raising error to support mocks)
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment. Please provide it via the Streamlit interface or .env file.")
