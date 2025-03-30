import logging
from config import LOGS_DIR

# Define logging level and log file
LOGGING_LVL = "INFO"  # Can be changed to 'DEBUG', 'ERROR', 'INFO', 'WARNING', etc.
LOG_FILE = LOGS_DIR / "app.log"

# Create log directory if it does not exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Clear any existing handlers to avoid duplication
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Convert string logging level to logging constant
log_level = getattr(logging, LOGGING_LVL.upper(), logging.INFO)

# Configure logging
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),  # Overwrite file on each run
        logging.StreamHandler(),  # Output logs to the console
    ],
)

# Get logger instance
logger = logging.getLogger(__name__)


def log_message(level, message, **kwargs):
    """
    Logs a message with additional context.

    Args:
        level (int): Logging level (e.g., logging.INFO, logging.ERROR).
        message (str): Log message.
        **kwargs: Additional context key-value pairs.
    """
    context = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.log(level, f"{message} | {context}" if context else message)
