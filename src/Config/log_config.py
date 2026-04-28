import logging
import os

def setup_logging():
    LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    
    LOG_FILE = os.path.join(LOG_DIR, "system.log")
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler()
        ],
        force=True # Ensure it overrides any other basicConfig
    )

# Automatically setup on import
setup_logging()