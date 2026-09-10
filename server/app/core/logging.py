"""
ShipRule CDLP - Central Logging Configuration
"""

import logging

def get_logger(name: str = "ShipRuleAPI") -> logging.Logger:
    """Returns a configured logger instance for application modules."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = get_logger()
