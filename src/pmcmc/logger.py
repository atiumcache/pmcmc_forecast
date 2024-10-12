import logging
import os
from datetime import datetime

import pytz
import toml

from src.utils import paths


def get_logger():
    timezone = pytz.timezone("US/Arizona")
    time_now = datetime.now(tz=timezone)
    logger = logging.getLogger(__name__)

    log_dir = os.path.join(paths.OUTPUT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging_path = os.path.join(
        log_dir, f'{time_now.strftime("%Y-%m-%d_%H-%M-%S")}.log'
    )
    config_path = os.path.join(paths.PMCMC_DIR, "config.toml")

    log_level_config = toml.load(config_path)["logging"]["level"]
    log_level = getattr(logging, log_level_config.upper(), logging.INFO)
    logging.basicConfig(filename=logging_path, encoding="utf-8", level=log_level)
    return logger
