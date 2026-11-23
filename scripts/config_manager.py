import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.json"


def load_config(config_path: Path = None, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Load configuration from JSON file and apply optional CLI overrides.
    """
    cfg_file = config_path or DEFAULT_CONFIG_PATH
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found at: {cfg_file}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    if overrides:
        for k, v in overrides.items():
            if v is not None:
                config[k] = v

    # Normalize paths
    config["input_dir"] = os.path.normpath(config["input_dir"])
    config["output_dir"] = os.path.normpath(config["output_dir"])

    # Ensure output directories exist
    ensure_directories(config["output_dir"])

    return config


def ensure_directories(output_base: str):
    """
    Ensure all standard output directories exist.
    """
    dirs = [
        os.path.join(output_base, "reports"),
        os.path.join(output_base, "raw_extracted"),
        os.path.join(output_base, "chandipur", "daily"),
        os.path.join(output_base, "regional", "daily"),
        os.path.join(output_base, "combined"),
        os.path.join(output_base, "logs"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def setup_logging(output_dir: str, level: str = "INFO") -> logging.Logger:
    """
    Setup logging to both console and log file.
    """
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "pipeline.log")

    logger = logging.getLogger("NASA_LIS_Pipeline")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers if setup_logging is called multiple times
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Stream (Console) handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger
