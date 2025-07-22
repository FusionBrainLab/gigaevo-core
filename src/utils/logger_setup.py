"""
Enhanced Colored Logging Setup for MetaEvolve System

Comprehensive loguru configuration with colored console output and file logging.
"""

from datetime import datetime, timezone
import os
import sys

from loguru import logger


def setup_logger(
    log_dir: str = "logs",
    level: str = "INFO",
    rotation: str = "50 MB",
    retention: str = "30 days",
    enable_colors: bool = True,
) -> str:
    """
    Set up enhanced colored logging with file and console output.

    Args:
        log_dir: Directory for log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: Log rotation policy (e.g., "50 MB", "1 day")
        retention: Log retention policy (e.g., "30 days", "1 month")
        enable_colors: Whether to enable colored console output

    Returns:
        Path to the main log file
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"evolution_{timestamp}.log")

    # Remove any existing handlers to avoid duplicates
    logger.remove()

    # Enhanced console format with comprehensive coloring
    if enable_colors and sys.stdout.isatty():
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<blue>{function}</blue>:<yellow>{line}</yellow> | "
            "<level>{message}</level>"
        )
    else:
        # Fallback format for non-TTY environments
        console_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )

    # Add console handler with enhanced coloring
    logger.add(
        lambda msg: print(msg, end=""),
        level=level,
        format=console_format,
        colorize=enable_colors and sys.stdout.isatty(),
        backtrace=True,
        diagnose=True,
    )

    # Add file handler (no colors in file)
    logger.add(
        log_file,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation=rotation,
        retention=retention,
        compression="zip",
        encoding="utf-8",
        backtrace=True,
        diagnose=True,
    )

    logger.info(
        f"🎨 Enhanced colored logger initialized. Logging to console and {log_file}"
    )
    logger.debug(
        f"📊 Log level: {level}, Colors: {enable_colors and sys.stdout.isatty()}"
    )
    return log_file
