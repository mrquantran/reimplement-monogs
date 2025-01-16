import logging
import colorlog

# Configure logging with color
handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)s:%(message)s",
        log_colors={
            "DEBUG": "white",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
)

logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)  # Set default logging level


# logger.info("Backend stopped and joined the main thread")
# logger.warning("This is a warning")
# logger.error("An error occurred")
# logger.debug("Debug message (won't show by default)")
# logger.critical("Critical issue!")