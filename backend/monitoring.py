import os
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

try:
    from prometheus_fastapi_instrumentator import Instrumentator
except Exception:
    Instrumentator = None

try:
    import sentry_sdk
except Exception:
    sentry_sdk = None


def setup_logging(base_dir: Path):
    """
    Creates a logger that writes logs to:
    logs/backend.log
    and also prints logs in the terminal.
    """
    log_dir = base_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("scam_phishing_api")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        file_handler = RotatingFileHandler(
            log_dir / "backend.log",
            maxBytes=2_000_000,
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def setup_prometheus(app, logger=None):
    """
    Adds /metrics endpoint for Prometheus.
    If the package is not installed, the app keeps working normally.
    """
    if Instrumentator is None:
        if logger:
            logger.warning(
                "event=prometheus_not_enabled | reason=prometheus_fastapi_instrumentator_not_installed"
            )
        return

    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    if logger:
        logger.info("event=prometheus_enabled | endpoint=/metrics")


def setup_sentry(logger=None):
    """
    Enables Sentry only if SENTRY_DSN exists in environment variables.
    If SENTRY_DSN is missing, nothing breaks.
    """
    sentry_dsn = os.getenv("SENTRY_DSN")

    if sentry_sdk is None:
        if logger:
            logger.warning("event=sentry_not_enabled | reason=sentry_sdk_not_installed")
        return

    if not sentry_dsn:
        if logger:
            logger.info("event=sentry_not_enabled | reason=SENTRY_DSN_not_set")
        return

    sentry_sdk.init(
        dsn=sentry_dsn,
        traces_sample_rate=1.0,
        send_default_pii=False
    )

    if logger:
        logger.info("event=sentry_enabled")


def setup_monitoring(app, base_dir: Path):
    """
    Main function used inside main.py.
    It initializes:
    1. Backend logs
    2. Prometheus metrics endpoint /metrics
    3. Sentry error tracking if SENTRY_DSN exists
    """
    logger = setup_logging(base_dir)
    setup_prometheus(app, logger)
    setup_sentry(logger)

    logger.info("event=monitoring_initialized")

    return logger
