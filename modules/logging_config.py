import os
import logging

try:
    from tqdm.auto import tqdm

    class TqdmLoggingHandler(logging.Handler):
        def __init__(self, level=logging.INFO):
            super().__init__(level)

        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except Exception:
                self.handleError(record)

    TQDM_IMPORTED = True
except ImportError:
    # tqdm does not exist before first launch
    # I will import once the UI finishes seting up the enviroment and reloads.
    TQDM_IMPORTED = False

def setup_logging(loglevel):
    if loglevel is None:
        loglevel = os.environ.get("SD_WEBUI_LOG_LEVEL")

    loghandlers = []

    if TQDM_IMPORTED:
        loghandlers.append(TqdmLoggingHandler())

    if loglevel:
        log_level = getattr(logging, loglevel.upper(), None) or logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=loghandlers
        )
