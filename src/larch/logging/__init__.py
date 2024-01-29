from __future__ import annotations

import datetime
import logging
import sys
import time
from contextlib import contextmanager

LOGGER_NAME = __name__.split(".")[0]
FILE_LOG_FORMAT = "%(name)s.%(levelname)s: %(message)s"
CONSOLE_LOG_FORMAT = "{asctime} [{elapsedTime}] {name:s}.{levelname:s}: {message:s}"
DEFAULT_LOG_LEVEL = logging.INFO


def timesize_single(t):
    if t < 60:
        return f"{t:.2f}s"
    elif t < 3600:
        return f"{t / 60:.2f}m"
    elif t < 86400:
        return f"{t / 3600:.2f}h"
    else:
        return f"{t / 86400:.2f}d"


def timesize_stack(t):
    if t < 60:
        return f"{t:.2f}s"
    elif t < 3600:
        return f"{t // 60:.0f}m {timesize_stack(t % 60)}"
    elif t < 86400:
        return f"{t // 3600:.0f}h {timesize_stack(t % 3600)}"
    else:
        return f"{t // 86400:.0f}d {timesize_stack(t % 86400)}"


class ElapsedTimeFormatter(logging.Formatter):
    def format(self, record):
        duration_milliseconds = record.relativeCreated
        hours, rem = divmod(duration_milliseconds / 1000, 3600)
        minutes, seconds = divmod(rem, 60)
        if hours:
            record.elapsedTime = f"{int(hours):0>2}:{int(minutes):0>2}:{seconds:07.4f}"
        else:
            record.elapsedTime = f"{int(minutes):0>2}:{seconds:07.4f}"
        return super().format(record)


def log_to_console(level=None):
    if level is None:
        level = DEFAULT_LOG_LEVEL

    logger = logging.getLogger(LOGGER_NAME)

    if level < logger.level or logger.level == logging.NOTSET:
        logger.setLevel(level)

    # avoid creation of multiple stream handlers for logging to console
    for entry in logger.handlers:
        if (isinstance(entry, logging.StreamHandler)) and (
            entry.formatter._fmt == CONSOLE_LOG_FORMAT
        ):
            if level < entry.level:
                entry.setLevel(level)
            logger.propagate = False
            return logger

    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setLevel(level)
    formatter = ElapsedTimeFormatter(CONSOLE_LOG_FORMAT, style="{")
    formatter.default_time_format = "%H:%M:%S"
    formatter.default_msec_format = "%s.%03d"
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        logging_start_time = datetime.datetime.fromtimestamp(
            logging._startTime
        ).strftime("%Y-%m-%d %I:%M:%S %p")
    except Exception:
        pass
    else:
        logger.info(f"Logging started {logging_start_time}")

    return logger


def log_to_file(filename, level=None):
    if level is None:
        level = DEFAULT_LOG_LEVEL

    logger = logging.getLogger(LOGGER_NAME)

    # avoid creation of multiple file handlers for logging to the same file
    for entry in logger.handlers:
        if (isinstance(entry, logging.FileHandler)) and (
            entry.baseFilename == filename
        ):
            return logger

    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(FILE_LOG_FORMAT))
    logger.addHandler(file_handler)

    return logger


logger = log = log_to_console()


@contextmanager
def timing_log(label=""):
    start_time = time.time()
    log.critical(f"<TIME BEGINS> {label}")
    try:
        yield
    except Exception:  # noqa: E722
        log.critical(
            f"<TIME ERROR!> {label} <{timesize_stack(time.time() - start_time)}>"
        )
        raise
    else:
        log.critical(
            f"< TIME ENDS > {label} <{timesize_stack(time.time() - start_time)}>"
        )


class TimingLog:
    def __init__(self, label="", log=None, level=50):
        global logger
        if log is None:
            log = logger
        self.label = label
        self.log = log
        self.level = level
        self.split_time = None
        self.current_task = ""

    def __enter__(self):
        self.start_time = time.time()
        self.log.log(self.level, f"<BEGIN> {self.label}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        now = time.time()
        if self.split_time is not None:
            self.log.log(
                self.level,
                f"<SPLIT> {self.label} / Final <{timesize_stack(now - self.split_time)}>",
            )
        if exc_type is None:
            self.log.log(
                self.level,
                f"<-END-> {self.label} <{timesize_stack(now - self.start_time)}>",
            )
        else:
            self.log.log(
                self.level,
                f"<ERROR> {self.label} <{timesize_stack(now - self.start_time)}>",
            )

    def split(self, note=""):
        if self.split_time is None:
            self.split_time = self.start_time
        now = time.time()
        if note:
            note = " / " + note
        self.log.log(
            self.level,
            f"<SPLIT> {self.label}{note} <{timesize_stack(now - self.split_time)}>",
        )
        self.split_time = now
