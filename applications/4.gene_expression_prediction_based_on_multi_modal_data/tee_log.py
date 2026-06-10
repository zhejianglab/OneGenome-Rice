#!/usr/bin/env python3
# Author: Yu XU <xuyu@genomics.cn>
# Created: 2026-03-28
"""
Shared utility for teeing stdout/stderr to files.

Usage:
    from tee_log import tee_output
    
    with tee_output(output_dir, "stdout.log", "stderr.log"):
        print("This goes to both console and file")
"""

import contextlib
import os
import sys
import threading
from typing import TextIO


class TeeStream:
    """Write to an original stream and a dedicated log file (thread-safe)."""

    def __init__(self, stream: TextIO, log_file: TextIO, lock: threading.Lock):
        self._stream = stream
        self._log = log_file
        self._lock = lock

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._stream.flush()
        with self._lock:
            self._log.write(data)
            self._log.flush()
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        with self._lock:
            self._log.flush()

    def fileno(self) -> int:
        return self._stream.fileno()

    def isatty(self) -> bool:
        return self._stream.isatty()


@contextlib.contextmanager
def tee_output(output_base_dir: str, out_file: str, err_file: str):
    """Tee stdout and stderr to files.
    
    Args:
        output_base_dir: Directory for log files (created if doesn't exist)
        out_file: Filename for stdout log (e.g., "train.o" or "run.o")
        err_file: Filename for stderr log (e.g., "train.e" or "run.e")
        
    Yields:
        None
        
    Example:
        with tee_output("/path/to/output", "run.o", "run.e"):
            print("Goes to both console and /path/to/output/run.o")
            sys.stderr.write("Goes to both console and /path/to/output/run.e")
    """
    os.makedirs(output_base_dir, exist_ok=True)
    out_path = os.path.join(output_base_dir, out_file)
    err_path = os.path.join(output_base_dir, err_file)
    
    lock_out = threading.Lock()
    lock_err = threading.Lock()
    
    log_out = open(out_path, "a", encoding="utf-8", errors="replace")
    log_err = open(err_path, "a", encoding="utf-8", errors="replace")
    
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = TeeStream(old_out, log_out, lock_out)
    sys.stderr = TeeStream(old_err, log_err, lock_err)
    
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = old_out
        sys.stderr = old_err
        log_out.close()
        log_err.close()
