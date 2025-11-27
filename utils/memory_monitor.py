"""
Memory monitoring and management utilities for parallel processing.

Provides tools to track memory usage, enforce limits, and prevent
out-of-memory errors during batch processing.
"""

import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """
    Snapshot of current memory usage.

    Attributes:
        total_mb: Total system memory in MB
        available_mb: Available system memory in MB
        used_mb: Used system memory in MB
        percent: Memory usage percentage
        process_mb: Current process memory in MB
    """

    total_mb: float
    available_mb: float
    used_mb: float
    percent: float
    process_mb: float

    def __repr__(self) -> str:
        """String representation of memory snapshot."""
        return (
            f"MemorySnapshot(total={self.total_mb:.1f}MB, "
            f"available={self.available_mb:.1f}MB, "
            f"used={self.used_mb:.1f}MB, "
            f"percent={self.percent:.1f}%, "
            f"process={self.process_mb:.1f}MB)"
        )


class MemoryMonitor:
    """
    Monitor system and process memory usage.

    Provides memory tracking, threshold monitoring, and warnings
    when memory usage approaches limits.
    """

    def __init__(
        self, warning_threshold_mb: float = 1000.0, critical_threshold_mb: float = 500.0
    ) -> None:
        """
        Initialize memory monitor.

        Args:
            warning_threshold_mb: Available memory threshold for warnings
            critical_threshold_mb: Available memory threshold for errors
        """
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self._process = psutil.Process(os.getpid())
        self._lock = threading.Lock()

        logger.info(
            f"MemoryMonitor initialized: "
            f"warning={warning_threshold_mb}MB, "
            f"critical={critical_threshold_mb}MB"
        )

    def get_snapshot(self) -> MemorySnapshot:
        """
        Get current memory snapshot.

        Returns:
            MemorySnapshot with current memory stats
        """
        with self._lock:
            mem = psutil.virtual_memory()
            process_mem = self._process.memory_info().rss

            snapshot = MemorySnapshot(
                total_mb=mem.total / (1024 * 1024),
                available_mb=mem.available / (1024 * 1024),
                used_mb=mem.used / (1024 * 1024),
                percent=mem.percent,
                process_mb=process_mem / (1024 * 1024),
            )

            return snapshot

    def check_memory(self) -> Optional[str]:
        """
        Check if memory usage is within acceptable limits.

        Returns:
            None if OK, warning/error message if threshold exceeded
        """
        snapshot = self.get_snapshot()

        if snapshot.available_mb < self.critical_threshold_mb:
            message = (
                f"CRITICAL: Available memory {snapshot.available_mb:.1f}MB "
                f"below critical threshold {self.critical_threshold_mb}MB"
            )
            logger.error(message)
            return message

        elif snapshot.available_mb < self.warning_threshold_mb:
            message = (
                f"WARNING: Available memory {snapshot.available_mb:.1f}MB "
                f"below warning threshold {self.warning_threshold_mb}MB"
            )
            logger.warning(message)
            return message

        return None

    def log_memory_stats(self) -> None:
        """Log current memory statistics."""
        snapshot = self.get_snapshot()
        logger.info(f"Memory stats: {snapshot}")

    @staticmethod
    def get_optimal_workers(per_worker_mb: float = 512.0, reserve_mb: float = 2048.0) -> int:
        """
        Calculate optimal number of workers based on available memory.

        Args:
            per_worker_mb: Estimated memory per worker in MB
            reserve_mb: Memory to reserve for system in MB

        Returns:
            Recommended number of workers
        """
        mem = psutil.virtual_memory()
        available_mb = mem.available / (1024 * 1024)
        usable_mb = max(0, available_mb - reserve_mb)

        workers = max(1, int(usable_mb / per_worker_mb))

        logger.info(
            f"Optimal workers calculation: "
            f"available={available_mb:.1f}MB, "
            f"usable={usable_mb:.1f}MB, "
            f"per_worker={per_worker_mb}MB, "
            f"recommended_workers={workers}"
        )

        return workers
