"""
Performance profiling utilities for Game Camera Analyzer.

Provides easy-to-use profiling tools, decorators, and context managers
for identifying performance bottlenecks.
"""

import cProfile
import functools
import io
import logging
import pstats
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, TypeVar, cast

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class ProfileResult:
    """
    Results from a profiling session.

    Attributes:
        function_name: Name of profiled function
        total_time: Total execution time in seconds
        call_count: Number of times function was called
        top_functions: List of (function, time, calls) for top consumers
        profile_stats: Raw pstats.Stats object
    """

    function_name: str
    total_time: float
    call_count: int
    top_functions: List[tuple] = field(default_factory=list)
    profile_stats: Optional[pstats.Stats] = None

    def print_summary(self, top_n: int = 20) -> None:
        """
        Print profiling summary.

        Args:
            top_n: Number of top functions to display
        """
        print(f"\n{'='*80}")
        print(f"Profile Results: {self.function_name}")
        print(f"{'='*80}")
        print(f"Total Time: {self.total_time:.4f}s")
        print(f"Call Count: {self.call_count}")
        print(f"\nTop {top_n} Functions by Time:")
        print(f"{'-'*80}")

        if self.profile_stats:
            self.profile_stats.print_stats(top_n)

    def save_to_file(self, filepath: Path) -> None:
        """
        Save profiling results to file.

        Args:
            filepath: Output file path
        """
        if self.profile_stats:
            self.profile_stats.dump_stats(str(filepath))
            logger.info(f"Profiling results saved to {filepath}")


class Profiler:
    """
    Context manager and decorator for function profiling.

    Provides easy profiling of functions and code blocks with
    detailed performance statistics.
    """

    def __init__(self, name: str = "profile", enabled: bool = True) -> None:
        """
        Initialize profiler.

        Args:
            name: Name for this profiling session
            enabled: Whether profiling is enabled
        """
        self.name = name
        self.enabled = enabled
        self.profiler = cProfile.Profile() if enabled else None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self) -> "Profiler":
        """Start profiling."""
        if self.enabled and self.profiler:
            self.start_time = time.time()
            self.profiler.enable()
            logger.debug(f"Profiling started: {self.name}")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop profiling and log results."""
        if self.enabled and self.profiler:
            self.profiler.disable()
            self.end_time = time.time()
            duration = self.end_time - self.start_time if self.start_time else 0
            logger.info(f"Profiling complete: {self.name} ({duration:.4f}s)")

    def get_results(self, top_n: int = 20) -> ProfileResult:
        """
        Get profiling results.

        Args:
            top_n: Number of top functions to include

        Returns:
            ProfileResult object
        """
        if not self.enabled or not self.profiler:
            raise RuntimeError("Profiler not enabled or not run")

        s = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=s)
        stats.sort_stats("cumulative")

        total_time = (self.end_time or 0) - (self.start_time or 0)

        # Extract top functions
        top_functions = []
        for func, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:top_n]:
            top_functions.append((str(func), ct, cc))

        return ProfileResult(
            function_name=self.name,
            total_time=total_time,
            call_count=1,
            top_functions=top_functions,
            profile_stats=stats,
        )


def profile_function(name: Optional[str] = None, enabled: bool = True) -> Callable[[F], F]:
    """
    Decorator to profile a function.

    Args:
        name: Name for profiling session (defaults to function name)
        enabled: Whether profiling is enabled

    Returns:
        Decorated function

    Example:
        @profile_function(name="my_slow_function")
        def my_slow_function(x):
            return sum(range(x))
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            profiler_name = name or func.__name__

            with Profiler(name=profiler_name, enabled=enabled) as profiler:
                result = func(*args, **kwargs)

            if enabled:
                prof_results = profiler.get_results()
                logger.info(f"Function {profiler_name} took {prof_results.total_time:.4f}s")

            return result

        return cast(F, wrapper)

    return decorator


@contextmanager
def profile_block(name: str, log_results: bool = True):
    """
    Context manager to profile a code block.

    Args:
        name: Name for this profiling block
        log_results: Whether to log results after profiling

    Yields:
        Profiler instance

    Example:
        with profile_block("image_processing") as profiler:
            # ... code to profile ...
            pass

        results = profiler.get_results()
        results.print_summary()
    """
    profiler = Profiler(name=name, enabled=True)

    try:
        with profiler:
            yield profiler
    finally:
        if log_results:
            try:
                results = profiler.get_results()
                logger.info(f"Block '{name}' execution time: {results.total_time:.4f}s")
            except Exception as e:
                logger.error(f"Failed to get profiling results: {e}")


class PerformanceTimer:
    """
    Simple timer for measuring code execution time.

    Lighter weight than full profiling, useful for quick measurements.
    """

    def __init__(self, name: str, enabled: bool = True) -> None:
        """
        Initialize performance timer.

        Args:
            name: Name for this timer
            enabled: Whether timing is enabled
        """
        self.name = name
        self.enabled = enabled
        self.start_time: Optional[float] = None
        self.duration: Optional[float] = None

    def __enter__(self) -> "PerformanceTimer":
        """Start timer."""
        if self.enabled:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop timer and log duration."""
        if self.enabled and self.start_time is not None:
            self.duration = time.perf_counter() - self.start_time
            logger.info(f"{self.name}: {self.duration*1000:.2f}ms")

    def get_duration_ms(self) -> float:
        """
        Get duration in milliseconds.

        Returns:
            Duration in milliseconds or 0 if not run
        """
        return (self.duration or 0) * 1000
