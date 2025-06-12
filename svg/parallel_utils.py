"""
Parallel processing utilities for API calls and batch operations.
Provides specialized tools for:
1. Asynchronous API requests with proper rate limiting
2. Multithreading for I/O-bound tasks
3. Multiprocessing for CPU-bound tasks
All with timeout handling and result processing.
"""

import os
import json
import time
import asyncio
import signal
import threading
import multiprocessing
import multiprocessing.pool
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, TimeoutError
from functools import partial
from typing import List, Callable, Dict, Any, Optional, Union, Tuple, Iterable, TypeVar
from tqdm.auto import tqdm

try:
    from loguru import logger
except ImportError:
    import logging
    # Create a logger that mimics loguru's interface if loguru is not available
    class LoguruCompatLogger:
        def __init__(self):
            self.logger = logging.getLogger("parallel_utils")
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(levelname)s | %(asctime)s | %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
        def debug(self, message, *args, **kwargs):
            self.logger.debug(message, *args, **kwargs)
            
        def info(self, message, *args, **kwargs):
            self.logger.info(message, *args, **kwargs)
            
        def warning(self, message, *args, **kwargs):
            self.logger.warning(message, *args, **kwargs)
            
        def error(self, message, *args, **kwargs):
            self.logger.error(message, *args, **kwargs)
            
        def critical(self, message, *args, **kwargs):
            self.logger.critical(message, *args, **kwargs)
            
        def configure(self, **kwargs):
            if "level" in kwargs:
                self.logger.setLevel(kwargs["level"])
            
    logger = LoguruCompatLogger()


class AsyncApiCaller:
    """
    Specialized class for making asynchronous API calls with proper rate limiting,
    timeout handling, and immediate result saving.
    
    Features:
    - Token bucket rate limiting to avoid API rate limit errors
    - Configurable concurrency with max_workers parameter
    - Per-request timeout handling
    - Immediate result saving to avoid data loss
    - Progress tracking with tqdm
    - Detailed logging of success/failure
    """
    
    @classmethod
    async def _execute_with_timeout(cls, 
                                   fn: Callable, 
                                   item: Any, 
                                   timeout: int = 60) -> Optional[Any]:
        """
        Execute a function with timeout in a thread pool.
        
        Args:
            fn: The function to call
            item: The input data
            timeout: Timeout in seconds
            
        Returns:
            The function result or None if timed out
        """
        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, fn, item),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None
    
    @classmethod
    async def process_items(cls,
                          items: List[Any],
                          process_fn: Callable,
                          output_file: Optional[str] = None,
                          rate_limit: int = 20,
                          max_workers: Optional[int] = None,
                          timeout: int = 60,
                          save_interval: int = 1) -> List[Any]:
        """
        Process items asynchronously with proper rate limiting and immediate saving.
        
        Args:
            items: List of items to process
            process_fn: Function to call on each item
            output_file: File to save results (will append if exists)
            rate_limit: Maximum requests per second
            max_workers: Maximum concurrent tasks (defaults to 2*rate_limit)
            timeout: Timeout in seconds for each task
            save_interval: How many results to process before saving stats
            
        Returns:
            List of successful results
        """
        if max_workers is None:
            max_workers = min(100, rate_limit * 2)  # Default: 2x rate_limit up to 100
        
        # Create directory for output file if needed
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
        # Create rate limiter using token bucket algorithm
        class RateLimiter:
            def __init__(self, rate_limit):
                self.rate_limit = rate_limit
                self.tokens = rate_limit
                self.updated_at = time.monotonic()
                self.lock = asyncio.Lock()
                
            async def acquire(self):
                async with self.lock:
                    now = time.monotonic()
                    
                    # Add new tokens based on time elapsed
                    elapsed = now - self.updated_at
                    new_tokens = elapsed * self.rate_limit
                    self.tokens = min(self.rate_limit, self.tokens + new_tokens)
                    self.updated_at = now
                    
                    # If no tokens, wait until next token is available
                    if self.tokens < 1:
                        wait_time = (1 - self.tokens) / self.rate_limit
                        await asyncio.sleep(wait_time)
                        self.tokens = 0
                        self.updated_at = time.monotonic()
                    else:
                        self.tokens -= 1
        
        # Initialize rate limiter and concurrency control
        rate_limiter = RateLimiter(rate_limit)
        semaphore = asyncio.Semaphore(max_workers)
        file_lock = asyncio.Lock()
        
        # Stats tracking
        processed = 0
        succeeded = 0
        failed = 0
        timed_out = 0
        
        # Process a single item
        async def process_item(item, idx):
            nonlocal processed, succeeded, failed, timed_out
            
            async with semaphore:  # Limit concurrency
                await rate_limiter.acquire()  # Respect rate limits
                
                start_time = time.time()
                try:
                    # Run task with timeout
                    result = await cls._execute_with_timeout(
                        process_fn, item, timeout=timeout
                    )
                    
                    elapsed = time.time() - start_time
                    
                    # Handle result
                    if result is None:
                        timed_out += 1
                        print(f"[{idx}] Task timed out after {timeout}s")
                        return None
                    
                    # Success - save result immediately if requested
                    succeeded += 1
                    processed += 1
                    
                    if output_file:
                        async with file_lock:
                            with open(output_file, 'a') as f:
                                f.write(json.dumps(result) + '\n')
                    
                    # Log progress periodically
                    if processed % save_interval == 0:
                        print(f"Processed: {processed}, Success: {succeeded}, "
                              f"Failed: {failed}, Timeouts: {timed_out}, "
                              f"Last time: {elapsed:.2f}s")
                    
                    return result
                    
                except Exception as e:
                    elapsed = time.time() - start_time
                    failed += 1
                    processed += 1
                    print(f"L [{idx}] Error: {str(e)[:100]}... ({elapsed:.2f}s)")
                    return None
        
        # Create and run all tasks with progress bar
        tasks = [process_item(item, i) for i, item in enumerate(items)]
        results = []
        
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing API calls"):
            try:
                result = await f
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"Unexpected error in task: {e}")
        
        # Final stats
        print(f"\nCompleted {len(items)} tasks:")
        print(f"- Succeeded: {succeeded}")
        print(f"- Failed: {failed}")
        print(f"- Timed out: {timed_out}")
        
        return results
    
    @classmethod
    def run(cls,
           items: List[Any],
           process_fn: Callable,
           output_file: Optional[str] = None,
           write_mode: str = 'a',
           rate_limit: int = 20,
           max_workers: Optional[int] = None,
           timeout: int = 60,
           save_interval: int = 1) -> List[Any]:
        """
        Process items asynchronously with proper rate limiting.
        This is a convenience wrapper that handles the event loop setup.
        
        Args:
            items: List of items to process
            process_fn: Function to call on each item
            output_file: File to save results (will append if exists)
            write_mode: File write mode ('a' for append, 'w' for overwrite)
            rate_limit: Maximum requests per second
            max_workers: Maximum concurrent tasks (defaults to 2*rate_limit)
            timeout: Timeout in seconds for each task
            save_interval: How many results to process before saving stats
            
        Returns:
            List of successful results
        """
        print(f"Starting async API processing with:")
        print(f"- Items: {len(items)}")
        print(f"- Rate limit: {rate_limit} req/s")
        print(f"- Max workers: {max_workers or (rate_limit * 2)}")
        print(f"- Timeout: {timeout} seconds")
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if write_mode == 'w' and os.path.exists(output_file):
            print(f"Output file {output_file} exists. Overwriting.")
            with open(output_file, 'w') as f:
                f.write('')
            
        try:
            return loop.run_until_complete(
                cls.process_items(
                    items=items,
                    process_fn=process_fn,
                    output_file=output_file,
                    rate_limit=rate_limit,
                    max_workers=max_workers,
                    timeout=timeout,
                    save_interval=save_interval
                )
            )
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Saved results are preserved in the output file.")
            return []


# Type variable for generic function
T = TypeVar('T')
R = TypeVar('R')


class TimeoutHandler:
    """
    Timeout handler for functions without native timeout support.
    Used internally by parallelization utilities.
    """
    
    @staticmethod
    def _timeout_handler(signum, frame):
        raise TimeoutError("Function execution timed out")
    
    @classmethod
    def run_with_timeout(cls, func: Callable[..., R], *args, timeout: Optional[int] = None, **kwargs) -> Optional[R]:
        """
        Run a function with a timeout.
        
        Args:
            func: Function to run
            timeout: Timeout in seconds (None for no timeout)
            args: Positional arguments to pass to the function
            kwargs: Keyword arguments to pass to the function
            
        Returns:
            Function result or None if timed out
        """
        if timeout is None:
            return func(*args, **kwargs)
            
        # Set up the timeout handler
        original_handler = signal.signal(signal.SIGALRM, cls._timeout_handler)
        signal.alarm(timeout)
        
        try:
            result = func(*args, **kwargs)
            return result
        except TimeoutError:
            return None
        finally:
            # Reset the alarm and restore the original handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)


class ThreadPool:
    """
    Thread pool for parallel execution of I/O-bound tasks with timeout support.
    Uses concurrent.futures.ThreadPoolExecutor directly for better efficiency.
    
    Features:
    - Progress tracking with tqdm
    - Individual task timeout using concurrent.futures
    - Exception handling for each task with proper logging
    - Support for mapping a function over an iterable
    """
    
    @classmethod
    def map(cls,
           func: Callable[[T], R],
           items: Iterable[T],
           max_workers: Optional[int] = None,
           timeout: Optional[int] = None,
           show_progress: bool = True,
           desc: str = "Processing",
           error_value: Any = None,
           log_level: str = "INFO") -> List[R]:
        """
        Apply a function to each item in an iterable in parallel using threads.
        
        Args:
            func: Function to apply to each item
            items: Iterable of items to process
            max_workers: Maximum number of threads (None for auto-detection)
            timeout: Timeout in seconds for each task (None for no timeout)
            show_progress: Whether to show progress bar
            desc: Description for the progress bar
            error_value: Value to return for failed items (default None)
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            
        Returns:
            List of results in the same order as input items
        """
        items = list(items)  # Convert to list for tqdm and indexing
        results = [error_value] * len(items)
        
        # Set up logging
        original_level = None
        log_method = getattr(logger, log_level.lower(), logger.info)
        
        log_method(f"Starting thread pool execution with {len(items)} items")
        if max_workers:
            log_method(f"Using {max_workers} worker threads")
        
        # Process a single item with appropriate error handling
        def process_item(item_with_idx):
            idx, item = item_with_idx
            start_time = time.time()
            
            try:
                # Execute the function
                return idx, func(item)
            except Exception as e:
                logger.error(f"Error processing item {idx}: {str(e)}")
                return idx, error_value
            finally:
                elapsed = time.time() - start_time
                logger.debug(f"Item {idx} processed in {elapsed:.2f}s")
        
        # Create executor and process items with progress tracking
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the executor
            items_with_idx = list(enumerate(items))
            futures = [executor.submit(process_item, item_idx) for item_idx in items_with_idx]
            
            # Process results as they complete
            if show_progress:
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(items),
                    desc=desc
                ):
                    try:
                        if timeout:
                            idx, result = future.result(timeout=timeout)
                        else:
                            idx, result = future.result()
                        results[idx] = result
                    except concurrent.futures.TimeoutError:
                        logger.warning(f"Task timed out after {timeout} seconds")
                    except Exception as e:
                        logger.error(f"Unexpected error in task: {str(e)}")
            else:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        if timeout:
                            idx, result = future.result(timeout=timeout)
                        else:
                            idx, result = future.result()
                        results[idx] = result
                    except concurrent.futures.TimeoutError:
                        logger.warning(f"Task timed out after {timeout} seconds")
                    except Exception as e:
                        logger.error(f"Unexpected error in task: {str(e)}")
            
        success_count = sum(1 for r in results if r is not error_value)
        log_method(f"Completed {len(items)} tasks: {success_count} succeeded, {len(items) - success_count} failed")
        
        return results
    
    @classmethod
    def run(cls,
           func: Callable[..., R],
           items: List[Any],
           max_workers: Optional[int] = None,
           timeout: Optional[int] = None,
           show_progress: bool = True,
           desc: str = "Processing",
           error_value: Any = None,
           log_level: str = "INFO") -> List[R]:
        """
        Run a function on each item in a list in parallel using threads.
        
        Args:
            func: Function to call on each item
            items: List of items to process (each item is passed as the sole argument to func)
            max_workers: Maximum number of threads (None for auto-detection)
            timeout: Timeout in seconds for each task (None for no timeout)
            show_progress: Whether to show progress bar
            desc: Description for the progress bar
            error_value: Value to return for failed items (default None)
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            
        Returns:
            List of results in the same order as input items
        """
        return cls.map(func, items, max_workers, timeout, show_progress, desc, error_value, log_level)


class ProcessPool:
    """
    Process pool for parallel execution of CPU-bound tasks with timeout support.
    Optimized for computationally intensive tasks that benefit from multiple cores.
    
    Features:
    - Progress tracking with tqdm
    - Individual task timeout
    - Exception handling for each task with proper logging
    - Support for mapping a function over an iterable
    - Process context control ('spawn', 'fork', 'forkserver')
    """
    
    @staticmethod
    def _worker_with_timeout(func, item, timeout):
        """
        Worker function with timeout support for process pool.
        Uses TimeoutHandler for timeouts.
        """
        return TimeoutHandler.run_with_timeout(func, item, timeout=timeout)
    
    @classmethod
    def map(cls,
           func: Callable[[T], R],
           items: Iterable[T],
           max_workers: Optional[int] = None,
           timeout: Optional[int] = None,
           show_progress: bool = True,
           desc: str = "Processing",
           error_value: Any = None,
           context: str = "spawn",
           log_level: str = "INFO") -> List[R]:
        """
        Apply a function to each item in an iterable in parallel using processes.
        
        Args:
            func: Function to apply to each item
            items: Iterable of items to process
            max_workers: Maximum number of processes (None for auto-detection)
            timeout: Timeout in seconds for each task (None for no timeout)
            show_progress: Whether to show progress bar
            desc: Description for the progress bar
            error_value: Value to return for failed items (default None)
            context: Multiprocessing context ('spawn', 'fork', or 'forkserver')
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            
        Returns:
            List of results in the same order as input items
        """
        items = list(items)  # Convert to list for tqdm and indexing
        results = [error_value] * len(items)
        
        # Set up logging
        log_method = getattr(logger, log_level.lower(), logger.info)
        
        log_method(f"Starting process pool execution with {len(items)} items")
        if max_workers:
            log_method(f"Using {max_workers} worker processes with {context} context")
        else:
            log_method(f"Using automatic worker count with {context} context")
        
        # Use ProcessPoolExecutor with the specified context
        # For better control and more consistent behavior
        mp_context = multiprocessing.get_context(context)
        
        # Worker function that handles timeouts and logging
        def process_item(item_with_idx):
            idx, item = item_with_idx
            start_time = time.time()
            
            try:
                # Use the timeout handler to execute the function
                result = cls._worker_with_timeout(func, item, timeout)
                elapsed = time.time() - start_time
                
                if result is None and timeout is not None:
                    # Likely a timeout occurred
                    return idx, error_value, f"Task timed out after {timeout}s"
                
                return idx, result, None
            except Exception as e:
                elapsed = time.time() - start_time
                return idx, error_value, f"Error: {str(e)} ({elapsed:.2f}s)"
            
        # Create a ProcessPoolExecutor and submit all tasks
        with ProcessPoolExecutor(
            max_workers=max_workers, 
            mp_context=mp_context
        ) as executor:
            # Submit all tasks
            futures = [executor.submit(process_item, (i, item)) for i, item in enumerate(items)]
            
            # Process results with progress tracking
            if show_progress:
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(items),
                    desc=desc
                ):
                    try:
                        idx, result, error_msg = future.result()
                        if error_msg:
                            logger.error(f"Item {idx}: {error_msg}")
                        results[idx] = result
                    except Exception as e:
                        logger.error(f"Unexpected executor error: {str(e)}")
            else:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        idx, result, error_msg = future.result()
                        if error_msg:
                            logger.error(f"Item {idx}: {error_msg}")
                        results[idx] = result
                    except Exception as e:
                        logger.error(f"Unexpected executor error: {str(e)}")
        
        success_count = sum(1 for r in results if r is not error_value)
        log_method(f"Completed {len(items)} tasks: {success_count} succeeded, {len(items) - success_count} failed")
        
        return results
    
    @classmethod
    def run(cls,
           func: Callable[..., R],
           items: List[Any],
           max_workers: Optional[int] = None,
           timeout: Optional[int] = None,
           show_progress: bool = True,
           desc: str = "Processing",
           error_value: Any = None,
           context: str = "spawn",
           log_level: str = "INFO") -> List[R]:
        """
        Run a function on each item in a list in parallel using processes.
        
        Args:
            func: Function to call on each item
            items: List of items to process (each item is passed as the sole argument to func)
            max_workers: Maximum number of processes (None for auto-detection)
            timeout: Timeout in seconds for each task (None for no timeout)
            show_progress: Whether to show progress bar
            desc: Description for the progress bar
            error_value: Value to return for failed items (default None)
            context: Multiprocessing context ('spawn', 'fork', or 'forkserver')
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            
        Returns:
            List of results in the same order as input items
        """
        return cls.map(func, items, max_workers, timeout, show_progress, desc, error_value, context, log_level)


def configure_logging(level: str = "INFO", 
                  format: Optional[str] = None,
                  sink: Optional[Any] = None) -> None:
    """
    Configure the logger for parallel utilities.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        format: Log format string (only used with loguru)
        sink: Output destination (only used with loguru)
    """
    # Check if we're using the real loguru or the compatibility layer
    if hasattr(logger, 'configure') and callable(logger.configure):
        # We have loguru
        config = {"levels": [{"name": level, "level": level}]}
        if format:
            config["format"] = format
        if sink:
            config["handlers"] = [{"sink": sink, "level": level}]
        logger.configure(**config)
    else:
        # Using compatibility layer
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.configure(level=numeric_level)


def parallel_map(func: Callable[[T], R], 
                items: Iterable[T], 
                method: str = "thread",
                max_workers: Optional[int] = None,
                timeout: Optional[int] = None,
                show_progress: bool = True,
                desc: str = "Processing",
                error_value: Any = None,
                context: str = "spawn",
                log_level: str = "INFO") -> List[R]:
    """
    Convenience function for parallel processing with any method.
    
    Args:
        func: Function to apply to each item
        items: Iterable of items to process
        method: Parallelization method ('thread', 'process', or 'async')
        max_workers: Maximum number of workers (None for auto-detection)
        timeout: Timeout in seconds for each task (None for no timeout)
        show_progress: Whether to show progress bar
        desc: Description for the progress bar
        error_value: Value to return for failed items (default None)
        context: Multiprocessing context (only used for 'process' method)
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        
    Returns:
        List of results in the same order as input items
    """
    if method == "thread":
        return ThreadPool.map(func, items, max_workers, timeout, show_progress, desc, error_value, log_level)
    elif method == "process":
        return ProcessPool.map(func, items, max_workers, timeout, show_progress, desc, error_value, context, log_level)
    elif method == "async":
        # Convert to simple process_fn for AsyncApiCaller
        def process_fn(item):
            return func(item)
        
        # Set up logging for async mode
        log_method = getattr(logger, log_level.lower(), logger.info)
        log_method(f"Starting async pool execution with {len(list(items))} items")
        
        # Use AsyncApiCaller with appropriate parameters
        items_list = list(items)
        return AsyncApiCaller.run(
            items=items_list,
            process_fn=process_fn,
            output_file=None,
            max_workers=max_workers,
            timeout=timeout if timeout is not None else 60,
            save_interval=10 if show_progress else 1000
        )
    else:
        raise ValueError(f"Unknown parallelization method: {method}. Choose 'thread', 'process', or 'async'.")