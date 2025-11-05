import asyncio
import concurrent.futures
import traceback

from loguru import logger


def handle_exceptions(task: asyncio.Task) -> None:
    """
    Handle exceptions from async tasks by logging them and stopping the event loop.

    Args:
        task: The asyncio task to check for exceptions
    """
    exception = task.exception()
    if exception is None:
        logger.trace(f"[worker {task.get_name()}]: No exception")
        return

    traceback_lines = traceback.format_tb(exception.__traceback__)
    traceback_str = "".join(traceback_lines)
    exception_str = str(exception)
    logger.exception(f"[worker {task.get_name()}]:\n{traceback_str}{exception_str}")
    # throw a tantrum and fuck up everything
    asyncio.get_running_loop().stop()


def handle_future_exceptions(future: concurrent.futures.Future) -> None:
    """
    Handle exceptions from futures by logging them.

    Args:
        future: The concurrent.futures.Future to check for exceptions
    """
    if not future.done():
        return
    try:
        exception = future.exception(timeout=0)
        if exception is not None:
            tb_str = "".join(traceback.format_tb(exception.__traceback__))
            logger.exception(f"[worker]:\n{tb_str}{exception}")
            raise exception
    except concurrent.futures.TimeoutError:
        pass
