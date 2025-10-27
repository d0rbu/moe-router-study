import asyncio
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
    # Stop the event loop gracefully instead of closing it while running
    loop = asyncio.get_running_loop()
    loop.stop()
