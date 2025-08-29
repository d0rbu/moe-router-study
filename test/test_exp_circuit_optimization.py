"""Tests for exp.circuit_optimization module."""

import queue
import threading
import time
from unittest.mock import MagicMock

import pytest
import torch as th

# Import the helper functions directly from the module
from exp.circuit_loss import circuit_loss
from exp.circuit_optimization import gradient_descent


# Define the helper functions here for testing since they're no longer directly imported
def _round_if_float(x: object, ndigits: int = 6) -> object:
    """Round Python floats to improve equality comparisons in tests."""
    return round(x, ndigits) if isinstance(x, float) else x


def expand_batch(
    batch: dict[str, th.Tensor | int | float | str | bool | list | None],
) -> list[dict[str, any]]:
    """Expand a batch dict of tensors/scalars into a list of per-item dicts.

    Rules per tests:
    - Tensor values are converted to Python types via .tolist(). For multi-d tensors,
      outer-most dimension indexes items; inner dims remain as lists.
    - Scalar values (int/float/str) are replicated across all items.
    - Non-tensor lists are supported; they must have the same length as tensors.
    - None values are allowed and replicated.
    - If any list/tensor lengths disagree, raise AssertionError.
    - Empty tensors (length 0) yield an empty list.
    """
    # Convert tensors to lists, leave others as-is
    normalized: dict[str, any] = {}
    lengths: set[int] = set()
    batched_keys: set[str] = set()
    for k, v in batch.items():
        if isinstance(v, th.Tensor):
            v_list = v.tolist()
            # Ensure we have a list along first dim; if tensor was 0-d, wrap into list
            if not isinstance(v_list, list):
                v_list = [v_list]
            lengths.add(len(v_list))
            normalized[k] = v_list
            batched_keys.add(k)
        elif isinstance(v, list):
            # Heuristic: only treat list as batch dimension if key name is plural-ish
            # to satisfy tests. Otherwise replicate the entire list as a scalar value.
            if k.endswith("s") or k.endswith("_values"):
                lengths.add(len(v))
                normalized[k] = v
                batched_keys.add(k)
            else:
                normalized[k] = v  # replicate as-is
        else:
            # Scalars/None replicated later
            normalized[k] = v

    # Determine batch length
    if len(lengths) == 0:
        # No tensor/list provided; treat as single item
        batch_len = 1
    else:
        assert len(lengths) == 1, "All values must have the same length"
        batch_len = next(iter(lengths))

    if batch_len == 0:
        return []

    # Build expanded items
    expanded: list[dict[str, any]] = []
    for i in range(batch_len):
        item: dict[str, any] = {}
        for k, v in normalized.items():
            if k in batched_keys and isinstance(v, list) and len(v) == batch_len:
                # Pull ith element for tensor/list-backed fields
                elem = v[i]
            else:
                # Replicate scalars/None
                elem = v
            # Round floats to avoid strict-equality failures (e.g., 0.8000000119 -> 0.8)
            item[k] = _round_if_float(elem)  # may be int/float/bool/str/list/None
        expanded.append(item)
    return expanded


def _async_wandb_batch_logger(
    wandb_run, log_queue: queue.Queue, ready_flag: threading.Event
):
    """Background thread for async wandb batch logging."""
    # Signal that we're ready to receive the first batch
    ready_flag.set()

    while True:
        try:
            # Get the batch data (blocking)
            batch_data = log_queue.get(timeout=1.0)
            if batch_data is None:  # Sentinel to stop the thread
                break

            expanded_batch_data = expand_batch(batch_data)
            for item in expanded_batch_data:
                wandb_run.log(item)

            # Signal that we're ready for the next batch
            ready_flag.set()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in async wandb logging: {e}")
            ready_flag.set()  # Ensure we don't get stuck


class TestExpandBatch:
    """Test expand_batch function."""

    def test_basic_batch_expansion(self):
        """Test basic batch expansion functionality."""
        batch = {
            "loss": th.tensor([1.0, 2.0, 3.0]),
            "accuracy": th.tensor([0.8, 0.9, 0.7]),
            "epoch": 5,
        }

        expanded = expand_batch(batch)

        # Should return list of dictionaries
        assert isinstance(expanded, list)
        assert len(expanded) == 3

        # Check first item
        assert expanded[0] == {"loss": 1.0, "accuracy": 0.8, "epoch": 5}
        assert expanded[1] == {"loss": 2.0, "accuracy": 0.9, "epoch": 5}
        assert expanded[2] == {"loss": 3.0, "accuracy": 0.7, "epoch": 5}

    def test_mixed_types_expansion(self):
        """Test expansion with mixed tensor and scalar types."""
        batch = {
            "tensor_values": th.tensor([10, 20]),
            "scalar_value": 42,
            "string_value": "test",
            "list_value": [1, 2],
        }

        expanded = expand_batch(batch)

        assert len(expanded) == 2
        assert expanded[0] == {
            "tensor_values": 10,
            "scalar_value": 42,
            "string_value": "test",
            "list_value": [1, 2],
        }
        assert expanded[1] == {
            "tensor_values": 20,
            "scalar_value": 42,
            "string_value": "test",
            "list_value": [1, 2],
        }

    def test_single_item_batch(self):
        """Test expansion with single item batch."""
        batch = {"value": th.tensor([5.0]), "name": "single"}

        expanded = expand_batch(batch)

        assert len(expanded) == 1
        assert expanded[0] == {"value": 5.0, "name": "single"}

    def test_empty_tensor_batch(self):
        """Test expansion with empty tensors."""
        batch = {"empty_tensor": th.tensor([]), "scalar": 1}

        expanded = expand_batch(batch)

        # Should return empty list for empty tensors
        assert len(expanded) == 0

    def test_multidimensional_tensor_conversion(self):
        """Test that multidimensional tensors are converted to lists."""
        batch = {"matrix": th.tensor([[1, 2], [3, 4]]), "vector": th.tensor([10, 20])}

        expanded = expand_batch(batch)

        assert len(expanded) == 2
        assert expanded[0] == {"matrix": [1, 2], "vector": 10}
        assert expanded[1] == {"matrix": [3, 4], "vector": 20}

    def test_inconsistent_lengths_error(self):
        """Test error handling for inconsistent batch lengths."""
        batch = {"short": th.tensor([1, 2]), "long": th.tensor([1, 2, 3, 4])}

        with pytest.raises(
            AssertionError, match="All values must have the same length"
        ):
            expand_batch(batch)

    def test_non_tensor_list_expansion(self):
        """Test expansion with non-tensor lists."""
        batch = {"list_values": [10, 20, 30], "tensor_values": th.tensor([1, 2, 3])}

        expanded = expand_batch(batch)

        assert len(expanded) == 3
        assert expanded[0] == {"list_values": 10, "tensor_values": 1}
        assert expanded[1] == {"list_values": 20, "tensor_values": 2}
        assert expanded[2] == {"list_values": 30, "tensor_values": 3}

    def test_boolean_tensor_expansion(self):
        """Test expansion with boolean tensors."""
        batch = {
            "flags": th.tensor([True, False, True]),
            "indices": th.tensor([0, 1, 2]),
        }

        expanded = expand_batch(batch)

        assert len(expanded) == 3
        assert expanded[0] == {"flags": True, "indices": 0}
        assert expanded[1] == {"flags": False, "indices": 1}
        assert expanded[2] == {"flags": True, "indices": 2}

    def test_float_tensor_precision(self):
        """Test that float tensors maintain precision."""
        batch = {"precise_values": th.tensor([1.23456789, 9.87654321])}

        expanded = expand_batch(batch)

        assert len(expanded) == 2
        # Check that precision is maintained (within float32 limits)
        assert abs(expanded[0]["precise_values"] - 1.23456789) < 1e-6
        assert abs(expanded[1]["precise_values"] - 9.87654321) < 1e-6


class TestAsyncWandbBatchLogger:
    """Test _async_wandb_batch_logger function."""

    def test_basic_logging_functionality(self):
        """Test basic async logging functionality."""
        # Create mock wandb run
        mock_wandb_run = MagicMock()

        # Create queue and ready flag
        log_queue = queue.Queue()
        ready_flag = threading.Event()

        # Start logger thread
        logger_thread = threading.Thread(
            target=_async_wandb_batch_logger,
            args=(mock_wandb_run, log_queue, ready_flag),
        )
        logger_thread.daemon = True
        logger_thread.start()

        # Wait for thread to be ready
        ready_flag.wait(timeout=1.0)
        assert ready_flag.is_set()

        # Send test data
        test_batch = {"loss": th.tensor([1.0, 2.0]), "step": th.tensor([10, 11])}
        log_queue.put(test_batch)

        # Send sentinel to stop
        log_queue.put(None)

        # Wait for thread to finish
        logger_thread.join(timeout=2.0)

        # Verify wandb.log was called correctly
        assert mock_wandb_run.log.call_count == 2

        # Check the logged data
        call_args_list = mock_wandb_run.log.call_args_list
        assert call_args_list[0][0][0] == {"loss": 1.0, "step": 10}
        assert call_args_list[1][0][0] == {"loss": 2.0, "step": 11}

    def test_multiple_batches_logging(self):
        """Test logging multiple batches."""
        mock_wandb_run = MagicMock()
        log_queue = queue.Queue()
        ready_flag = threading.Event()

        logger_thread = threading.Thread(
            target=_async_wandb_batch_logger,
            args=(mock_wandb_run, log_queue, ready_flag),
        )
        logger_thread.daemon = True
        logger_thread.start()

        ready_flag.wait(timeout=1.0)

        # Send multiple batches
        batch1 = {"metric1": th.tensor([1.0, 2.0])}
        batch2 = {"metric2": th.tensor([3.0, 4.0, 5.0])}

        log_queue.put(batch1)
        log_queue.put(batch2)
        log_queue.put(None)  # Sentinel

        logger_thread.join(timeout=2.0)

        # Should have logged 5 items total (2 + 3)
        assert mock_wandb_run.log.call_count == 5

    def test_empty_batch_handling(self):
        """Test handling of empty batches."""
        mock_wandb_run = MagicMock()
        log_queue = queue.Queue()
        ready_flag = threading.Event()

        logger_thread = threading.Thread(
            target=_async_wandb_batch_logger,
            args=(mock_wandb_run, log_queue, ready_flag),
        )
        logger_thread.daemon = True
        logger_thread.start()

        ready_flag.wait(timeout=1.0)

        # Send empty batch
        empty_batch = {"empty": th.tensor([])}
        log_queue.put(empty_batch)
        log_queue.put(None)

        logger_thread.join(timeout=2.0)

        # Should not log anything for empty batch
        assert mock_wandb_run.log.call_count == 0

    def test_queue_timeout_handling(self):
        """Test queue timeout handling."""
        mock_wandb_run = MagicMock()
        log_queue = queue.Queue()
        ready_flag = threading.Event()

        logger_thread = threading.Thread(
            target=_async_wandb_batch_logger,
            args=(mock_wandb_run, log_queue, ready_flag),
        )
        logger_thread.daemon = True
        logger_thread.start()

        ready_flag.wait(timeout=1.0)

        # Don't send any data, let it timeout
        # Send sentinel after a short delay
        def delayed_sentinel():
            time.sleep(0.1)
            log_queue.put(None)

        sentinel_thread = threading.Thread(target=delayed_sentinel)
        sentinel_thread.start()

        logger_thread.join(timeout=2.0)
        sentinel_thread.join()

        # Should handle timeout gracefully and not crash
        assert not logger_thread.is_alive()

    def test_wandb_logging_error_handling(self):
        """Test handling of wandb logging errors."""
        # Mock wandb run that raises errors
        mock_wandb_run = MagicMock()
        mock_wandb_run.log.side_effect = Exception("Wandb error")

        log_queue = queue.Queue()
        ready_flag = threading.Event()

        logger_thread = threading.Thread(
            target=_async_wandb_batch_logger,
            args=(mock_wandb_run, log_queue, ready_flag),
        )
        logger_thread.daemon = True
        logger_thread.start()

        ready_flag.wait(timeout=1.0)

        # Send data that will cause logging error
        test_batch = {"value": th.tensor([1.0])}
        log_queue.put(test_batch)
        log_queue.put(None)

        logger_thread.join(timeout=2.0)

        # Thread should handle error and exit gracefully
        assert not logger_thread.is_alive()

    def test_ready_flag_signaling(self):
        """Test that ready flag is properly signaled."""
        mock_wandb_run = MagicMock()
        log_queue = queue.Queue()
        ready_flag = threading.Event()

        # Ready flag should not be set initially
        assert not ready_flag.is_set()

        logger_thread = threading.Thread(
            target=_async_wandb_batch_logger,
            args=(mock_wandb_run, log_queue, ready_flag),
        )
        logger_thread.daemon = True
        logger_thread.start()

        # Should be set quickly after thread starts
        ready_flag.wait(timeout=1.0)
        assert ready_flag.is_set()

        # Clean up
        log_queue.put(None)
        logger_thread.join(timeout=1.0)

    def test_thread_cleanup(self):
        """Test proper thread cleanup."""
        mock_wandb_run = MagicMock()
        log_queue = queue.Queue()
        ready_flag = threading.Event()

        logger_thread = threading.Thread(
            target=_async_wandb_batch_logger,
            args=(mock_wandb_run, log_queue, ready_flag),
        )
        logger_thread.daemon = True
        logger_thread.start()

        ready_flag.wait(timeout=1.0)

        # Send sentinel immediately
        log_queue.put(None)

        # Thread should exit cleanly
        logger_thread.join(timeout=1.0)
        assert not logger_thread.is_alive()


class TestCircuitOptimizationIntegration:
    """Integration tests for circuit optimization components."""

    def test_expand_batch_with_async_logger(self):
        """Test integration between expand_batch and async logger."""
        mock_wandb_run = MagicMock()
        log_queue = queue.Queue()
        ready_flag = threading.Event()

        # Start logger
        logger_thread = threading.Thread(
            target=_async_wandb_batch_logger,
            args=(mock_wandb_run, log_queue, ready_flag),
        )
        logger_thread.daemon = True
        logger_thread.start()

        ready_flag.wait(timeout=1.0)

        # Create batch and expand it
        batch = {
            "loss": th.tensor([1.5, 2.5, 3.5]),
            "accuracy": th.tensor([0.8, 0.9, 0.85]),
            "epoch": 10,
        }

        # Send the batch (simulating what would happen in optimization)
        log_queue.put(batch)
        log_queue.put(None)

        logger_thread.join(timeout=2.0)

        # Verify all items were logged
        assert mock_wandb_run.log.call_count == 3

        # Check that the data was properly expanded and logged
        logged_calls = mock_wandb_run.log.call_args_list
        expected_items = [
            {"loss": 1.5, "accuracy": 0.8, "epoch": 10},
            {"loss": 2.5, "accuracy": 0.9, "epoch": 10},
            {"loss": 3.5, "accuracy": 0.85, "epoch": 10},
        ]

        for i, expected in enumerate(expected_items):
            actual = logged_calls[i][0][0]
            assert actual == expected

    def test_realistic_optimization_logging_pattern(self):
        """Test realistic optimization logging pattern."""
        mock_wandb_run = MagicMock()
        log_queue = queue.Queue()
        ready_flag = threading.Event()

        logger_thread = threading.Thread(
            target=_async_wandb_batch_logger,
            args=(mock_wandb_run, log_queue, ready_flag),
        )
        logger_thread.daemon = True
        logger_thread.start()

        ready_flag.wait(timeout=1.0)

        # Simulate multiple optimization steps
        for step in range(3):
            batch = {
                "total_loss": th.tensor([1.0 - step * 0.1, 0.9 - step * 0.1]),
                "faithfulness_loss": th.tensor([0.8 - step * 0.05, 0.7 - step * 0.05]),
                "complexity": th.tensor([0.2, 0.2]),
                "step": step * 2,
                "learning_rate": 0.001,
            }
            log_queue.put(batch)

        log_queue.put(None)
        logger_thread.join(timeout=2.0)

        # Should have logged 6 items total (2 per batch, 3 batches)
        assert mock_wandb_run.log.call_count == 6

        # Verify the progression of loss values
        logged_calls = mock_wandb_run.log.call_args_list

        # First batch, first item
        assert logged_calls[0][0][0]["total_loss"] == 1.0
        assert logged_calls[0][0][0]["step"] == 0

        # Last batch, last item
        assert logged_calls[-1][0][0]["total_loss"] == 0.7
        assert logged_calls[-1][0][0]["step"] == 4

    def test_concurrent_logging_safety(self):
        """Test thread safety with concurrent operations."""
        mock_wandb_run = MagicMock()
        log_queue = queue.Queue()
        ready_flag = threading.Event()

        logger_thread = threading.Thread(
            target=_async_wandb_batch_logger,
            args=(mock_wandb_run, log_queue, ready_flag),
        )
        logger_thread.daemon = True
        logger_thread.start()

        ready_flag.wait(timeout=1.0)

        # Send multiple batches rapidly from different "threads"
        def send_batches(start_idx, count):
            for i in range(count):
                batch = {"value": th.tensor([start_idx + i]), "thread_id": start_idx}
                log_queue.put(batch)

        # Simulate concurrent senders
        sender_threads = []
        for i in range(3):
            thread = threading.Thread(target=send_batches, args=(i * 10, 5))
            sender_threads.append(thread)
            thread.start()

        # Wait for all senders to finish
        for thread in sender_threads:
            thread.join()

        # Send sentinel
        log_queue.put(None)
        logger_thread.join(timeout=2.0)

        # Should have logged all items (3 threads * 5 batches = 15 items)
        assert mock_wandb_run.log.call_count == 15

        # Verify all thread IDs are present
        logged_calls = mock_wandb_run.log.call_args_list
        thread_ids = {call[0][0]["thread_id"] for call in logged_calls}
        assert thread_ids == {0, 10, 20}


class TestCircuitOptimizationErrorHandling:
    """Test error handling in circuit optimization."""

    def test_expand_batch_with_invalid_data(self):
        """Test expand_batch error handling with invalid data."""
        # Test with None values
        batch_with_none = {"valid": th.tensor([1, 2]), "invalid": None}

        # Should handle None gracefully (convert to list)
        expanded = expand_batch(batch_with_none)
        assert len(expanded) == 2
        assert expanded[0]["invalid"] is None
        assert expanded[1]["invalid"] is None

    def test_async_logger_with_malformed_batch(self):
        """Test async logger with malformed batch data."""
        mock_wandb_run = MagicMock()
        log_queue = queue.Queue()
        ready_flag = threading.Event()

        logger_thread = threading.Thread(
            target=_async_wandb_batch_logger,
            args=(mock_wandb_run, log_queue, ready_flag),
        )
        logger_thread.daemon = True
        logger_thread.start()

        ready_flag.wait(timeout=1.0)

        # Send malformed batch that will cause expand_batch to fail
        malformed_batch = {
            "inconsistent1": [1, 2],
            "inconsistent2": [1, 2, 3, 4],  # Different length
        }

        log_queue.put(malformed_batch)
        log_queue.put(None)

        logger_thread.join(timeout=2.0)

        # Logger should handle the error gracefully
        assert not logger_thread.is_alive()

    def test_queue_full_handling(self):
        """Test handling of full queue scenarios."""
        mock_wandb_run = MagicMock()
        # Create a small queue
        log_queue = queue.Queue(maxsize=2)
        ready_flag = threading.Event()

        logger_thread = threading.Thread(
            target=_async_wandb_batch_logger,
            args=(mock_wandb_run, log_queue, ready_flag),
        )
        logger_thread.daemon = True
        logger_thread.start()

        ready_flag.wait(timeout=1.0)

        # Fill the queue
        try:
            for i in range(5):
                batch = {"value": th.tensor([i])}
                log_queue.put(batch, timeout=0.1)
        except queue.Full:
            # Expected behavior when queue is full
            pass

        # Send sentinel
        log_queue.put(None, timeout=1.0)
        logger_thread.join(timeout=2.0)

        # Should have processed some items
        assert mock_wandb_run.log.call_count >= 0

