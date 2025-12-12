"""Tests for core.data module."""

from unittest.mock import MagicMock, patch

import pytest

from core.data import (
    DATASETS,
    fineweb_10bt_text,
    get_dataset_fn,
    lmsys_chat_1m_text,
    smollm2_small,
    toy_text,
)


class TestToyText:
    """Test toy_text dataset function."""

    def test_toy_text_basic(self):
        """Test basic toy_text functionality."""
        result = toy_text()

        # Should return an iterable
        assert hasattr(result, "__iter__")

        # Convert to list to test content
        samples = list(result)

        # Should have expected samples
        assert len(samples) == 4
        assert "Tiny sample 1" in samples
        assert "Tiny sample 2" in samples
        assert "Tiny sample 3" in samples
        assert "Tiny sample 4" in samples

    def test_toy_text_with_tokenizer(self):
        """Test toy_text with tokenizer parameter (should be ignored)."""
        mock_tokenizer = MagicMock()
        result = toy_text(mock_tokenizer)

        samples = list(result)
        assert len(samples) == 4
        # Tokenizer should not be used
        mock_tokenizer.assert_not_called()

    def test_toy_text_deterministic(self):
        """Test that toy_text returns the same samples each time."""
        result1 = list(toy_text())
        result2 = list(toy_text())

        assert result1 == result2


class TestFineweb10btText:
    """Test fineweb_10bt_text dataset function."""

    @patch("core.data.load_dataset")
    def test_fineweb_10bt_text_calls_correct_dataset(self, mock_load_dataset):
        """Test that fineweb_10bt_text calls load_dataset with correct parameters."""
        # Mock dataset response - we only care about the call, not the return
        mock_load_dataset.return_value = {"text": []}

        fineweb_10bt_text()

        # Verify the function calls HuggingFace with the right dataset and config
        mock_load_dataset.assert_called_once_with(
            "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
        )

    @patch("core.data.load_dataset")
    def test_fineweb_10bt_text_extracts_text_field(self, mock_load_dataset):
        """Test that fineweb_10bt_text correctly extracts the 'text' field."""
        # Create a mock dataset that behaves like a real HF dataset
        mock_dataset = {
            "text": ["Sample 1", "Sample 2"],
            "url": ["http://example1.com", "http://example2.com"],
            "timestamp": ["2023-01-01", "2023-01-02"],
        }
        mock_load_dataset.return_value = mock_dataset

        result = fineweb_10bt_text()

        # Should extract only the text field, ignoring other fields
        assert result == ["Sample 1", "Sample 2"]
        assert "url" not in str(result)
        assert "timestamp" not in str(result)

    @patch("core.data.load_dataset")
    def test_fineweb_10bt_text_ignores_tokenizer(self, mock_load_dataset):
        """Test that fineweb_10bt_text ignores tokenizer parameter."""
        mock_load_dataset.return_value = {"text": ["Sample"]}

        mock_tokenizer = MagicMock()
        result = fineweb_10bt_text(mock_tokenizer)

        # The key test: tokenizer should never be called
        mock_tokenizer.assert_not_called()
        # Function should still work normally
        assert result == ["Sample"]


class TestSmollm2Small:
    """Test smollm2_small dataset function."""

    @patch("core.data.load_dataset")
    def test_smollm2_small_calls_correct_dataset(self, mock_load_dataset):
        """Test that smollm2_small calls load_dataset with correct parameters."""
        mock_load_dataset.return_value = {"text": []}

        smollm2_small()

        # Verify it calls the right dataset with 1% split
        mock_load_dataset.assert_called_once_with(
            "EleutherAI/SmolLM2-135M-10B", split="train[:1%]"
        )

    @patch("core.data.load_dataset")
    def test_smollm2_small_extracts_text_field(self, mock_load_dataset):
        """Test that smollm2_small correctly extracts the 'text' field."""
        # Mock a dataset with multiple fields like a real HF dataset
        mock_dataset = {
            "text": ["SmolLM sample 1", "SmolLM sample 2"],
            "meta": [{"source": "web"}, {"source": "books"}],
            "id": [12345, 67890],
        }
        mock_load_dataset.return_value = mock_dataset

        result = smollm2_small()

        # Should only return text content, not metadata
        assert result == ["SmolLM sample 1", "SmolLM sample 2"]
        assert all(isinstance(item, str) for item in result)

    @patch("core.data.load_dataset")
    def test_smollm2_small_ignores_tokenizer(self, mock_load_dataset):
        """Test that smollm2_small ignores tokenizer parameter."""
        mock_load_dataset.return_value = {"text": ["Sample"]}

        mock_tokenizer = MagicMock()
        result = smollm2_small(mock_tokenizer)

        # Tokenizer should never be called for this dataset
        mock_tokenizer.assert_not_called()
        assert result == ["Sample"]


class TestLmsysChatText:
    """Test lmsys_chat_1m_text dataset function."""

    def test_lmsys_chat_1m_text_conversation_formatting_logic(self):
        """Test the actual conversation formatting logic."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = (
            "User: Hello\nAssistant: Hi there!"
        )

        with patch("core.data.load_dataset") as mock_load_dataset:
            # Create a mock dataset that yields conversation data
            # In streaming mode, the function checks if ds is a dict
            mock_dataset = {
                "conversation": [
                    [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi there!"},
                    ]
                ]
            }
            mock_load_dataset.return_value = mock_dataset

            result = lmsys_chat_1m_text(mock_tokenizer)

            # Test that it's iterable and processes conversations
            assert hasattr(result, "__iter__")

            # Test that the tokenizer's apply_chat_template is actually called
            # This tests the real logic of the function
            list(result)  # Force iteration to trigger the formatting
            mock_tokenizer.apply_chat_template.assert_called()

    @patch("core.data.load_dataset")
    def test_lmsys_chat_1m_text_streaming_mode(self, mock_load_dataset):
        """Test lmsys_chat_1m_text in streaming mode."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted chat"

        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = [[{"role": "user", "content": "Hello"}]]
        mock_load_dataset.return_value = mock_dataset

        # Test streaming mode (default)
        lmsys_chat_1m_text(mock_tokenizer, streaming=True)

        mock_load_dataset.assert_called_once_with(
            "lmsys/lmsys-chat-1m", split="train", streaming=True
        )

    @patch("core.data.load_dataset")
    def test_lmsys_chat_1m_text_non_streaming_mode(self, mock_load_dataset):
        """Test lmsys_chat_1m_text in non-streaming mode."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted chat"

        # Mock dataset with length and select methods
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.__getitem__.return_value = [[{"role": "user", "content": "Hello"}]]
        mock_load_dataset.return_value = mock_dataset

        with patch("core.data.assert_type", return_value=mock_dataset):
            # Test non-streaming mode
            result = lmsys_chat_1m_text(
                mock_tokenizer, start_idx=10, stop_idx=20, streaming=False
            )

            # Consume the generator to trigger the select call
            list(result)

            mock_load_dataset.assert_called_once_with(
                "lmsys/lmsys-chat-1m", split="train", streaming=False
            )

            # Should call select with correct range
            mock_dataset.select.assert_called_once()

    @patch("core.data.load_dataset")
    def test_lmsys_chat_1m_text_streaming_with_indices_error(self, mock_load_dataset):
        """Test that streaming mode doesn't accept start/stop indices."""
        mock_tokenizer = MagicMock()

        with pytest.raises(AssertionError, match="Streaming mode does not support"):
            lmsys_chat_1m_text(
                mock_tokenizer, start_idx=10, stop_idx=20, streaming=True
            )

    @patch("core.data.load_dataset")
    def test_lmsys_chat_1m_text_validation_errors(self, mock_load_dataset):
        """Test validation errors in non-streaming mode."""
        mock_tokenizer = MagicMock()

        # Create a mock dataset that behaves like the real Dataset class
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_load_dataset.return_value = mock_dataset

        # Mock assert_type to return the mock dataset
        with patch("core.data.assert_type", return_value=mock_dataset):
            # Test negative indices
            with pytest.raises(
                AssertionError, match="start_idx and stop_idx to be non-negative"
            ):
                lmsys_chat_1m_text(
                    mock_tokenizer, start_idx=-1, stop_idx=10, streaming=False
                )

            # Test start_idx >= stop_idx
            with pytest.raises(
                AssertionError, match="start_idx must be less than stop_idx"
            ):
                lmsys_chat_1m_text(
                    mock_tokenizer, start_idx=20, stop_idx=10, streaming=False
                )

            # Test start_idx >= dataset length
            with pytest.raises(
                AssertionError, match="start_idx must be less than the length"
            ):
                lmsys_chat_1m_text(
                    mock_tokenizer, start_idx=150, stop_idx=200, streaming=False
                )

            # Test stop_idx > dataset length
            with pytest.raises(
                AssertionError, match="stop_idx must be less than or equal"
            ):
                lmsys_chat_1m_text(
                    mock_tokenizer, start_idx=10, stop_idx=150, streaming=False
                )

    @patch("core.data.load_dataset")
    def test_lmsys_chat_1m_text_stop_idx_zero(self, mock_load_dataset):
        """Test that stop_idx=0 uses dataset length."""
        mock_tokenizer = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.__getitem__.return_value = []
        mock_load_dataset.return_value = mock_dataset

        with patch("core.data.assert_type", return_value=mock_dataset):
            result = lmsys_chat_1m_text(
                mock_tokenizer,
                start_idx=10,
                stop_idx=0,  # Should use dataset length
                streaming=False,
            )

            # Consume the generator to trigger the select call
            list(result)

            # Should select from 10 to 100
            mock_dataset.select.assert_called_once()
            call_args = mock_dataset.select.call_args[0][0]
            assert call_args.start == 10
            assert call_args.stop == 100

    def test_lmsys_chat_1m_text_validates_string_output(self):
        """Test that the function validates tokenizer output is a string."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = 123  # Not a string

        with patch("core.data.load_dataset") as mock_load_dataset:
            # In streaming mode, the function checks if ds is a dict
            mock_dataset = {"conversation": [[{"role": "user", "content": "Hello"}]]}
            mock_load_dataset.return_value = mock_dataset

            result_iter = lmsys_chat_1m_text(mock_tokenizer)

            # This tests the actual validation logic in the function
            with pytest.raises(ValueError, match="Expected chat to be a string"):
                list(result_iter)

    def test_lmsys_chat_1m_text_conversation_structure_validation(self):
        """Test that conversations are passed correctly to the tokenizer."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted output"

        # Test conversation with specific structure
        test_conversation = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
        ]

        with patch("core.data.load_dataset") as mock_load_dataset:
            # In streaming mode, the function checks if ds is a dict
            mock_dataset = {"conversation": [test_conversation]}
            mock_load_dataset.return_value = mock_dataset

            result_iter = lmsys_chat_1m_text(mock_tokenizer)
            list(result_iter)  # Force iteration

            # Verify the tokenizer was called with the exact conversation structure
            mock_tokenizer.apply_chat_template.assert_called_with(
                test_conversation, tokenize=False
            )

    @patch("core.data.os.path.exists")
    @patch("core.data.load_dataset")
    def test_lmsys_chat_1m_text_local_path(self, mock_load_dataset, mock_exists):
        """Test loading from local path when it exists."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted"

        # Mock local path exists
        mock_exists.return_value = True

        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = []
        mock_load_dataset.return_value = mock_dataset

        with patch("core.data.DATASET_DIRNAME", "/mock/dataset/dir"):
            lmsys_chat_1m_text(mock_tokenizer)

            # Should load from local path
            expected_path = "/mock/dataset/dir/lmsys/lmsys-chat-1m"
            mock_load_dataset.assert_called_once_with(
                expected_path, split="train", streaming=True
            )


class TestDatasetsConstant:
    """Test DATASETS constant."""

    def test_datasets_structure(self):
        """Test that DATASETS has expected structure."""
        assert isinstance(DATASETS, dict)
        assert len(DATASETS) > 0

        for name, func in DATASETS.items():
            assert isinstance(name, str)
            assert callable(func)

    def test_datasets_content(self):
        """Test that expected datasets are present."""
        expected_datasets = ["fw", "toy", "lmsys", "smol"]

        for dataset_name in expected_datasets:
            assert dataset_name in DATASETS

    def test_datasets_mapping(self):
        """Test that dataset functions are correctly mapped."""
        assert DATASETS["fw"] is fineweb_10bt_text
        assert DATASETS["toy"] is toy_text
        assert DATASETS["lmsys"] is lmsys_chat_1m_text
        assert DATASETS["smol"] is smollm2_small


class TestGetDatasetFn:
    """Test get_dataset_fn function."""

    def test_get_dataset_fn_valid(self):
        """Test get_dataset_fn with valid dataset names."""
        for dataset_name in DATASETS:
            result = get_dataset_fn(dataset_name)
            assert callable(result)
            assert result is DATASETS[dataset_name]

    def test_get_dataset_fn_invalid(self):
        """Test get_dataset_fn with invalid dataset name."""
        with pytest.raises(ValueError, match="Dataset nonexistent not found"):
            get_dataset_fn("nonexistent")

    def test_get_dataset_fn_empty_string(self):
        """Test get_dataset_fn with empty string."""
        with pytest.raises(ValueError, match="Dataset  not found"):
            get_dataset_fn("")


class TestDatasetIntegration:
    """Integration tests for dataset functionality."""

    def test_toy_dataset_integration(self):
        """Test toy dataset can be used end-to-end without external dependencies."""
        dataset_fn = get_dataset_fn("toy")

        # Should work without tokenizer - this tests real functionality
        result = dataset_fn(None)  # type: ignore[arg-type]
        samples = list(result)

        # Test actual content and behavior
        assert len(samples) == 4
        assert all(isinstance(sample, str) for sample in samples)
        assert all("Tiny sample" in sample for sample in samples)

        # Test deterministic behavior
        result2 = list(dataset_fn(None))  # type: ignore[arg-type]
        assert samples == result2

    def test_dataset_function_signatures_compatibility(self):
        """Test that all dataset functions accept the same signature."""
        # Test that all functions can be called with (tokenizer=None)
        for dataset_name, dataset_fn in DATASETS.items():
            if dataset_name == "toy":
                # toy_text should work with None tokenizer
                result = dataset_fn(None)  # type: ignore[arg-type]
                assert hasattr(result, "__iter__")
            else:
                # Other functions should at least be callable
                assert callable(dataset_fn)
                # We can't test them without mocking external dependencies
                # but we can verify they have the expected signature
                import inspect

                sig = inspect.signature(dataset_fn)
                # Should accept a tokenizer parameter
                assert len(sig.parameters) >= 1

    def test_dataset_registry_completeness(self):
        """Test that the DATASETS registry contains expected functions."""
        expected_datasets = {"fw", "toy", "lmsys", "smol"}
        actual_datasets = set(DATASETS.keys())

        assert expected_datasets.issubset(actual_datasets), (
            f"Missing datasets: {expected_datasets - actual_datasets}"
        )

        # Test that get_dataset_fn works for all registered datasets
        for dataset_name in expected_datasets:
            fn = get_dataset_fn(dataset_name)
            assert callable(fn)
            assert fn is DATASETS[dataset_name]


class TestDatasetMainScript:
    """Test the main script functionality."""

    @patch("core.data.AutoTokenizer.from_pretrained")
    @patch("core.data.load_dataset")
    def test_main_script_structure(self, mock_load_dataset, mock_tokenizer):
        """Test that the main script structure is valid."""
        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        # Mock datasets
        mock_load_dataset.return_value = {"text": ["Sample"]}

        # The main script code should be importable and structured correctly
        # We can't easily test the actual execution, but we can verify the structure

        # Check that the main block exists and uses expected components
        import core.data

        # Verify the main components are available
        assert hasattr(core.data, "DATASETS")
        assert hasattr(core.data, "AutoTokenizer")
        assert hasattr(core.data, "tqdm")

    def test_dataset_error_handling(self):
        """Test error handling in dataset functions."""
        # Test that functions handle missing dependencies gracefully
        # This is more of a smoke test to ensure imports work

        # All dataset functions should be importable
        from core.data import (
            fineweb_10bt_text,
            lmsys_chat_1m_text,
            smollm2_small,
            toy_text,
        )

        # toy_text should always work
        result = toy_text()
        assert hasattr(result, "__iter__")

        # Other functions may require external dependencies
        # but should at least be callable
        assert callable(fineweb_10bt_text)
        assert callable(smollm2_small)
        assert callable(lmsys_chat_1m_text)
