"""Tests for exp.expert_importance module."""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch as th


class MockStandardizedTransformer:
    """Mock for StandardizedTransformer class."""

    def __init__(self, *args, **kwargs):
        self.layers_with_routers = [0]
        self.routers = {}
        self.attentions = {}
        self.mlps = {}

        # Setup mock router weights
        for layer_idx in self.layers_with_routers:
            # Router weights: (num_experts, hidden_size)
            self.routers[layer_idx] = MagicMock()
            self.routers[layer_idx].weight = th.randn(2, 16)

            # Attention weights
            self.attentions[layer_idx] = MagicMock()
            self.attentions[layer_idx].q_proj = MagicMock()
            self.attentions[layer_idx].q_proj.weight = th.randn(16, 16)
            self.attentions[layer_idx].k_proj = MagicMock()
            self.attentions[layer_idx].k_proj.weight = th.randn(16, 16)
            self.attentions[layer_idx].o_proj = MagicMock()
            self.attentions[layer_idx].o_proj.weight = th.randn(16, 16)

            # MLP experts
            self.mlps[layer_idx] = MagicMock()
            self.mlps[layer_idx].experts = []
            for _ in range(2):  # 2 experts
                expert = MagicMock()
                expert.up_proj = MagicMock()
                expert.up_proj.weight = th.randn(16, 16)  # (Dmlp, D)
                expert.gate_proj = MagicMock()
                expert.gate_proj.weight = th.randn(16, 16)  # (Dmlp, D)
                expert.down_proj = MagicMock()
                expert.down_proj.weight = th.randn(16, 16)  # (D, Dmlp)
                self.mlps[layer_idx].experts.append(expert)


class TestExpertImportance:
    """Test expert_importance function."""

    def test_expert_importance_basic(self, temp_dir, monkeypatch):
        """Test basic functionality of expert_importance."""
        # Mock StandardizedTransformer
        mock_transformer = MockStandardizedTransformer()

        # Mock MODELS dictionary
        mock_models = {
            "test_model": MagicMock(
                hf_name="test/model",
                checkpoints=[1000, 2000, 3000],
            )
        }

        # Set up experiment directory
        experiment_dir = os.path.join(str(temp_dir), "test_experiment")
        os.makedirs(experiment_dir, exist_ok=True)

        # Set up expert importance directory
        expert_importance_dir = os.path.join(experiment_dir, "expert_importance")
        os.makedirs(expert_importance_dir, exist_ok=True)

        with (
            patch("exp.get_experiment_dir", return_value=experiment_dir),
            patch("exp.expert_importance.MODELS", mock_models),
            patch(
                "exp.expert_importance.StandardizedTransformer",
                return_value=mock_transformer,
            ),
        ):
            # Import here to avoid module-level binding issues
            from exp.expert_importance import expert_importance

            # Run the function
            expert_importance(
                model_name="test_model",
                checkpoint_idx=0,
                device="cpu",
                experiment_name="test_experiment",
            )

            # Check that output file was created
            output_file = os.path.join(expert_importance_dir, "all.pt")
            assert os.path.exists(output_file)

            # Load and verify the output
            entries = th.load(output_file)

            # Check that we have entries
            assert len(entries) > 0
            assert isinstance(entries, list)

            # Check structure of entries
            for entry in entries:
                assert isinstance(entry, dict)
                assert "model_name" in entry
                assert "checkpoint_idx" in entry
                assert "revision" in entry
                assert "base_layer_idx" in entry
                assert "derived_layer_idx" in entry
                assert "base_expert_idx" in entry
                assert "component" in entry
                assert "role" in entry
                assert "importance_vector" in entry
                assert "l2" in entry
                assert "param_type" in entry

                # Check that importance_vector is a tensor
                assert isinstance(entry["importance_vector"], th.Tensor)

                # Check that l2 is a float
                assert isinstance(entry["l2"], float)

                # Check that role is either "reader" or "writer"
                assert entry["role"] in ["reader", "writer"]

                # Check that param_type is either "moe" or "attn"
                assert entry["param_type"] in ["moe", "attn"]

                # Check that the appropriate path fields exist
                if entry["param_type"] == "moe":
                    assert "derived_expert_idx" in entry
                    assert "base_param_path" in entry
                    assert "derived_param_path" in entry
                else:  # attn
                    assert "base_param_path" in entry

    def test_expert_importance_invalid_model(self):
        """Test expert_importance with invalid model name."""
        with pytest.raises(ValueError, match="Model .* not found"):
            # Import here to avoid module-level binding issues
            from exp.expert_importance import expert_importance

            expert_importance(model_name="nonexistent_model")

    @pytest.mark.xfail(
        reason="Test passes individually but fails when run with other tests"
    )
    def test_expert_importance_calculation(self, temp_dir, monkeypatch):
        """Test that importance calculations are correct."""
        # Create a simplified mock with known values for verification
        mock_transformer = MagicMock()
        mock_transformer.layers_with_routers = [0]

        # Create router weight with known values
        router_weight = th.tensor([[1.0, 0.0], [0.0, 1.0]])  # 2 experts, 2-dim hidden
        mock_router = MagicMock()
        mock_router.weight = router_weight
        mock_transformer.routers = {0: mock_router}

        # Create attention weights with known values
        q_weight = th.tensor([[2.0, 0.0], [0.0, 2.0]])
        k_weight = th.tensor([[3.0, 0.0], [0.0, 3.0]])
        o_weight = th.tensor([[4.0, 0.0], [0.0, 4.0]])

        mock_q_proj = MagicMock()
        mock_q_proj.weight = q_weight
        mock_k_proj = MagicMock()
        mock_k_proj.weight = k_weight
        mock_o_proj = MagicMock()
        mock_o_proj.weight = o_weight

        mock_attention = MagicMock()
        mock_attention.q_proj = mock_q_proj
        mock_attention.k_proj = mock_k_proj
        mock_attention.o_proj = mock_o_proj
        mock_transformer.attentions = {0: mock_attention}

        # Create MLP expert weights with known values
        up_weight_1 = th.tensor([[5.0, 0.0], [0.0, 0.0]])
        gate_weight_1 = th.tensor([[6.0, 0.0], [0.0, 0.0]])
        down_weight_1 = th.tensor([[7.0, 0.0], [0.0, 0.0]])

        up_weight_2 = th.tensor([[0.0, 8.0], [0.0, 0.0]])
        gate_weight_2 = th.tensor([[0.0, 9.0], [0.0, 0.0]])
        down_weight_2 = th.tensor([[0.0, 0.0], [10.0, 0.0]])

        expert1 = MagicMock()
        expert1.up_proj = MagicMock()
        expert1.up_proj.weight = up_weight_1
        expert1.gate_proj = MagicMock()
        expert1.gate_proj.weight = gate_weight_1
        expert1.down_proj = MagicMock()
        expert1.down_proj.weight = down_weight_1

        expert2 = MagicMock()
        expert2.up_proj = MagicMock()
        expert2.up_proj.weight = up_weight_2
        expert2.gate_proj = MagicMock()
        expert2.gate_proj.weight = gate_weight_2
        expert2.down_proj = MagicMock()
        expert2.down_proj.weight = down_weight_2

        mock_mlp = MagicMock()
        mock_mlp.experts = [expert1, expert2]
        mock_transformer.mlps = {0: mock_mlp}

        # Mock MODELS dictionary
        mock_models = {
            "test_model": MagicMock(
                hf_name="test/model",
                checkpoints=[1000],
            )
        }

        # Set up experiment directory
        experiment_dir = os.path.join(str(temp_dir), "test_experiment")
        os.makedirs(experiment_dir, exist_ok=True)

        # Set up expert importance directory
        expert_importance_dir = os.path.join(experiment_dir, "expert_importance")
        os.makedirs(expert_importance_dir, exist_ok=True)

        with (
            patch("exp.get_experiment_dir", return_value=experiment_dir),
            patch("exp.expert_importance.MODELS", mock_models),
            patch(
                "exp.expert_importance.StandardizedTransformer",
                return_value=mock_transformer,
            ),
        ):
            # Import here to avoid module-level binding issues
            from exp.expert_importance import expert_importance

            # Run the function
            expert_importance(
                model_name="test_model", device="cpu", experiment_name="test_experiment"
            )

            # Load and verify the output
            output_file = os.path.join(expert_importance_dir, "all.pt")
            entries = th.load(output_file)

            # Find specific entries to check calculations
            for entry in entries:
                # Check MoE entries
                if (
                    entry.get("param_type") == "moe"
                    and entry.get("base_expert_idx") == 0
                    and entry.get("derived_expert_idx") == 0
                    and entry.get("component") == "mlp.up_proj"
                ):
                    # Expert 0, up_proj should have importance vector [5.0, 0.0]
                    assert th.allclose(
                        entry["importance_vector"], th.tensor([5.0, 0.0])
                    )
                    # L2 norm should be 5.0
                    assert pytest.approx(entry["l2"]) == 5.0

                # Check attention entries
                if (
                    entry.get("param_type") == "attn"
                    and entry.get("base_expert_idx") == 1
                    and entry.get("component") == "attn.q_proj"
                ):
                    # Expert 1, q_proj should have importance vector [0.0, 2.0]
                    assert th.allclose(
                        entry["importance_vector"], th.tensor([0.0, 2.0])
                    )
                    # L2 norm should be 2.0
                    assert pytest.approx(entry["l2"]) == 2.0

    def test_expert_importance_with_revision(self, temp_dir, monkeypatch):
        """Test expert_importance with specific revision."""
        # Mock StandardizedTransformer
        mock_transformer = MockStandardizedTransformer()

        # Mock MODELS dictionary
        mock_models = {
            "test_model": MagicMock(
                hf_name="test/model",
                checkpoints=["step1000", "step2000"],
            )
        }

        # Set up experiment directory
        experiment_dir = os.path.join(str(temp_dir), "test_experiment")
        os.makedirs(experiment_dir, exist_ok=True)

        # Set up expert importance directory
        expert_importance_dir = os.path.join(experiment_dir, "expert_importance")
        os.makedirs(expert_importance_dir, exist_ok=True)

        with (
            patch("exp.get_experiment_dir", return_value=experiment_dir),
            patch("exp.expert_importance.MODELS", mock_models),
            patch(
                "exp.expert_importance.StandardizedTransformer",
                return_value=mock_transformer,
            ) as mock_transformer_cls,
        ):
            # Import here to avoid module-level binding issues
            from exp.expert_importance import expert_importance

            # Run the function with checkpoint_idx
            expert_importance(
                model_name="test_model",
                checkpoint_idx=1,
                device="cpu",
                experiment_name="test_experiment",
            )

            # Check that StandardizedTransformer was called with correct revision
            mock_transformer_cls.assert_called_once()
            _, kwargs = mock_transformer_cls.call_args
            assert kwargs["revision"] == "step2000"
