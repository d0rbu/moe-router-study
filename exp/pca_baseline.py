"""
Create PCA baseline autoencoders using SAEBench's PCA implementation.

This script uses SAEBench's PCASAE and fit_PCA_gpu to create PCA baselines
that can be evaluated alongside trained SAEs.
"""

import json
from pathlib import Path
import sys
from typing import Any

import arguably
from loguru import logger
from sae_bench.custom_saes.pca_sae import PCASAE, fit_PCA, fit_PCA_gpu
from sae_bench.sae_bench_utils import dataset_utils
import torch as th
from transformer_lens import HookedTransformer

from core.dtype import get_dtype

# Get model config to determine d_model
from core.model import get_model_config
from exp import OUTPUT_DIR


def convert_pca_to_dictionary_format(
    pca: PCASAE,
    output_dir: Path,
    config: dict[str, Any],
) -> None:
    """
    Convert SAEBench PCA format to dictionary learning format.

    SAEBench uses W_enc, W_dec, mean.
    Dictionary learning uses encoder.weight, decoder.weight, bias.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert format: W_enc is (d_in, d_in), W_dec is (d_in, d_in)
    # Dictionary format: encoder.weight is (dict_size, activation_dim), decoder.weight is (activation_dim, dict_size)
    # bias is (activation_dim,)

    # W_enc in SAEBench is the PCA components (n_components, d_in) after transpose
    # W_dec is components.T (d_in, n_components)
    # But actually looking at the code: W_enc = components, W_dec = components.T

    # For dictionary format:
    # encoder.weight should be (dict_size, activation_dim) = (d_in, d_in) = W_enc.T
    # decoder.weight should be (activation_dim, dict_size) = (d_in, d_in) = W_dec
    # bias should be mean

    state_dict = {
        "encoder.weight": pca.W_enc.data.T.clone(),  # (d_in, d_in) -> (d_in, d_in) but transposed for encoder
        "encoder.bias": th.zeros(pca.cfg.d_in, device=pca.device, dtype=pca.dtype),
        "decoder.weight": pca.W_dec.data.clone(),  # (d_in, d_in)
        "bias": pca.mean.data.clone(),  # (d_in,)
    }

    # Save in dictionary learning format
    ae_path = output_dir / "ae.pt"
    th.save(state_dict, ae_path)
    logger.info(f"Saved autoencoder to {ae_path}")

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {config_path}")


@arguably.command()
def create_pca_baseline(
    *,
    model_name: str = "olmoe-i",
    dataset_name: str = "lmsys/lmsys-chat-1m",
    layer: int,
    submodule_name: str,
    context_length: int = 2048,
    num_tokens: int = 200_000_000,
    llm_batch_size: int = 128,
    pca_batch_size: int = 100_000,
    dtype: str = "bf16",
    log_level: str = "INFO",
    use_gpu: bool = True,
) -> None:
    """
    Create a PCA baseline autoencoder using SAEBench's implementation.

    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        layer: Layer index to create baseline for
        submodule_name: Submodule name (e.g., "mlp_output", "attn_output")
        context_length: Context length for tokenization
        num_tokens: Number of tokens to use for PCA fitting
        llm_batch_size: Batch size for model forward passes
        pca_batch_size: Batch size for PCA fitting
        dtype: Data type
        log_level: Logging level
        use_gpu: Whether to use GPU-accelerated PCA (requires cuml)
    """
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    # Output directory for PCA baseline
    output_dir = Path(OUTPUT_DIR) / "pca_baseline" / f"layer_{layer}_{submodule_name}"

    # Check if already exists
    if (output_dir / "ae.pt").exists() and (output_dir / "config.json").exists():
        logger.info(f"PCA baseline already exists at {output_dir}, skipping")
        return

    logger.info(f"Creating PCA baseline for layer {layer}, submodule {submodule_name}")
    logger.info(f"Output directory: {output_dir}")

    model_config = get_model_config(model_name)
    hf_name = model_config.hf_name

    # Load model using transformer_lens (required by SAEBench)
    logger.info(f"Loading model {hf_name}...")
    th_dtype = get_dtype(dtype)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained_no_processing(
        hf_name,
        device=device,
        dtype=th_dtype,
    )

    d_model = model.cfg.d_model
    logger.info(f"Model dimension: {d_model}")

    # Load and tokenize dataset
    logger.info(f"Loading and tokenizing dataset {dataset_name}...")
    tokens_BL = dataset_utils.load_and_tokenize_dataset(
        dataset_name,
        context_length,
        num_tokens,
        model.tokenizer,  # type: ignore
        column_name="conversation",
    )
    logger.info(f"Loaded {len(tokens_BL)} sequences")

    # Create PCA SAE
    logger.info("Initializing PCA SAE...")
    pca = PCASAE(
        d_in=d_model,
        model_name=hf_name,
        hook_layer=layer,
        device=device,
        dtype=th_dtype,
    )

    # Fit PCA
    logger.info("Fitting PCA...")
    if use_gpu:
        try:
            pca = fit_PCA_gpu(
                pca,
                model,
                tokens_BL,
                llm_batch_size,
                pca_batch_size,
            )
        except ImportError:
            logger.warning("cuML not available, falling back to CPU PCA")
            pca = fit_PCA(
                pca,
                model,
                tokens_BL,
                llm_batch_size,
                pca_batch_size,
            )
    else:
        pca = fit_PCA(
            pca,
            model,
            tokens_BL,
            llm_batch_size,
            pca_batch_size,
        )

    # Create config
    config = {
        "trainer": {
            "layer": layer,
            "submodule_name": submodule_name,
            "dict_size": d_model,  # Full PCA uses all components
            "activation_dim": d_model,
            "lm_name": hf_name,
            "dict_class": "PCAAutoEncoder",
        },
        "pca_baseline": True,
    }

    # Convert and save
    logger.info("Converting to dictionary format and saving...")
    convert_pca_to_dictionary_format(
        pca=pca,
        output_dir=output_dir,
        config=config,
    )

    logger.info(f"✅ PCA baseline created at {output_dir}")


if __name__ == "__main__":
    arguably.run()
