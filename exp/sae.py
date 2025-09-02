from dataclasses import dataclass

# Commented out imports that are not available
# from dictionary_learning.trainers.matryoshka import (
#     MatryoshkaBatchTopKSAE,
#     MatryoshkaBatchTopKTrainer,
# )
# from dictionary_learning.trainers.top_k import BatchTopKSAE, BatchTopKTrainer
# from dictionary_learning.trainers.trainer import SAETrainer


@dataclass
class Architecture:
    """Architecture for SAE."""

    n_components: int
    l1_coefficient: float
    lr: float
    batch_size: int
    n_steps: int
    seed: int = 0


# Define architectures
ARCHITECTURES = {
    "small": Architecture(
        n_components=256,
        l1_coefficient=0.001,
        lr=0.001,
        batch_size=4096,
        n_steps=20000,
    ),
    "large": Architecture(
        n_components=512,
        l1_coefficient=0.001,
        lr=0.001,
        batch_size=4096,
        n_steps=20000,
    ),
}

# Commented out training configuration
# def trainSAE(
#     _dataset_name: str,
#     _layer_idx: int,
#     _name: str,
#     architecture: str = "small",
#     device: str = "cuda",
# ) -> None:
#     """Train an SAE on activations."""
#     arch = ARCHITECTURES[architecture]
#
#     # Create SAE
#     sae = BatchTopKSAE(
#         n_input_features=4096,
#         n_components=arch.n_components,
#         l1_coefficient=arch.l1_coefficient,
#         device=device,
#     )
#
#     # Create trainer
#     trainer = BatchTopKTrainer(
#         sae=sae,
#         lr=arch.lr,
#         batch_size=arch.batch_size,
#         n_steps=arch.n_steps,
#         seed=arch.seed,
#         device=device,
#     )
#
#     # Train SAE
#     trainer.train()
#
#     # Save SAE
#     trainer.save(f"{WEIGHT_DIR}/sae_{name}.pt")
