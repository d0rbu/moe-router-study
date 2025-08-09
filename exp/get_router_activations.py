from itertools import batched
import os

import arguably
from nnterp import StandardizedTransformer
import torch as th
from tqdm import tqdm

from core.data import DATASETS
from core.device_map import CUSTOM_DEVICES
from core.model import MODELS
from exp import OUTPUT_DIR, ROUTER_LOGITS_DIR


def save_router_logits(
    router_logit_collection: list[th.Tensor], top_k: int, file_idx: int
) -> None:
    router_logits = th.cat(router_logit_collection, dim=0)
    output: dict[str, th.Tensor] = {
        "topk": top_k,
        "router_logits": router_logits,
    }
    th.save(output, os.path.join(ROUTER_LOGITS_DIR, f"{file_idx}.pt"))


@arguably.command()
def get_router_activations(
    model_name: str = "olmoe",
    dataset: str = "fw",
    batch_size: int = 4,
    device: str = "cpu",
    tokens_per_file: int = 10_000,
) -> None:
    model_config = MODELS.get(model_name, None)

    if model_config is None:
        raise ValueError(f"Model {model_name} not found")

    dataset_fn = DATASETS.get(dataset, None)

    if dataset_fn is None:
        raise ValueError(f"Dataset {dataset} not found")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ROUTER_LOGITS_DIR, exist_ok=True)

    device_map = CUSTOM_DEVICES.get(device, lambda: device)()

    model = StandardizedTransformer(
        model_config.hf_name, check_attn_probs_with_trace=False, device_map=device_map
    )
    router_layers: list[int] = model.layers_with_routers
    top_k: int = model.router_probabilities.get_top_k()

    with th.no_grad():
        router_logit_collection: list[th.Tensor] = []
        router_logit_collection_size: int = 0
        router_logit_collection_idx: int = 0

        pbar = tqdm(total=tokens_per_file, desc="Filling up file")

        for batch in batched(dataset_fn(), batch_size):
            encoded_batch = model.tokenizer(batch, padding=True, return_tensors="pt")

            router_logits = []

            with model.trace(batch) as tracer:
                for layer in router_layers:
                    padding_mask: th.Tensor = encoded_batch.attention_mask.bool().view(
                        -1
                    )  # (batch_size * seq_len)

                    router_logits.append(
                        model.routers_output[layer].cpu()[padding_mask].save()
                    )
                tracer.stop()

            router_logits = th.stack(router_logits, dim=1)
            router_logit_collection.append(router_logits)
            router_logit_collection_size += router_logits.shape[0]
            pbar.update(router_logits.shape[0])

            # save the router probabilities to a file
            if router_logit_collection_size >= tokens_per_file:
                save_router_logits(
                    router_logit_collection, top_k, router_logit_collection_idx
                )
                router_logit_collection_idx += 1
                router_logit_collection_size = 0
                router_logit_collection = []
                pbar.reset()

        # save the remaining router probabilities to a file
        if router_logit_collection_size > 0:
            save_router_logits(
                router_logit_collection, top_k, router_logit_collection_idx
            )


if __name__ == "__main__":
    arguably.run()
