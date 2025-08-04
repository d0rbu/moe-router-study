from nnterp import StandardizedTransformer
from core.data import DATASETS
from core.model import MODELS
from tqdm import tqdm
from itertools import batched
import arguably
import torch as th


@arguably.command()
def main(model: str = "olmoe", dataset: str = "fw", batchsize: int = 1) -> None:
    model_config = MODELS.get(model, None)

    if model_config is None:
        raise ValueError(f"Model {model} not found")

    dataset_fn = DATASETS.get(dataset, None)

    if dataset_fn is None:
        raise ValueError(f"Dataset {dataset} not found")

    model = StandardizedTransformer(model_config.hf_name, device_map="cpu")
    router_layers: list[int] = model.layers_with_routers
    top_k: int = model.router_probabilities.get_top_k()

    for batch in tqdm(batched(dataset_fn(), batchsize), desc="Processing dataset"):
        with model.trace(batch) as tracer:
            router_probs = [model.router_probabilities[layer] for layer in router_layers]
            router_probs = th.stack(router_probs, dim=0)
            router_probs = th.softmax(router_probs, dim=-1)
            router_probs = th.topk(router_probs, k=top_k, dim=-1)
            router_probs = router_probs.values
            router_probs = router_probs.cpu().numpy()
            router_probs = router_probs.tolist()
            router_probs = [router_probs[layer] for layer in router_layers]


if __name__ == "__main__":
    arguably.run()
