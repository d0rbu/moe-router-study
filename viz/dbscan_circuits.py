
import arguably
from sklearn.cluster import DBSCAN

from exp.activations import load_activations_and_topk


@arguably.command()
def cluster_circuits() -> None:
    activated_experts, top_k = load_activations_and_topk()

    # (B, L, E) -> (B, L * E)
    activated_experts = activated_experts.view(activated_experts.shape[0], -1).float()

    # cluster the expert activations with dbscan
    dbscan = DBSCAN(eps=0.1, min_samples=10)
    clusters = dbscan.fit_predict(activated_experts)
    print(clusters)


if __name__ == "__main__":
    arguably.run()
