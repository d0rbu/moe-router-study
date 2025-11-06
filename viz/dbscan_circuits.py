from typing import Optional

import arguably
from sklearn.cluster import DBSCAN
import torch as th

from exp.activations import load_activations_and_topk


class Node:
    def __init__(
        self,
        num_children: int,
        max_depth: int,
        depth: int = 0,
        activated_experts: th.Tensor | None = None,
        children: list[Optional["Node"]] | None = None,
        count: int | None = None,
    ):
        if children is None:
            children: list[None] = [None] * num_children

        if len(children) > num_children:
            raise ValueError(
                f"Node must have less than num_children {num_children} children: {children}"
            )
        elif len(children) == num_children:
            self.children = children
        else:
            self.children: list[Node | None] = children + [None] * (
                num_children - len(children)
            )

        self.num_children = num_children
        self.max_depth = max_depth
        self.depth = depth
        self.leaf_count = count
        self.activated_experts = activated_experts

    @property
    def count(self) -> int:
        if self.leaf_count:
            return self.leaf_count

        return sum(child.count for child in self.children)

    @count.setter
    def count(self, value: int) -> None:
        assert all(child is None for child in self.children), (
            f"Node {self} must have no children"
        )

        self.leaf_count = value

    def is_leaf(self) -> bool:
        return self.leaf_count is not None

    def upsert_child(self, child_idx: int, count: int | None = None) -> bool:
        assert not self.is_leaf(), f"Leaf node {self} cannot have children"

        if self.children[child_idx] is None:
            self.children[child_idx] = Node(
                self.num_children, self.max_depth, depth=self.depth + 1, count=count
            )
            return True

        if count is not None:
            self.children[child_idx].count += count

        return False

    def validate(self, child_idx: int, layer: int = 0) -> None:
        assert layer <= self.max_depth, (
            f"Node layer {layer} must be less than or equal to max_depth {self.max_depth}"
        )
        assert child_idx < self.num_children, (
            f"Node child {child_idx} must be less than num_children {self.num_children}"
        )

        # if we are a leaf node
        if self.is_leaf():
            assert len(self.children) == 0, f"Leaf node {self} must have no children"
            return

        # if we are not a leaf node
        assert len(self.children) > 0, f"Non-leaf node {self} must have children"

        for child_idx, child in enumerate(self.children):
            child.validate(child_idx, layer + 1)

    def __str__(self) -> str:
        return f"Node(layer={self.layer}, children={self.children})"


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
