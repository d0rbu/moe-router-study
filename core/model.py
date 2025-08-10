from dataclasses import dataclass, field
import re

import huggingface_hub  # import module so it can be patched in tests


@dataclass
class Checkpoint:
    step: int
    num_tokens: int | None
    model_config: "ModelConfig"

    def __str__(self):
        # Handle None revision_format gracefully
        if not self.model_config.revision_format:
            return (
                f"step{self.step}"
                if self.num_tokens is None
                else f"step{self.step}-tokens{self.num_tokens}B"
            )
        return self.model_config.revision_format.format(self.step, self.num_tokens)


@dataclass
class ModelConfig:
    hf_name: str
    branch_regex: re.Pattern | None = None
    revision_format: str | None = None
    tokenizer_has_padding_token: bool = True
    checkpoints: list[Checkpoint] = field(default_factory=list)

    def __post_init__(self):
        if self.branch_regex is None or self.revision_format is None:
            self.checkpoints = []
            return

        # Accept either a precompiled Pattern or a string
        if isinstance(self.branch_regex, str):
            self.branch_regex = re.compile(self.branch_regex)

        refs = huggingface_hub.list_repo_refs(self.hf_name)
        self.all_branches = [branch.name for branch in refs.branches]

        checkpoints: list[Checkpoint] = []
        for branch in self.all_branches:
            match = re.match(self.branch_regex, branch)
            if not match:
                continue

            groups = match.groups()
            if len(groups) == 2:
                step, num_tokens = groups
                ckpt = Checkpoint(int(step), int(num_tokens), self)
            elif len(groups) == 1:
                step = groups[0]
                ckpt = Checkpoint(int(step), None, self)
            else:
                raise ValueError(f"Unexpected number of groups in branch {branch}")

            checkpoints.append(ckpt)

        self.checkpoints = sorted(checkpoints, key=lambda x: x.step)


MODELS: dict[str, ModelConfig] = {
    "olmoe": ModelConfig(
        hf_name="allenai/OLMoE-1B-7B-0924",
        branch_regex=re.compile(r"step(\d+)-tokens(\d+)B"),
        revision_format="step{}-tokens{}B",
    ),
    "phimoe": ModelConfig(
        hf_name="microsoft/Phi-3.5-MoE-instruct",
    ),
    "q3_30b": ModelConfig(
        hf_name="Qwen/Qwen3-30B-A3B",
    ),
}
