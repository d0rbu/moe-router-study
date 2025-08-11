from dataclasses import dataclass, field
import re

import huggingface_hub  # import module so tests can patch huggingface_hub.list_repo_refs


@dataclass
class Checkpoint:
    step: int
    num_tokens: int | None
    model_config: "ModelConfig"

    def __str__(self):
        # Handle missing format gracefully
        if self.model_config.revision_format is None:
            return f"step{self.step}"
        return self.model_config.revision_format.format(self.step, self.num_tokens)


@dataclass
class ModelConfig:
    hf_name: str
    branch_regex: re.Pattern | str | None = None
    revision_format: str | None = None
    tokenizer_has_padding_token: bool = True
    checkpoints: list[Checkpoint] = field(default_factory=list)
    # New: avoid network fetch at import-time for globals like MODELS
    eager_fetch: bool = True

    def __post_init__(self):
        if self.branch_regex is None or self.revision_format is None:
            self.checkpoints = []
            return

        # Accept both str and compiled patterns
        if isinstance(self.branch_regex, str):
            self.branch_regex = re.compile(self.branch_regex)

        # Skip fetching branches if eager_fetch is disabled
        if not self.eager_fetch:
            self.checkpoints = []
            return

        refs = huggingface_hub.list_repo_refs(self.hf_name)
        self.all_branches = [branch.name for branch in refs.branches]

        checkpoints: list[Checkpoint] = []
        for branch in self.all_branches:
            match = re.match(self.branch_regex, branch)
            if not match:
                continue

            groups = match.groups()
            if len(groups) == 2:
                step_str, num_tokens_str = groups
                num_tokens_val: int | None = int(num_tokens_str)
            elif len(groups) == 1:
                step_str = groups[0]
                num_tokens_val = None
            else:
                raise ValueError(f"Unexpected number of groups in branch {branch}")

            checkpoints.append(Checkpoint(int(step_str), num_tokens_val, self))

        self.checkpoints = sorted(checkpoints, key=lambda x: x.step)


MODELS: dict[str, ModelConfig] = {
    "olmoe": ModelConfig(
        hf_name="allenai/OLMoE-1B-7B-0924",
        branch_regex=re.compile(r"step(\d+)-tokens(\d+)B"),
        revision_format="step{}-tokens{}B",
        eager_fetch=False,  # avoid network during import
    ),
    "phimoe": ModelConfig(
        hf_name="microsoft/Phi-3.5-MoE-instruct",
    ),
    "q3_30b": ModelConfig(
        hf_name="Qwen/Qwen3-30B-A3B",
    ),
}
