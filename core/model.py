from dataclasses import dataclass, field
import re

import huggingface_hub  # import module so tests can patch huggingface_hub.list_repo_refs


@dataclass
class Checkpoint:
    step: int
    num_tokens: int | None
    model_config: "ModelConfig"

    def __str__(self):
        # Fail explicitly if revision_format is None
        if self.model_config.revision_format is None:
            raise ValueError(
                "ModelConfig.revision_format is required for string representation"
            )
        return self.model_config.revision_format.format(self.step, self.num_tokens)


@dataclass
class ModelConfig:
    hf_name: str
    branch_regex: re.Pattern | str | None = None
    revision_format: str | None = None
    tokenizer_has_padding_token: bool = True
    checkpoints: list[Checkpoint] = field(default_factory=list)
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

    def get_checkpoint(
        self, step: int, num_tokens: int | None = None
    ) -> Checkpoint | None:
        checkpoints_matching_step = [
            checkpoint for checkpoint in self.checkpoints if checkpoint.step == step
        ]

        if len(checkpoints_matching_step) == 0:
            return None

        if num_tokens is None:
            return checkpoints_matching_step[0]

        matching_checkpoints = [
            checkpoint
            for checkpoint in checkpoints_matching_step
            if checkpoint.num_tokens == num_tokens
        ]
        if len(matching_checkpoints) == 0:
            return None

        return matching_checkpoints[0]


MODELS: dict[str, ModelConfig] = {
    # base model
    "olmoe": ModelConfig(
        hf_name="allenai/OLMoE-1B-7B-0924",
        branch_regex=re.compile(r"step(\d+)-tokens(\d+)B"),
        revision_format="step{}-tokens{}B",
        eager_fetch=False,  # avoid network calls during import
    ),
    "phimoe": ModelConfig(
        hf_name="microsoft/Phi-3.5-MoE-instruct",
    ),
    "q3_30b": ModelConfig(
        hf_name="Qwen/Qwen3-30B-A3B",
    ),
    "gpt": ModelConfig(
        hf_name="openai/gpt-oss-20b",
    ),
    # posttrained model
    "olmoe-i": ModelConfig(
        hf_name="allenai/OLMoE-1B-7B-0125-Instruct",
        branch_regex=re.compile(r"step_(\d+)"),
        revision_format="step_{}",
        eager_fetch=False,  # avoid network calls during import
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    model_config = MODELS.get(model_name)
    if model_config is None:
        raise ValueError(f"Model {model_name} not found")

    return model_config
