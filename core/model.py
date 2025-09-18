from dataclasses import dataclass, field
import re

import huggingface_hub  # import module so tests can patch huggingface_hub.list_repo_refs


@dataclass
class Checkpoint:
    step: int | None
    num_tokens: int | None
    model_config: "ModelConfig"
    revision: str | None = None

    def __str__(self):
        if self.revision is not None:
            return self.revision

        # Fail explicitly if revision_format is None
        if self.model_config.revision_format is None:
            raise ValueError(
                "ModelConfig.revision_format is required for string representation"
            )
        return self.model_config.revision_format.format(self.step, self.num_tokens)


LATEST_REVISION = "main"


@dataclass
class ModelConfig:
    hf_name: str
    branch_regex: re.Pattern | str | None = None
    revision_format: str | None = None
    total_steps: int | None = None
    total_tokens: int | None = None
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

        self.checkpoints = sorted(checkpoints, key=lambda x: (x.step, x.num_tokens))

        if not self.checkpoints:
            self.checkpoints = [
                Checkpoint(
                    self.total_steps, self.total_tokens, self, revision=LATEST_REVISION
                )
            ]
            return

        max_steps = max(checkpoint.step for checkpoint in self.checkpoints)
        max_num_tokens = max(checkpoint.num_tokens for checkpoint in self.checkpoints)
        if self.total_steps is None:
            assert self.total_steps >= max_steps, (
                f"total_steps {self.total_steps} is less than the highest checkpoint {max_steps}"
            )

        if self.total_tokens is None:
            assert self.total_tokens >= max_num_tokens, (
                f"total_tokens {self.total_tokens} is less than the highest checkpoint {max_num_tokens}"
            )

        # don't add the main revision if it's already in the checkpoints
        if self.total_steps == max_steps and self.total_tokens == max_num_tokens:
            return

        self.checkpoints.append(
            Checkpoint(
                self.total_steps, self.total_tokens, self, revision=LATEST_REVISION
            )
        )

    def get_checkpoint(
        self, step: int | None = None, num_tokens: int | None = None
    ) -> Checkpoint | None:
        if step is None and num_tokens is None:
            return self.checkpoints[-1] if self.checkpoints else None

        if step is not None:
            checkpoints_matching_step = {
                checkpoint for checkpoint in self.checkpoints if checkpoint.step == step
            }
        else:
            checkpoints_matching_step = set(self.checkpoints)

        if num_tokens is not None:
            checkpoints_matching_num_tokens = {
                checkpoint
                for checkpoint in self.checkpoints
                if checkpoint.num_tokens == num_tokens
            }
        else:
            checkpoints_matching_num_tokens = set(self.checkpoints)

        matching_checkpoints = (
            checkpoints_matching_step & checkpoints_matching_num_tokens
        )
        if len(matching_checkpoints) == 0:
            return None

        sorted_matching_checkpoints = sorted(
            matching_checkpoints, key=lambda checkpoint: str(checkpoint)
        )
        return sorted_matching_checkpoints[-1]

    def get_checkpoint_strict(self, step: int, num_tokens: int) -> Checkpoint:
        checkpoint = self.get_checkpoint(step=step, num_tokens=num_tokens)
        if checkpoint is None:
            raise ValueError(
                f"Checkpoint for step {step} and num_tokens {num_tokens} not found"
            )

        return checkpoint


MODELS: dict[str, ModelConfig] = {
    # base model
    "olmoe": ModelConfig(
        hf_name="allenai/OLMoE-1B-7B-0924",
        branch_regex=re.compile(r"step(\d+)-tokens(\d+)B"),
        revision_format="step{}-tokens{}B",
        eager_fetch=False,  # avoid network calls during import
        total_steps=1_223_842,
        total_tokens=5133,  # 5133B tokens
    ),
    "phimoe": ModelConfig(
        hf_name="microsoft/Phi-3.5-MoE-instruct",
        total_tokens=4900,  # 4.9T tokens
    ),
    "q3_30b": ModelConfig(
        hf_name="Qwen/Qwen3-30B-A3B",
        total_tokens=36000,  # 36T tokens
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
