import os

import arguably
from huggingface_hub import snapshot_download

from exp.get_activations import MODEL_DIRNAME


@arguably.command()
def download(hf_name: str = "allenai/OLMoE-1B-7B-0125-Instruct", revision: str | None = None) -> None:
    if revision is None:
        snapshot_download(
            repo_id=hf_name,
            local_dir=os.path.join(MODEL_DIRNAME, hf_name),
            repo_type="model",
        )
    else:
        snapshot_download(
            repo_id=hf_name,
            local_dir=os.path.join(MODEL_DIRNAME, hf_name),
            revision=revision,
            repo_type="model",
        )


if __name__ == "__main__":
    download()
