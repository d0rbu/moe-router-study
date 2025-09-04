import os

import arguably
from huggingface_hub import snapshot_download

from exp import DATASET_DIRNAME


@arguably.command()
def download(hf_name: str = "lmsys/lmsys-chat-1m", revision: str | None = None) -> None:
    local_dir = os.path.join(DATASET_DIRNAME, hf_name)
    os.makedirs(local_dir, exist_ok=True)

    if revision is None:
        snapshot_download(
            repo_id=hf_name,
            local_dir=local_dir,
            repo_type="dataset",
        )
    else:
        snapshot_download(
            repo_id=hf_name,
            local_dir=local_dir,
            revision=revision,
            repo_type="dataset",
        )


if __name__ == "__main__":
    download()
