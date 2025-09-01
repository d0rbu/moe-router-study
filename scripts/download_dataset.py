import arguably
from huggingface_hub import snapshot_download


@arguably.command()
def download(hf_name: str = "lmsys/lmsys-chat-1m", revision: str | None = None) -> None:
    if revision is None:
        snapshot_download(
            repo_id=hf_name,
            local_dir=f"datasets/{hf_name}",
            repo_type="dataset",
        )
    else:
        snapshot_download(
            repo_id=hf_name,
            local_dir=f"datasets/{hf_name}",
            revision=revision,
            repo_type="dataset",
        )


if __name__ == "__main__":
    download()
