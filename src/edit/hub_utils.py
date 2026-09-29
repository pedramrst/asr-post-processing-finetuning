"""Hub helpers enforcing the edit method's rule for the shared repo
(PedramR/ASR_Post-processing): new results always go into a NEW folder --
nothing already in the repo is ever overwritten or deleted."""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi


def folder_files(repo_id: str, folder: str, repo_type: str = "model") -> list[str]:
    folder = folder.strip("/") + "/"
    return [f for f in HfApi().list_repo_files(repo_id, repo_type=repo_type) if f.startswith(folder)]


def ensure_new_folder(repo_id: str, folder: str, repo_type: str = "model") -> None:
    """Raises if `folder` already holds files in the repo."""
    existing = folder_files(repo_id, folder, repo_type)
    if existing:
        raise SystemExit(
            f"{repo_id}/{folder} already exists ({len(existing)} files, e.g. {existing[0]}) -- refusing to overwrite. "
            "Choose a new folder name.")


def upload_new_folder(repo_id: str, local_dir: str | Path, folder: str, commit_message: str) -> str:
    """Uploads local_dir to a folder that must not exist yet."""
    ensure_new_folder(repo_id, folder)
    HfApi().upload_folder(repo_id=repo_id, folder_path=str(local_dir), path_in_repo=folder.strip("/"),
                          commit_message=commit_message)
    return f"https://huggingface.co/{repo_id}/tree/main/{folder.strip('/')}"
