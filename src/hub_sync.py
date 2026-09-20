"""Sync a run's output_dir into a subfolder of one shared Hub model repo.

Trainer's built-in push_to_hub/hub_strategy assumes the whole target repo IS
one run's output, which doesn't fit "one shared repo, one subfolder per
experiment" -- so this uploads output_dir manually instead. delete_patterns
mirrors local state remotely: checkpoints pruned locally by save_total_limit
are pruned in the Hub repo too, so it doesn't accumulate every checkpoint
ever saved.
"""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import create_repo, upload_folder
from transformers import TrainerCallback


def repo_folder_name(cfg) -> str:
    return cfg.hub_repo_folder or Path(cfg.output_dir).name


def sync_output_dir(cfg, commit_message: str) -> None:
    folder = repo_folder_name(cfg)
    create_repo(cfg.hub_repo_id, repo_type="model", private=cfg.hub_private, exist_ok=True)
    upload_folder(
        repo_id=cfg.hub_repo_id,
        folder_path=cfg.output_dir,
        path_in_repo=folder,
        commit_message=commit_message,
        # relative to path_in_repo -- only prunes stale checkpoints within
        # THIS run's folder, never touches other experiments in the repo.
        delete_patterns=["checkpoint-*"],
    )


class SyncToHubCallback(TrainerCallback):
    """Uploads output_dir to the Hub every time a checkpoint is saved."""

    def __init__(self, cfg):
        self.cfg = cfg

    def on_save(self, args, state, control, **kwargs):
        sync_output_dir(self.cfg, commit_message=f"checkpoint-{state.global_step}")
