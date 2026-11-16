# patch-backup: Cosmos3-WAM as patches on upstream LeRobot

This branch keeps the earlier, patch-based version of the Cosmos3 training code, the one the first
30k-step runs were trained with (put_bottle_in_shelf, tactile encoder frozen / trainable,
2026-11-15). `main` has the same changes edited into a trimmed copy of LeRobot instead.

Layout (as in the UniVTAC repo, where this code lived under `policy/cosmos3/`):

- `policy/cosmos3/patches/`: `git am` on huggingface/lerobot `fe1c545`.
  - 0001: video clip decoupled from the action chunk (`video_stride`, `video_latent_frames`)
  - 0002: tactile modality in the generation pathway (UniVTAC encoder, per-key image offsets)
  - 0003: Real-Time Chunking guidance of the action sampler (inference)
- `policy/cosmos3/scripts/`: install / train / check sbatch for NHR alex, `sync_wandb.py`.

Use (the scripts expect `REPO` to be the directory containing `policy/cosmos3`):

```bash
git clone -b patch-backup git@github.com:Char1esFu/cosmos3-wam.git ~/Projects/cosmos3-wam-patch
REPO=~/Projects/cosmos3-wam-patch sbatch --export=NONE,REPO=$HOME/Projects/cosmos3-wam-patch \
    ~/Projects/cosmos3-wam-patch/policy/cosmos3/scripts/install_cluster.sbatch
sbatch --export=NONE,REPO=$HOME/Projects/cosmos3-wam-patch \
    ~/Projects/cosmos3-wam-patch/policy/cosmos3/scripts/train_cluster.sbatch put_bottle_in_shelf
```

The install script clones LeRobot into `$WS/cosmos3/lerobot`, applies the patches on branch
`univtac` and builds `$WS/cosmos3/.venv`.
