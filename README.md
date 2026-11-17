# Cosmos3-WAM

Fine-tuning of the pretrained video world-action model Cosmos3-Edge (`lerobot/Cosmos3-Edge`) on UniVTAC
demonstrations, with tactile sensing as a third generated modality next to video and actions.

This repository covers training (cluster scripts) and the policy code. Evaluation runs in the UniVTAC
simulator, through the deploy files in `univtac_eval/`.

## Model

Cosmos3-Edge is a Mixture-of-Transformers: a frozen understanding pathway reads the task prompt, and the
trainable generation pathway denoises with flow matching, sharing self-attention with it in every layer.
Here the generation pathway denoises, jointly:

- video: Wan2.2-VAE latents of the head | wrist canvas (or the head camera alone), the current frame clean,
  `video_latent_frames` future latent frames `video_stride` dataset frames apart;
- actions: the action chunk (`chunk_size` actions, one per dataset frame);
- touch: one UniVTAC ResNet18 latent per tactile stream and frame, the current frame clean and
  `tactile_future_steps` future frames `tactile_future_stride` dataset frames apart. The encoder is frozen,
  or trained with the future-frame targets taken from its EMA copy (`tactile_encoder_trainable`).

Each modality has its own input projection, output head and loss; the generation tokens attend to each other
bidirectionally. At inference only the actions are used. Real-Time Chunking (`rtc_config`) guides each sampled
chunk towards the unexecuted rest of the previous one.

## Code

`src/lerobot` is a trimmed copy of [LeRobot](https://github.com/huggingface/lerobot) (Apache-2.0, see
`LICENSE`) at commit `fe1c545`, keeping the training stack and the `cosmos3` policy. Changes against it:

- `policies/cosmos3`: video clip decoupled from the action chunk (`video_stride`, `video_latent_frames`);
  the tactile modality (`tactile_cosmos3.py`, `tactile_*` options); Real-Time Chunking of the action sampler.
- `datasets/factory.py`: per image key frame offsets (`image_delta_indices_for_key`), so the tactile streams
  load their own frames and the other images only the current one.
- Removed: the other policies, robots, teleoperators, motors, cameras, simulation environments and in-training
  environment evaluation, and the CLI tools other than `lerobot-train`.

## Setup (NHR alex)

```bash
git clone <this repo> ~/Projects/cosmos3-wam
bash ~/Projects/cosmos3-wam/scripts/install_cluster.sbatch   # venv + Cosmos3-Edge base checkpoint
cp <UniVTAC>/encoder/checkpoints/resnet18/20251128-125750/best.pth \
    $WS/cosmos3/ckpt/univtac_resnet18_20251128-125750.pth
```

Everything heavy lives on the anvme workspace (`$WS/cosmos3`: venv, checkpoints, outputs). The datasets are
the UniVTAC LeRobot datasets (`$WS/smolvla/data/local/<task>-clean-100_vitac`).

## Train

```bash
sbatch scripts/train_cluster.sbatch put_bottle_in_shelf
sbatch --export=NONE,TACTILE=none scripts/train_cluster.sbatch put_bottle_in_shelf  # vision only
```

The cameras follow the UniVTAC benchmark: head and wrist for insert_tube and lift_can, the head camera alone
for the other tasks. Environment variables: `TACTILE=none|frozen|trainable`, `TACTILE_STRIDE`,
`TACTILE_STEPS`, `VIDEO_STRIDE`, `VIDEO_LATENT_FRAMES`, `CHUNK_SIZE`, `BATCH_SIZE`, `STEPS`, `MODE=full|lora`.
`RESUME=1` continues a run from its last checkpoint (`--dependency=afterany:<job>`). `SMOKE=1` runs short
training checks. Runs log to Trackio, mirrored to wandb by `scripts/sync_wandb.py`.

Measured on one A40 (put_bottle_in_shelf, batch 16, bf16, gradient checkpointing, EMA): about 44 GB and
2.6 s/step; 30000 steps take about 21.5 h.

## Evaluate

See `univtac_eval/README.md`.
