# Evaluating in UniVTAC

The UniVTAC eval harness (`scripts/eval_policy.py` of the UniVTAC repo) runs in the IsaacLab environment; the
policy runs in a venv of this repo. `deploy_policy.Policy` starts `cosmos3_server.py` with that venv and talks
to it over HTTP, as the SmolVLA deploy of UniVTAC does.

1. Install this repo into a local venv (here `~/proj/cosmos3-wam/.venv`):

   ```bash
   uv venv ~/proj/cosmos3-wam/.venv --python 3.12
   VIRTUAL_ENV=~/proj/cosmos3-wam/.venv uv pip install -e <this repo> --torch-backend=cu128
   ```

2. Put the files of this directory into `policy/cosmos3/` of UniVTAC and set `cosmos3_python` in
   `deploy.yml` to that venv's python.

3. Copy a checkpoint (`checkpoints/last/pretrained_model` of a run, bfloat16 is enough) to
   `{cosmos3_ckpt_root}/{task}_{TRAIN_CONFIG}` and run, from the UniVTAC root:

   ```bash
   TRAIN_CONFIG=tac-trainable EP_NUM=100 \
       python scripts/eval_policy.py put_bottle_in_shelf clean cosmos3/deploy --headless
   ```

Execution: chunks of 50 actions, the first `cosmos3_replan_every` (10) executed, then the next chunk is sampled,
guided by Real-Time Chunking towards the rest of the previous one (`cosmos3_rtc`). Results go to
`eval_result/cosmos3/{task}/deploy/{TRAIN_CONFIG}-rtc10/`. On an RTX 4090 a chunk takes about 0.7 s (bfloat16,
30 sampling steps); policy and simulator together use about 18.5 GB.
