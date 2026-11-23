"""Cosmos3 Edge with touch (world action model of the cosmos3-wam repo) for the UniVTAC eval harness.

As for SmolVLA, the IsaacLab eval process and the policy run in separate Python environments: `Policy` (IsaacLab
env) starts `cosmos3_server.py` with the Cosmos3 venv (`cosmos3_python`, default ~/proj/cosmos3-wam/.venv) and talks
to it over HTTP; `_Cosmos3LocalPolicy` runs in the server.

Checkpoint: `cosmos3_ckpt_root/{task}_{TRAIN_CONFIG}` (a lerobot `pretrained_model` dir), e.g.
`TRAIN_CONFIG=tac-trainable` -> ~/proj/cosmos3-wam/ckpt/put_bottle_in_shelf_tac-trainable. A `ckpt_dir` arg overrides it.

Execution (`cosmos3_execution`, env COSMOS3_EXECUTION):
- chunk: the first `cosmos3_replan_every` actions of each chunk, then the next chunk (run tag `{TRAIN_CONFIG}-replan{n}`).
- rtc: as chunk, the sampling of each chunk guided towards the unexecuted rest of the previous one (Real-Time Chunking,
  synchronous, inference delay 0; run tag `-rtc{n}`).
- ensemble: a chunk every step, the action the weighted mean of the predictions of the last `cosmos3_ensemble_first_n`
  chunks for this step, weights exp(-k * i) from the oldest (FTP-1's temporal ensemble; run tag `-ens{n}`).
Tactile ablation (`cosmos3_tactile_mask`, env COSMOS3_TACTILE_MASK) of a model trained with touch:
- none: the tactile images as observed.
- static: the tactile images held at the first observation of the episode (run tag suffix `-tacstatic`): the touch
  input never changes. In lift_bottle and lift_can that is the open gripper (no contact), in the other tasks the
  grasp of the object done before the policy starts.
- static_all: as static, and the future tactile frames the model generates jointly with the actions are held at the
  same frame too (inpainted at every sampling step, `-tacstaticall`): the whole touch stream says nothing changes.
- drop: no tactile tokens at all (`-tacdrop`), the generation stream `[video | actions]` of a model trained without
  touch: the model trained with touch run without its touch input.
A checkpoint trained without touch (TRAIN_CONFIG tac-none...) has no tactile input to mask.
The action representation (absolute or relative to the joint state, `action_preset`) comes from the checkpoint: its
postprocessor returns absolute joint targets either way. The policy samples video latents, actions and tactile
latents jointly; only the actions are used.

Debug records (`cosmos3_debug`, env COSMOS3_DEBUG=1/0): `<eval dir>/debug/<seed>_<result>/` next to `video/`, one
folder per episode with the policy inputs, the executed actions and the simulator ground truth of the task (the
recorders of UniVTAC's `policy/_debug/`, picked by task name), plus the chunks and the predicted / observed tactile latents of the
model. The `Policy` wrapper records after the policy acted, in the IsaacLab process: the ground truth is only
written to disk, the policy gets the same observation as without debugging.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_COSMOS3_DIR = Path(__file__).resolve().parent
sys.path.append(str(_COSMOS3_DIR.parent))

from .._base_policy import BasePolicy  # noqa: E402

_DEFAULT_ROOT = Path.home() / "proj" / "cosmos3-wam"


def _deploy_cfg(args: dict) -> dict:
    with open(_COSMOS3_DIR / "deploy.yml", "r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.update({k: v for k, v in args.items() if k.startswith("cosmos3_")})
    return cfg


def _train_config_name(args: dict) -> str:
    return args.get("train_config_name") or os.environ.get("TRAIN_CONFIG") or "tac-trainable"


EXECUTION_MODES = ("chunk", "rtc", "ensemble")
TACTILE_MASKS = {"none": "", "static": "-tacstatic", "static_all": "-tacstaticall", "drop": "-tacdrop"}


def _replan_every(args: dict) -> int:
    return int(os.environ.get("COSMOS3_REPLAN_EVERY") or _deploy_cfg(args).get("cosmos3_replan_every") or 10)


def _execution(args: dict) -> str:
    """chunk | rtc | ensemble; COSMOS3_EXECUTION overrides cosmos3_execution (the old COSMOS3_RTC=1/0 still works)."""
    cfg = _deploy_cfg(args)
    mode = os.environ.get("COSMOS3_EXECUTION") or cfg.get("cosmos3_execution")
    if not mode:
        env = os.environ.get("COSMOS3_RTC")
        rtc = env.strip().lower() in ("1", "true", "yes") if env is not None else bool(cfg.get("cosmos3_rtc", False))
        mode = "rtc" if rtc else "chunk"
    if mode not in EXECUTION_MODES:
        raise ValueError(f"cosmos3_execution={mode!r} is not one of {EXECUTION_MODES}")
    return mode


def _ensemble_first_n(args: dict) -> int:
    return int(os.environ.get("COSMOS3_ENSEMBLE_FIRST_N") or _deploy_cfg(args).get("cosmos3_ensemble_first_n") or 20)


def _tactile_mask(args: dict) -> str:
    """none | static | static_all | drop; COSMOS3_TACTILE_MASK overrides cosmos3_tactile_mask."""
    mask = os.environ.get("COSMOS3_TACTILE_MASK") or _deploy_cfg(args).get("cosmos3_tactile_mask") or "none"
    if mask not in TACTILE_MASKS:
        raise ValueError(f"cosmos3_tactile_mask={mask!r} is not one of {list(TACTILE_MASKS)}")
    return mask


def _debug_enabled(args: dict) -> bool:
    """COSMOS3_DEBUG=1/0 overrides cosmos3_debug."""
    env = os.environ.get("COSMOS3_DEBUG")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes")
    return bool(_deploy_cfg(args).get("cosmos3_debug", False))


def get_run_tag(args: dict) -> str:
    """Name of the eval result dir: the checkpoint, plus the execution mode and the tactile ablation."""
    mode = _execution(args)
    if mode == "ensemble":
        tag = f"{_train_config_name(args)}-ens{_ensemble_first_n(args)}"
    else:
        tag = f"{_train_config_name(args)}-{'rtc' if mode == 'rtc' else 'replan'}{_replan_every(args)}"
    return tag + TACTILE_MASKS[_tactile_mask(args)]


def _instruction_for(task_name: str) -> str:
    """The training task string of the UniVTAC LeRobot datasets (put_bottle_in_shelf -> put the bottle in shelf)."""
    words = task_name.split("_")
    return f"{words[0]} the {' '.join(words[1:])}" if len(words) >= 2 else task_name


def _resolve_ckpt_dir(args: dict) -> Path:
    if args.get("ckpt_dir"):
        return Path(args["ckpt_dir"]).expanduser()
    root = Path(_deploy_cfg(args).get("cosmos3_ckpt_root") or _DEFAULT_ROOT / "ckpt").expanduser()
    return root / f"{args['task_name']}_{_train_config_name(args)}"


# ---------------------------------------------------------------------------
# Model runner (Cosmos3 venv, inside cosmos3_server.py)
# ---------------------------------------------------------------------------


class _Cosmos3LocalPolicy:
    # lerobot key -> (observation root, camera, image key)
    _IMAGE_MAP = {
        "observation.images.head": ("observation", "head", "rgb"),
        "observation.images.wrist": ("observation", "wrist", "rgb"),
        "observation.images.tactile_ll": ("tactile", "left_tactile", "rgb"),
        "observation.images.tactile_lr": ("tactile", "right_tactile", "rgb"),
        "observation.images.tactile_ll_marker": ("tactile", "left_tactile", "rgb_marker"),
        "observation.images.tactile_lr_marker": ("tactile", "right_tactile", "rgb_marker"),
    }

    def __init__(self, args: dict):
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.cosmos3.modeling_cosmos3 import Cosmos3EdgePolicy
        from lerobot.policies.cosmos3.processor_cosmos3 import make_cosmos3_pre_post_processors_from_pretrained

        cfg = _deploy_cfg(args)
        self.task_name = args["task_name"]
        self.replan_every = _replan_every(args)
        self.device = torch.device(args.get("device", "cuda"))
        ckpt_dir = _resolve_ckpt_dir(args)
        if not (ckpt_dir / "model.safetensors").exists():
            raise FileNotFoundError(f"Cosmos3 checkpoint not found: {ckpt_dir} (set TRAIN_CONFIG or ckpt_dir)")
        print(f"[cosmos3] loading {ckpt_dir}", flush=True)

        config = PreTrainedConfig.from_pretrained(str(ckpt_dir))
        config.device = str(self.device)
        config.dtype = getattr(torch, str(cfg.get("cosmos3_dtype", "bfloat16")))
        config.tokenizer_name = str(ckpt_dir / "tokenizer")
        config.tactile_encoder_path = None  # the encoder weights are in the checkpoint
        if cfg.get("cosmos3_num_inference_steps"):
            config.num_inference_steps = int(cfg["cosmos3_num_inference_steps"])
        if cfg.get("cosmos3_sampler"):
            config.sampler = str(cfg["cosmos3_sampler"])
        self.execution = _execution(args)
        self.use_rtc = self.execution == "rtc"
        self.ensemble_first_n = _ensemble_first_n(args)
        self.ensemble_k = float(cfg.get("cosmos3_ensemble_k", 0.01))
        self.tactile_mask = _tactile_mask(args)
        if self.tactile_mask != "none" and not config.tactile_keys:
            raise ValueError(
                f"cosmos3_tactile_mask={self.tactile_mask!r} needs a checkpoint trained with touch, {ckpt_dir} has no "
                "tactile input (use cosmos3_tactile_mask none)."
            )
        self._held_tactile: dict[str, Any] = {}  # the tactile images of the first observation, for the ablation
        self.debug = _debug_enabled(args)
        self._debug: dict[str, Any] = {}  # this step's model outputs for the debug records (`pop_debug`)
        if self.use_rtc:
            from lerobot.configs.types import RTCAttentionSchedule
            from lerobot.policies.rtc.configuration_rtc import RTCConfig

            config.rtc_config = RTCConfig(
                enabled=True,
                prefix_attention_schedule=RTCAttentionSchedule[str(cfg.get("cosmos3_rtc_schedule", "LINEAR")).upper()],
                max_guidance_weight=float(cfg.get("cosmos3_rtc_max_guidance_weight", 10.0)),
                execution_horizon=config.chunk_size - self.replan_every,
            )
        self._prev_chunk = None  # last chunk in model (normalized) space, for RTC
        self._prev_anchor = None  # the joint state that chunk is relative to (relative actions)
        self._history: list[tuple[Any, int]] = []  # (absolute chunk, step it was predicted at), for the ensemble
        self._step = 0
        self.config = config
        t0 = time.time()
        self.model = Cosmos3EdgePolicy.from_pretrained(str(ckpt_dir), config=config).to(self.device).eval()
        self.pre, self.post = make_cosmos3_pre_post_processors_from_pretrained(
            config,
            ckpt_dir,
            preprocessor_overrides={
                "device_processor": {"device": str(self.device)},
                "cosmos3_tokenizer_processor": {"tokenizer_name": str(ckpt_dir / "tokenizer")},
            },
        )
        from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
        from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep

        self._normalizer = next(s for s in self.pre.steps if isinstance(s, NormalizerProcessorStep))
        self._unnormalizer = next(s for s in self.post.steps if isinstance(s, UnnormalizerProcessorStep))
        self._relative = next(
            (s for s in self.pre.steps if isinstance(s, RelativeActionsProcessorStep) and s.enabled), None
        )
        keys = list(config.image_features) + list(config.tactile_keys)
        self._image_map = {k: self._IMAGE_MAP[k] for k in keys}
        self.image_size = tuple(next(iter(config.image_features.values())).shape[-2:])
        print(
            f"[cosmos3] loaded in {time.time() - t0:.0f}s: {config.dtype}, chunk {config.chunk_size}, "
            f"{config.num_inference_steps} sampling steps, video stride {config.video_stride} x "
            f"{config.video_latent_frames} latents, tactile {config.tactile_keys} "
            f"(trainable encoder: {config.tactile_encoder_trainable}); action preset "
            f"{getattr(config, 'action_preset', None) or 'absolute'}, relative {self._relative is not None}, "
            f"state input {config.use_state}, sampler {config.sampler}; tactile mask {self.tactile_mask}; "
            f"execution {self.execution} "
            f"({self.ensemble_first_n if self.execution == 'ensemble' else self.replan_every}); "
            f"streams {list(self._image_map)}",
            flush=True,
        )
        self._queue: deque = deque()
        self.instruction = _instruction_for(self.task_name)
        self.chunk_times: list[float] = []

    def encode_obs(self, observation: dict) -> dict:
        import torch
        import torch.nn.functional as F  # noqa: N812

        batch: dict[str, Any] = {}
        for key, (root, cam, img_key) in self._image_map.items():
            img = torch.as_tensor(np.asarray(observation[root][cam][img_key]), device=self.device)
            img = img.permute(2, 0, 1).float() / 255.0  # HWC uint8 -> CHW [0, 1], as the LeRobot dataset images
            if tuple(img.shape[-2:]) != self.image_size:
                img = F.interpolate(img[None], size=self.image_size, mode="bilinear", antialias=True)[0]
            if self.tactile_mask in ("static", "static_all") and key in self.config.tactile_keys:
                img = self._held_tactile.setdefault(key, img)  # held at the first observation of the episode
            batch[key] = img
        ee = torch.as_tensor(np.asarray(observation["embodiment"]["ee"]), device=self.device).float().view(-1)
        joint = torch.as_tensor(np.asarray(observation["embodiment"]["joint"]), device=self.device).float().view(-1)
        batch["observation.state"] = torch.cat([ee, joint])
        batch["task"] = self.instruction
        return batch

    def _reanchor(self, left_over, anchor):
        """
        The unexecuted rest of the previous chunk (normalized, relative to `self._prev_anchor`) relative to the new
        joint state `anchor` and normalized again, the RTC target of relative actions.
        """
        from lerobot.processor.relative_action_processor import to_absolute_actions, to_relative_actions

        mask = self._relative._build_mask(left_over.shape[-1])
        absolute = to_absolute_actions(self._unnormalizer._normalize_action(left_over, inverse=True), self._prev_anchor, mask)
        return self._normalizer._normalize_action(to_relative_actions(absolute, anchor, mask), inverse=False)

    def _predict(self, observation: dict, rtc: bool):
        """One chunk `[chunk_size, action_dim]` of absolute joint targets for the current observation."""
        import torch

        batch = self.pre(self.encode_obs(observation))
        anchor = self._relative.get_cached_state() if self._relative is not None else None
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        kwargs = {"tactile_static_future": self.tactile_mask == "static_all", "tactile_drop": self.tactile_mask == "drop"}
        if rtc and self._prev_chunk is not None:
            left_over = self._prev_chunk[:, self.replan_every :]
            if self._relative is not None:
                left_over = self._reanchor(left_over, anchor)
            kwargs.update(inference_delay=0, prev_chunk_left_over=left_over, execution_horizon=left_over.shape[1])
        with torch.no_grad():  # the RTC processor enables grad internally for its (network-free) correction
            raw_chunk = self.model.predict_action_chunk(batch, **kwargs)
        self._prev_chunk, self._prev_anchor = raw_chunk.detach(), anchor
        chunk = self.post(raw_chunk)  # unnormalized, absolute (relative actions plus the anchor)
        if self.debug:
            self._debug["chunk"] = torch.as_tensor(chunk, dtype=torch.float32).reshape(-1, chunk.shape[-1]).cpu().numpy()
            predicted = getattr(self.model.model, "last_predicted_tactile", None)
            if predicted is not None:  # [1, frames, streams, latent], the current frame first
                self._debug["pred_tactile"] = predicted[0].to(torch.float16).cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.chunk_times.append(time.time() - t0)
        if len(self.chunk_times) in (1, 2, 10) or len(self.chunk_times) % 100 == 0:
            print(f"[cosmos3] chunk {len(self.chunk_times)}: {self.chunk_times[-1]:.2f}s", flush=True)
        return torch.as_tensor(chunk, dtype=torch.float32).reshape(-1, chunk.shape[-1]).cpu().numpy()

    def _observed_tactile_latent(self, observation: dict):
        """The tactile latent `[streams, latent]` of the observed images (target encoder, as the training targets)."""
        import torch
        import torch.nn.functional as F  # noqa: N812

        images = []
        for key in self.config.tactile_keys:
            root, cam, img_key = self._image_map[key]
            img = torch.as_tensor(np.asarray(observation[root][cam][img_key]), device=self.device).permute(2, 0, 1).float() / 255.0
            if tuple(img.shape[-2:]) != self.image_size:
                img = F.interpolate(img[None], size=self.image_size, mode="bilinear", antialias=True)[0]
            images.append(img)
        with torch.no_grad():
            latent = self.model.model.network.encode_tactile(torch.stack(images)[None, None], target=True)
        return latent[0, 0].to(torch.float16).cpu().numpy()

    def pop_debug(self) -> dict:
        """The model outputs of the last `get_action` for the debug records (empty without `cosmos3_debug`)."""
        out, self._debug = self._debug, {}
        return out

    def get_action(self, observation: dict, instruction: str | None = None) -> np.ndarray:
        self._debug = {}
        if self.debug and self.config.tactile_keys:
            self._debug["obs_tactile"] = self._observed_tactile_latent(observation)  # as observed, unmasked
        if self.execution == "ensemble":
            self._history.append((self._predict(observation, rtc=False), self._step))
            # The chunk predicted at step s holds the action of step e at index e - s.
            self._history = [(c, s) for c, s in self._history if self._step - s < min(self.ensemble_first_n, len(c))]
            candidates = np.stack([c[self._step - s] for c, s in self._history])  # oldest first
            weights = np.exp(-self.ensemble_k * np.arange(len(candidates), dtype=np.float32))
            self._step += 1
            return (candidates * (weights / weights.sum())[:, None]).sum(axis=0)
        if not self._queue:
            self._queue.extend(self._predict(observation, rtc=self.use_rtc)[: self.replan_every])
        return self._queue.popleft()

    def reset(self):
        self._queue.clear()
        self._prev_chunk = self._prev_anchor = None
        self._history, self._step = [], 0
        self._held_tactile = {}
        self._debug = {}
        if self._relative is not None:
            self._relative.reset()
        self.model.reset()


# ---------------------------------------------------------------------------
# HTTP transport (both environments)
# ---------------------------------------------------------------------------


def _to_wire_payload(value: Any):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {"__ndarray__": True, "dtype": str(array.dtype), "shape": list(array.shape),
                "data": base64.b64encode(array.tobytes()).decode("ascii")}
    if isinstance(value, dict):
        return {str(k): _to_wire_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_wire_payload(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    return _to_wire_payload(np.asarray(value))


def _from_wire_payload(value: Any):
    if isinstance(value, dict):
        if value.get("__ndarray__") is True:
            data = base64.b64decode(value["data"].encode("ascii"))
            return np.frombuffer(data, dtype=np.dtype(value["dtype"])).reshape(value["shape"]).copy()
        return {k: _from_wire_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_wire_payload(v) for v in value]
    return value


class _Cosmos3Client:
    """Starts cosmos3_server.py with the Cosmos3 venv on a free local port (or uses `cosmos3_port`)."""

    def __init__(self, args: dict):
        cfg = _deploy_cfg(args)
        self.host = str(cfg.get("cosmos3_host", "127.0.0.1"))
        self.port = int(cfg.get("cosmos3_port") or 0)
        self.python = Path(cfg.get("cosmos3_python") or _DEFAULT_ROOT / ".venv" / "bin" / "python").expanduser()
        self.timeout = float(cfg.get("cosmos3_request_timeout", 600))
        self.authkey = str(cfg.get("cosmos3_authkey", ""))
        self._process: subprocess.Popen | None = None
        self._external = self.port != 0
        if not self._external:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                self.port = int(sock.getsockname()[1])
            self.authkey = secrets.token_hex(16)
            if not self.python.exists():
                raise FileNotFoundError(f"Cosmos3 python not found: {self.python}")
            self._process = subprocess.Popen(
                [str(self.python), str(_COSMOS3_DIR / "cosmos3_server.py"), "--host", self.host,
                 "--port", str(self.port), "--authkey", self.authkey],
                cwd=str(_COSMOS3_DIR.parents[1]),
                # A PYTHONPATH to another lerobot (SmolVLA's src) would shadow the Cosmos3 one.
                env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            )
        self.base_url = f"http://{self.host}:{self.port}"
        deadline = time.monotonic() + float(cfg.get("cosmos3_startup_timeout", 300))
        while True:
            try:
                if self._request("GET", "/health").get("status") == "ok":
                    break
            except Exception as exc:  # noqa: BLE001
                if self._process is not None and self._process.poll() is not None:
                    raise RuntimeError(f"Cosmos3 server exited with code {self._process.returncode}") from exc
                if time.monotonic() > deadline:
                    raise TimeoutError("timed out waiting for the Cosmos3 server") from exc
            time.sleep(0.5)
        self._post("/init", {"args": _to_wire_payload(args)})

    def _request(self, method: str, path: str, payload: Any = None):
        data = None if payload is None else json.dumps({**payload, "authkey": self.authkey}).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path, data=data, method=method,
            headers={"X-Cosmos3-Auth": self.authkey, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Cosmos3 server HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc

    def _post(self, path: str, payload: Any):
        response = self._request("POST", path, payload)
        if isinstance(response, dict) and response.get("type") == "error":
            raise RuntimeError(f"Cosmos3 server error:\n{response.get('error')}\n{response.get('traceback')}")
        return response

    def get_action(self, observation: dict, instruction: str | None, debug: bool = False) -> np.ndarray:
        """The next action; with `debug` the server's model outputs of this step go to `last_debug`."""
        payload = {"instruction": instruction, "observation": _to_wire_payload(observation)}
        if debug:
            payload["debug"] = True
        response = self._post("/act", payload)
        self.last_debug = _from_wire_payload(response["debug"]) if debug and "debug" in response else None
        return np.asarray(_from_wire_payload(response["action"]), dtype=np.float32)

    def reset(self) -> None:
        self._post("/reset", {})

    def close(self) -> None:
        try:
            self._post("/reset" if self._external else "/shutdown", {})
        except Exception:  # noqa: BLE001
            pass
        if self._process is not None:
            try:
                self._process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._process.kill()


class Policy(BasePolicy):
    """Deploy-time Cosmos3 wrapper for UniVTAC (IsaacLab side)."""

    def __init__(self, args: dict):
        self.task_name = args["task_name"]
        self.debug = _debug_enabled(args)
        self.model = _Cosmos3Client(args)
        self._recorder = None
        self._task = None

    def eval(self, task, observation):
        import torch

        action = self.model.get_action(observation, None, debug=self.debug).reshape(-1)
        if self.debug:
            self._record(task, observation, action)
        # Trained on embodiment/joint: 7 arm joints + 2 mirrored finger joints; take_action("qpos") takes
        # 7 arm joints + 1 gripper value.
        action_t = torch.from_numpy(action[:8].copy()).to(task.device).float()
        return task.take_action(action_t, action_type="qpos")

    def _record(self, task, observation, action) -> None:
        """Debug record of this step (after the policy acted; never raises into the eval)."""
        try:
            if self._recorder is None or self._recorder.seed != int(task.cfg.seed):
                self._finish_episode()
                from .._debug import make_recorder

                self._recorder, self._task = make_recorder(self.task_name, task, task.save_root / "debug"), task
            self._recorder.record(task, observation, action, self.model.last_debug)
        except Exception as exc:  # noqa: BLE001
            print(f"[cosmos3] debug record failed: {exc!r}", flush=True)

    def _finish_episode(self) -> None:
        """Write the recorded episode under the result the eval stored in metadata.json (video/<seed>_<result>)."""
        recorder, self._recorder = self._recorder, None
        if recorder is None:
            return
        try:
            metadata = {}
            if self._task.metadata_path.exists():
                metadata = json.loads(self._task.metadata_path.read_text()).get(str(recorder.seed), {})
            recorder.finish(metadata)
        except Exception as exc:  # noqa: BLE001
            print(f"[cosmos3] debug finish failed: {exc!r}", flush=True)

    def reset(self):
        self._finish_episode()  # the previous episode is over, its result is in metadata.json
        self.model.reset()

    def close(self):
        self._finish_episode()
        self.model.close()
