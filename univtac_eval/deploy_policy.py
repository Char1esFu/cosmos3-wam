"""Cosmos3 Edge with touch (world action model of the cosmos3-wam repo) for the UniVTAC eval harness.

As for SmolVLA, the IsaacLab eval process and the policy run in separate Python environments: `Policy` (IsaacLab
env) starts `cosmos3_server.py` with the Cosmos3 venv (`cosmos3_python`, default ~/proj/cosmos3-wam/.venv) and talks
to it over HTTP; `_Cosmos3LocalPolicy` runs in the server.

Checkpoint: `cosmos3_ckpt_root/{task}_{TRAIN_CONFIG}` (a lerobot `pretrained_model` dir), e.g.
`TRAIN_CONFIG=tac-trainable` -> ~/proj/cosmos3-wam/ckpt/put_bottle_in_shelf_tac-trainable. A `ckpt_dir` arg overrides it.

Execution: chunks of `chunk_size` (50) actions, the first `cosmos3_replan_every` executed before the next chunk is
sampled (receding horizon, run tag `{TRAIN_CONFIG}-replan{n}`). With `cosmos3_rtc` (run tag `-rtc{n}`) the sampling of
each chunk is guided towards the unexecuted rest of the previous one (Real-Time Chunking, synchronous, inference
delay 0, as for SmolVLA). The policy samples video latents, actions and tactile
latents jointly; only the actions are used.
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


def _replan_every(args: dict) -> int:
    return int(os.environ.get("COSMOS3_REPLAN_EVERY") or _deploy_cfg(args).get("cosmos3_replan_every") or 10)


def _use_rtc(args: dict) -> bool:
    """Real-Time Chunking guidance between consecutive chunks; COSMOS3_RTC=1/0 overrides cosmos3_rtc."""
    env = os.environ.get("COSMOS3_RTC")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes")
    return bool(_deploy_cfg(args).get("cosmos3_rtc", False))


def get_run_tag(args: dict) -> str:
    """Name of the eval result dir: the checkpoint, plus the execution mode."""
    mode = "rtc" if _use_rtc(args) else "replan"
    return f"{_train_config_name(args)}-{mode}{_replan_every(args)}"


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
        self.use_rtc = _use_rtc(args)
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
        keys = list(config.image_features) + list(config.tactile_keys)
        self._image_map = {k: self._IMAGE_MAP[k] for k in keys}
        self.image_size = tuple(next(iter(config.image_features.values())).shape[-2:])
        print(
            f"[cosmos3] loaded in {time.time() - t0:.0f}s: {config.dtype}, chunk {config.chunk_size}, "
            f"{config.num_inference_steps} sampling steps, video stride {config.video_stride} x "
            f"{config.video_latent_frames} latents, tactile {config.tactile_keys} "
            f"(trainable encoder: {config.tactile_encoder_trainable}); execute {self.replan_every} actions per chunk"
            f"{', RTC guidance' if self.use_rtc else ''}; "
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
            batch[key] = img
        ee = torch.as_tensor(np.asarray(observation["embodiment"]["ee"]), device=self.device).float().view(-1)
        joint = torch.as_tensor(np.asarray(observation["embodiment"]["joint"]), device=self.device).float().view(-1)
        batch["observation.state"] = torch.cat([ee, joint])
        batch["task"] = self.instruction
        return batch

    def get_action(self, observation: dict, instruction: str | None = None) -> np.ndarray:
        import torch

        if self._queue:
            return self._queue.popleft()
        batch = self.pre(self.encode_obs(observation))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        kwargs = {}
        if self.use_rtc and self._prev_chunk is not None:
            left_over = self._prev_chunk[:, self.replan_every :]
            kwargs = {"inference_delay": 0, "prev_chunk_left_over": left_over, "execution_horizon": left_over.shape[1]}
        with torch.no_grad():  # the RTC processor enables grad internally for its (network-free) correction
            raw_chunk = self.model.predict_action_chunk(batch, **kwargs)
        self._prev_chunk = raw_chunk.detach()
        chunk = self.post(raw_chunk)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.chunk_times.append(time.time() - t0)
        if len(self.chunk_times) in (1, 2, 10) or len(self.chunk_times) % 100 == 0:
            print(f"[cosmos3] chunk {len(self.chunk_times)}: {self.chunk_times[-1]:.2f}s", flush=True)
        chunk = torch.as_tensor(chunk, dtype=torch.float32).reshape(-1, chunk.shape[-1])
        self._queue.extend(a.cpu().numpy() for a in chunk[: self.replan_every])
        return self._queue.popleft()

    def reset(self):
        self._queue.clear()
        self._prev_chunk = None
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

    def get_action(self, observation: dict, instruction: str | None) -> np.ndarray:
        response = self._post("/act", {"instruction": instruction, "observation": _to_wire_payload(observation)})
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
        self.model = _Cosmos3Client(args)

    def eval(self, task, observation):
        import torch

        action = self.model.get_action(observation, None).reshape(-1)
        # Trained on embodiment/joint: 7 arm joints + 2 mirrored finger joints; take_action("qpos") takes
        # 7 arm joints + 1 gripper value.
        action_t = torch.from_numpy(action[:8].copy()).to(task.device).float()
        return task.take_action(action_t, action_type="qpos")

    def reset(self):
        self.model.reset()

    def close(self):
        self.model.close()
