#!/usr/bin/env python

"""
Action presets of the Cosmos3 policy (UniVTAC): named bundles of the action representation and its normalization,
picked with `--policy.action_preset=<name>` (sbatch: ACTION_PRESET). `Cosmos3Config.__post_init__` sets the listed
config fields to the preset's values (they win over the same fields given separately); adding a method is adding
an entry here.

The UniVTAC Franka layout: actions are `[7 arm joints, 2 mirrored finger joints]` (`action_0` .. `action_8`), the
joint state the same 9 values (`state_indices` of the 16 value `[ee(7), joint(9)]` state).

- absolute: absolute joint targets, quantile normalization (the default config, nothing changed).
- ftp1: the action representation of FTP-1 (openpi pi0.5 style `mix`): the arm joints relative to the current joint
  state (`action - state`, every step of the chunk against the same state), the fingers absolute, z-score
  normalization clipped to +-10.
"""

from typing import Any

from lerobot.configs.types import NormalizationMode

UNIVTAC_FINGER_ACTIONS = ["action_7", "action_8"]

ACTION_PRESETS: dict[str, dict[str, Any]] = {
    "absolute": {},
    "ftp1": {
        "use_relative_actions": True,
        "relative_exclude_joints": UNIVTAC_FINGER_ACTIONS,
        "action_normalization": NormalizationMode.MEAN_STD,
        "zscore_clip": 10.0,
    },
}
