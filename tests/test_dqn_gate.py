"""
Tests for the DQN patch-inspection gate (stop action + reward shaping).

These build a PatchInspectionEnv without its heavy __init__ (which would run
EfficientNet over a dataset) and exercise the step logic directly.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    DQN_STEP_PENALTY,
    DQN_DEFECT_REWARD,
    DQN_STOP_BONUS,
    DQN_MISS_PENALTY,
)
from src.models.dqn_env import PatchInspectionEnv


def make_env(patch_labels):
    """A 2x2-grid env with one cached image whose patch defect labels are given."""
    env = object.__new__(PatchInspectionEnv)
    labels = np.array(patch_labels, dtype=np.float32)
    env.n_patches = len(labels)
    env.stop_action = env.n_patches
    env.grid_size = 2
    env._cached_patch_labels = [labels]
    env._current_idx = 0
    env._image_features = np.zeros(3, dtype=np.float32)
    env._inspected_mask = np.zeros(env.n_patches, dtype=np.float32)
    env._defects_found = 0
    env._defects_total = int(labels.sum())
    return env


def test_stop_action_terminates():
    env = make_env([1, 0, 0, 0])
    _, reward, terminated, truncated, info = env.step(env.stop_action)
    assert terminated and not truncated
    assert info["stopped"] is True


def test_stop_with_full_recall_pays_bonus():
    env = make_env([1, 0, 0, 0])
    env.step(0)  # inspect the one defective patch
    _, reward, terminated, _, info = env.step(env.stop_action)
    assert info["recall"] == 1.0
    assert reward == DQN_STOP_BONUS


def test_stop_leaving_defect_is_penalized():
    env = make_env([1, 1, 0, 0])
    _, reward, _, _, info = env.step(env.stop_action)
    # recall 0, two missed defects
    assert reward == DQN_STOP_BONUS * 0.0 - DQN_MISS_PENALTY * 2


def test_new_defect_pays_reward_once():
    env = make_env([1, 0, 0, 0])
    _, r1, *_ = env.step(0)
    assert r1 == DQN_STEP_PENALTY + DQN_DEFECT_REWARD
    _, r2, *_ = env.step(0)  # re-inspect same patch
    assert r2 == DQN_STEP_PENALTY


def test_full_sweep_terminates_and_settles():
    env = make_env([1, 0, 0, 0])
    for a in range(env.n_patches):
        _, reward, terminated, _, info = env.step(a)
    assert terminated
    assert info["patches_inspected"] == env.n_patches
    assert info["recall"] == 1.0


def test_clean_image_stop_is_rewarded():
    env = make_env([0, 0, 0, 0])
    _, reward, _, _, info = env.step(env.stop_action)
    assert reward == DQN_STOP_BONUS
    assert info["recall"] == 1.0
