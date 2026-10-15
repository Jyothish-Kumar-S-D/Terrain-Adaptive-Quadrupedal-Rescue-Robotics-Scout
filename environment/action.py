"""Action space helpers for the 26-D continuous gait-cycle action space."""

from __future__ import annotations

import numpy as np

from .config import ActionConfig
from .ik import CycleCommand, LEG_NAMES, LegStep, SIDE


def build_action_space():
    """Return (low, high, nominal_params) arrays for the 26-D action space."""
    cfg = ActionConfig()
    low = np.array([
        *cfg.phase_offset,
        *cfg.swing_frac,
        cfg.cycle_time,
        *cfg.step_height,
        *cfg.length_x,
        *cfg.length_y,
        *cfg.stance_z,
        cfg.stance_width,
    ], dtype=np.float32)
    high = np.array([
        1.0, 1.0, 1.0, 1.0,
        0.35, 0.35, 0.35, 0.35,
        1.8,
        0.16, 0.16, 0.16, 0.16,
        0.08, 0.08, 0.08, 0.08,
        0.04, 0.04, 0.04, 0.04,
        -0.16, -0.16, -0.16, -0.16,
        0.04,
    ], dtype=np.float32)
    nominal = np.array([
        0.25, 0.75, 0.50, 0.00,
        0.20, 0.20, 0.20, 0.20,
        1.0,
        0.12, 0.12, 0.12, 0.12,
        0.06, 0.06, 0.06, 0.06,
        0.0, 0.0, 0.0, 0.0,
        -0.21, -0.21, -0.21, -0.21,
        0.02,
    ], dtype=np.float32)
    return low, high, nominal


ACTION_LOW, ACTION_HIGH, NOMINAL_ACTION_PARAMS = build_action_space()
ACTION_PARAM_NAMES = [
    "phase_offset_FR", "phase_offset_FL", "phase_offset_RR", "phase_offset_RL",
    "swing_frac_FR", "swing_frac_FL", "swing_frac_RR", "swing_frac_RL",
    "cycle_time",
    "step_height_FR", "step_height_FL", "step_height_RR", "step_height_RL",
    "length_x_FR", "length_x_FL", "length_x_RR", "length_x_RL",
    "length_y_FR", "length_y_FL", "length_y_RR", "length_y_RL",
    "stance_z_FR", "stance_z_FL", "stance_z_RR", "stance_z_RL",
    "stance_width",
]


def normalize_action(params: np.ndarray) -> np.ndarray:
    """Map physical gait cycle parameters to normalized action in [-1.0, 1.0]."""
    p_arr = np.asarray(params, dtype=np.float32)
    norm = 2.0 * (p_arr - ACTION_LOW) / (ACTION_HIGH - ACTION_LOW) - 1.0
    return np.clip(norm, -1.0, 1.0).astype(np.float32)


def denormalize_action(action: np.ndarray) -> np.ndarray:
    """Map normalized action [-1.0, 1.0] to physical gait cycle parameters."""
    a_arr = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    return (ACTION_LOW + 0.5 * (a_arr + 1.0) * (ACTION_HIGH - ACTION_LOW)).astype(np.float32)


def get_nominal_cycle_action() -> np.ndarray:
    """Return nominal normalized action vector in [-1.0, 1.0]."""
    return normalize_action(NOMINAL_ACTION_PARAMS)


def action_to_cycle_command(action: np.ndarray) -> CycleCommand:
    """Decode normalized action [-1.0, 1.0]^26 into an executable CycleCommand."""
    params = denormalize_action(action)
    phase_offset = {
        "FR": float(params[0]), "FL": float(params[1]),
        "RR": float(params[2]), "RL": float(params[3]),
    }
    swing_frac = {
        "FR": float(params[4]), "FL": float(params[5]),
        "RR": float(params[6]), "RL": float(params[7]),
    }
    cycle_time = float(params[8])
    step_height = {
        "FR": float(params[9]), "FL": float(params[10]),
        "RR": float(params[11]), "RL": float(params[12]),
    }
    length_x = {
        "FR": float(params[13]), "FL": float(params[14]),
        "RR": float(params[15]), "RL": float(params[16]),
    }
    length_y = {
        "FR": float(params[17]), "FL": float(params[18]),
        "RR": float(params[19]), "RL": float(params[20]),
    }
    stance_z = {
        "FR": float(params[21]), "FL": float(params[22]),
        "RR": float(params[23]), "RL": float(params[24]),
    }
    stance_width = float(params[25])

    legs = {}
    for n in LEG_NAMES:
        legs[n] = LegStep(
            length_x=length_x[n],
            length_y=length_y[n],
            height=step_height[n],
            stance_z=stance_z[n],
            offset_y=SIDE[n] * stance_width,
            swing_frac=swing_frac[n],
        )

    mean_stance_z = float(np.mean(list(stance_z.values())))
    mean_swing_frac = float(np.mean(list(swing_frac.values())))

    return CycleCommand(
        legs=legs,
        cycle_time=cycle_time,
        stance_z=mean_stance_z,
        swing_frac=mean_swing_frac,
        stance_width=stance_width,
        phase_offset=phase_offset,
    )