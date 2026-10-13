"""
a1_ik.py -- analytic IK + per-cycle, per-leg crawl (wave) gait generator
for the Unitree A1 URDF. Pure math: no pybullet import.

Public frame convention (what YOU pass in)
------------------------------------------
Hip-relative *physical* body frame:  +x = forward, +y = left, +z = up.
  length_x > 0  -> leg walks the body forward   (< 0 backward)
  length_y > 0  -> leg walks the body to the left (< 0 to the right)
  height        -> swing foot clearance above the stance plane

Internally leg_ik() keeps the original script's math, whose x axis is
mirrored w.r.t. the physical frame (positive thigh angle moves the foot
toward -x for a +Y joint axis). That is why the old script needed
FORWARD_SIGN = -1. Here it is absorbed once in IK_X_SIGN. If the robot
walks backward for a positive length_x, flip IK_X_SIGN to +1.

Per-cycle parameters (CycleCommand)
-----------------------------------
  legs[leg]      LegStep(length_x, length_y, height)  -- each of FR/FL/RR/RL
  cycle_time     s per gait cycle
  stance_z       nominal foot depth below hip (negative)
  swing_frac     fraction of a leg's cycle spent in swing (<= 0.25 keeps
                 the crawl property: only one leg airborne at a time)
  cg_offset_x    static foot shift in x (negative = feet behind hips,
                 pulls the trunk forward over the support polygon)
  phase_offset   per-leg phase offsets (default RL->FL->RR->FR)

Timing / continuity
-------------------
A new CycleCommand becomes active at the next cycle boundary. Each leg
latches its step parameters at the start of ITS OWN swing, and the swing
starts from wherever the foot actually is, so changing step lengths
between cycles never makes a foot jump. Change phase_offset only if you
accept a transient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Optional, Tuple

# ----------------------------------------------------------------------
# Kinematic constants (from the parent-frame offsets in the URDF dump)
# ----------------------------------------------------------------------
L1 = 0.083165
L2 = math.sqrt(0.003237**2 + 0.022327**2 + 0.172674**2)
L3 = math.sqrt(0.006435**2 + 0.092612**2)
MAX_REACH = L2 + L3

LEG_NAMES = ("FR", "FL", "RR", "RL")
SIDE = {"FR": -1, "FL": 1, "RR": -1, "RL": 1}  # +1 left, -1 right

IK_X_SIGN = -1  # physical x -> IK-frame x (see module docstring)

DEFAULT_PHASE_OFFSET = {"RL": 0.0, "FL": 0.75, "RR": 0.5, "FR": 0.25}

Vec3 = Tuple[float, float, float]


# ----------------------------------------------------------------------
# Analytic IK / FK (IK-frame; same math as the original script)
# ----------------------------------------------------------------------
def leg_ik(x, y, z, side_sign, l1=L1, l2=L2, l3=L3):
    """Foot (x, y, z) relative to hip origin, IK frame -> (q1, q2, q3)."""
    l1s = side_sign * l1

    d = max(0.0, y * y + z * z - l1s * l1s)
    b = -math.sqrt(d)  # vertical extent of the leg in the pitch plane

    q1 = math.atan2(z, y) - math.atan2(b, l1s)  # hip ab/ad

    a = x
    r2 = min(a * a + b * b, (l2 + l3) ** 2)  # clamp to workspace
    cos_q3 = max(-1.0, min(1.0, (r2 - l2 * l2 - l3 * l3) / (2 * l2 * l3)))
    q3 = -math.acos(cos_q3)  # knee bends the A1 way

    phi = math.atan2(a, -b)
    q2 = phi - math.atan2(l3 * math.sin(q3), l2 + l3 * math.cos(q3))
    return q1, q2, q3


def leg_fk(q1, q2, q3, side_sign, l1=L1, l2=L2, l3=L3) -> Vec3:
    """Inverse of leg_ik (IK frame). Used for the self-test."""
    l1s = side_sign * l1
    a = l2 * math.sin(q2) + l3 * math.sin(q2 + q3)
    b = -(l2 * math.cos(q2) + l3 * math.cos(q2 + q3))
    y = l1s * math.cos(q1) - b * math.sin(q1)
    z = l1s * math.sin(q1) + b * math.cos(q1)
    return a, y, z


def foot_to_joints(leg: str, dx: float, dy: float, z: float):
    """Physical hip-relative foot position -> joint angles.

    dx: fore-aft (+ forward). dy: lateral offset from the neutral hip
    offset (+ left). z: depth (negative below hip).
    """
    side = SIDE[leg]
    return leg_ik(IK_X_SIGN * dx, side * L1 + dy, z, side)


# ----------------------------------------------------------------------
# Command structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LegStep:
    length_x: float = 0.08  # fore-aft stride (+ forward)
    length_y: float = 0.0   # lateral stride (+ left)
    height: float = 0.10    # swing clearance
    stance_z: Optional[float] = None  # optional per-leg stance depth
    offset_y: float = 0.0   # lateral foot stance offset (+ left, - right)


@dataclass
class CycleCommand:
    legs: Dict[str, LegStep]
    cycle_time: float = 1.0
    stance_z: float = -0.21
    swing_frac: float = 0.15
    cg_offset_x: float = -0.065
    stance_width: float = 0.0
    phase_offset: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PHASE_OFFSET)
    )

    def __post_init__(self):
        missing = set(LEG_NAMES) - set(self.legs)
        if missing:
            raise ValueError(f"legs missing entries for {sorted(missing)}")
        missing = set(LEG_NAMES) - set(self.phase_offset)
        if missing:
            raise ValueError(f"phase_offset missing entries for {sorted(missing)}")
        if self.cycle_time <= 0.0:
            raise ValueError("cycle_time must be > 0")
        if not 0.0 < self.swing_frac < 1.0:
            raise ValueError("swing_frac must be in (0, 1)")
        if not -MAX_REACH < self.stance_z < 0.0:
            raise ValueError(f"stance_z must be in (-{MAX_REACH:.3f}, 0)")

    # -- convenience constructors ------------------------------------
    @classmethod
    def uniform(cls, length_x=0.08, length_y=0.0, height=0.10, **kw):
        """Same LegStep on all four legs."""
        step = LegStep(length_x, length_y, height)
        return cls(legs={n: step for n in LEG_NAMES}, **kw)

    @classmethod
    def differential(cls, length_x=0.06, turn=0.0, height=0.12,
                     length_y=0.0, leg_z=None, stance_width=0.02, **kw):
        """Skid-steer style turn with bounded outer stride and inner leg deceleration.
        
        Guarantees outer legs NEVER over-stride beyond length_x into kinematic singularity,
        while inner legs decelerate proportionally to execute the turn cleanly.
        """
        legs = {}
        turn = max(-1.0, min(1.0, float(turn)))
        for n in LEG_NAMES:
            # For turn > 0 (CCW/left): right legs (SIDE < 0) are outer, left legs (SIDE > 0) are inner
            # For turn < 0 (CW/right): left legs (SIDE > 0) are outer, right legs (SIDE < 0) are inner
            is_outer = (turn >= 0 and SIDE[n] < 0) or (turn < 0 and SIDE[n] > 0)
            scale = 1.0 if is_outer else max(0.15, 1.0 - abs(turn))
            z_n = leg_z.get(n) if leg_z is not None else None
            y_off = SIDE[n] * stance_width
            legs[n] = LegStep(length_x * scale, length_y, height, stance_z=z_n, offset_y=y_off)
        return cls(legs=legs, stance_width=stance_width, **kw)

    def with_leg(self, leg: str, **step_kw) -> "CycleCommand":
        """Copy of this command with one leg's LegStep fields changed."""
        legs = dict(self.legs)
        legs[leg] = replace(legs[leg], **step_kw)
        return replace(self, legs=legs)


# ----------------------------------------------------------------------
# Gait generator
# ----------------------------------------------------------------------
class _LegState:
    __slots__ = ("step", "z", "cg", "in_swing", "pos", "swing_from", "offset_y")

    def __init__(self):
        self.step = LegStep()
        self.z = -0.22
        self.cg = 0.0
        self.offset_y = 0.0
        self.in_swing = False
        self.pos: Vec3 = (0.0, 0.0, -0.22)
        self.swing_from: Vec3 = self.pos


def _smoothstep(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


class CrawlGait:
    """Crawl gait driven by CycleCommand(s).

    gait = CrawlGait(cmd0, cycle_provider=fn)
    each tick: joints = gait.update(dt)   # {leg: (q1, q2, q3)}

    cycle_provider(cycle_index) -> CycleCommand | None is called at each
    cycle boundary (None = keep the current command). gait.set_next_cycle
    (cmd) queues a command for the next boundary and takes priority.
    blend_time eases the feet from neutral into the gait on start-up.
    """

    def __init__(self, command: CycleCommand,
                 cycle_provider: Optional[Callable[[int], Optional[CycleCommand]]] = None,
                 blend_time: float = 0.5):
        self.cmd = command
        self.cycle_provider = cycle_provider
        self.blend_time = blend_time
        self._pending: Optional[CycleCommand] = None

        self.t = 0.0
        self.cycle_index = 0
        # start just after RL's swing window so no leg begins mid-swing
        self.cycle_phase = command.swing_frac

        self._neutral: Vec3 = (command.cg_offset_x, 0.0, command.stance_z)
        self.legs: Dict[str, _LegState] = {n: _LegState() for n in LEG_NAMES}
        self.foot_targets: Dict[str, Vec3] = {}
        self.joint_targets: Dict[str, Tuple[float, float, float]] = {}

        for n, st in self.legs.items():
            self._latch(n, st, command)
            lp = (self.cycle_phase + command.phase_offset[n]) % 1.0
            st.pos = self._stance_pos(st, self._stance_s(lp, command))
            st.swing_from = st.pos

    # -- public API ---------------------------------------------------
    def set_next_cycle(self, cmd: CycleCommand) -> None:
        self._pending = cmd

    @property
    def swinging(self):
        return [n for n, st in self.legs.items() if st.in_swing]

    def update(self, dt: float,
               actual_foot_pos: Optional[Dict[str, Vec3]] = None
               ) -> Dict[str, Tuple[float, float, float]]:
        self.t += dt
        self.cycle_phase += dt / self.cmd.cycle_time
        while self.cycle_phase >= 1.0:
            self.cycle_phase -= 1.0
            self.cycle_index += 1
            self._begin_cycle()

        cmd = self.cmd
        w = 1.0 if self.blend_time <= 0 else _smoothstep(self.t / self.blend_time)
        nx, ny, nz = self._neutral

        for n, st in self.legs.items():
            lp = (self.cycle_phase + cmd.phase_offset[n]) % 1.0
            if lp < cmd.swing_frac:
                if not st.in_swing:
                    self._latch(n, st, cmd)
                    st.in_swing = True
                    # Obstacle-aware swing start: if foot is resting on an elevated obstacle,
                    # lift off from actual contact height rather than jumping down into the obstacle.
                    if actual_foot_pos is not None and n in actual_foot_pos:
                        af = actual_foot_pos[n]
                        st.swing_from = (af[0], af[1], max(st.pos[2], af[2]))
                    else:
                        st.swing_from = st.pos
                st.pos = self._swing_pos(st, lp / cmd.swing_frac)
            else:
                st.in_swing = False
                st.pos = self._stance_pos(st, self._stance_s(lp, cmd))

            x = nx + w * (st.pos[0] - nx)
            y = ny + w * (st.pos[1] - ny)
            z = nz + w * (st.pos[2] - nz)
            self.foot_targets[n] = (x, y, z)
            self.joint_targets[n] = foot_to_joints(n, x, y, z)

        return self.joint_targets

    # -- internals ----------------------------------------------------
    def _begin_cycle(self):
        nxt, self._pending = self._pending, None
        if nxt is None and self.cycle_provider is not None:
            nxt = self.cycle_provider(self.cycle_index)
        if nxt is not None:
            self.cmd = nxt

    @staticmethod
    def _latch(leg, st: _LegState, cmd: CycleCommand):
        st.step = cmd.legs[leg]
        st.z = cmd.legs[leg].stance_z if cmd.legs[leg].stance_z is not None else cmd.stance_z
        st.cg = cmd.cg_offset_x
        st.offset_y = cmd.legs[leg].offset_y if cmd.legs[leg].offset_y != 0.0 else (SIDE[leg] * cmd.stance_width)

    @staticmethod
    def _stance_s(lp, cmd):
        return (lp - cmd.swing_frac) / (1.0 - cmd.swing_frac)

    @staticmethod
    def _stance_pos(st: _LegState, s: float) -> Vec3:
        # foot sweeps +L/2 -> -L/2 (front to back for +length_x), which
        # is what pushes the body in the commanded direction
        s = max(0.0, min(1.0, s))
        k = 0.5 - s
        return (st.cg + st.step.length_x * k, st.offset_y + st.step.length_y * k, st.z)

    @staticmethod
    def _swing_pos(st: _LegState, s: float) -> Vec3:
        s = max(0.0, min(1.0, s))
        e = 0.5 * (1.0 - math.cos(math.pi * s))  # zero velocity at both ends
        fx, fy, fz = st.swing_from
        tx = st.cg + 0.5 * st.step.length_x
        ty = st.offset_y + 0.5 * st.step.length_y
        tz = st.z
        # High-stepping clearance trajectory for pyramids:
        # Rapid liftoff (power 0.40) ensures >60% clearance at s=0.10 to clear 8cm pyramid ridges
        clearance = math.sin(math.pi * s) ** 0.40 if 0.0 < s < 1.0 else 0.0
        return (
            fx + (tx - fx) * e,
            fy + (ty - fy) * e,
            fz + (tz - fz) * e + st.step.height * clearance,
        )


# ----------------------------------------------------------------------
# Self-test:  python a1_ik.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import random

    worst = 0.0
    for _ in range(2000):
        leg = random.choice(LEG_NAMES)
        side = SIDE[leg]
        x = random.uniform(-0.06, 0.06)
        y = side * L1 + random.uniform(-0.03, 0.03)
        z = random.uniform(-0.23, -0.15)
        a, yy, zz = leg_fk(*leg_ik(x, y, z, side), side)
        worst = max(worst, abs(a - x), abs(yy - y), abs(zz - z))
    print(f"IK->FK round-trip worst error: {worst:.2e} m")

    g = CrawlGait(CycleCommand.uniform(0.08, 0.02, 0.10))
    dt, worst_jump = 1 / 240, 0.0
    prev = None
    for i in range(240 * 6):
        g.update(dt)
        cur = dict(g.foot_targets)
        if prev is not None and i > 240:
            for n in LEG_NAMES:
                worst_jump = max(worst_jump, max(abs(cur[n][k] - prev[n][k]) for k in range(3)))
        prev = cur
        assert len(g.swinging) <= 1, "more than one leg swinging"
    print(f"max per-tick foot step: {worst_jump*1000:.2f} mm, <=1 leg swings: OK")