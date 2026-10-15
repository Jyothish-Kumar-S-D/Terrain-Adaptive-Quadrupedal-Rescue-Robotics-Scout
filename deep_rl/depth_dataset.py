"""Save depth-camera frames and their proprioceptive context separately."""

from pathlib import Path
import time

import numpy as np


class DepthFrameStore:
    def __init__(self, directory="depth_frames", interval=0.25):
        self.directory = Path(directory)
        self.interval = float(interval)
        self.last_saved = 0.0
        self.frame_index = 0
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, depth, state, force=False):
        now = time.monotonic()
        if not force and now - self.last_saved < self.interval:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.directory / f"depth_{stamp}_{self.frame_index:06d}.npz"
        np.savez_compressed(
            path,
            depth=np.asarray(depth, dtype=np.float32),
            state=np.asarray(state, dtype=np.float32),
            timestamp=np.float64(time.time()),
        )
        self.frame_index += 1
        self.last_saved = now
        return path
