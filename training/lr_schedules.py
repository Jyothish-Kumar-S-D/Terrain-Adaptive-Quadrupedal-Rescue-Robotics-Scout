import math
from typing import Callable

class ExponentialDecaySchedule:
    def __init__(self, start: float, end: float, decay_rate: float = 5.0):
        self.start = float(start)
        self.end = float(end)
        self.decay_rate = float(decay_rate)
    def __call__(self, progress_remaining: float) -> float:
        t = 1.0 - progress_remaining
        return self.end + (self.start - self.end) * math.exp(-self.decay_rate * t)

