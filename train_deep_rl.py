"""Train the depth PPO policy.

Run with the project's Python 3.10 environment:
    venv/bin/python train_deep_rl.py --timesteps 100000

Training runs PyBullet in DIRECT mode, captures depth images, and scores
locomotion using path following, speed, stability, and progress rewards.
"""

import argparse

from deep_rl import DepthPPOController


def main():
    parser = argparse.ArgumentParser(description="Train A1 depth PPO policy")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default="models/a1_depth_ppo")
    args = parser.parse_args()
    controller = DepthPPOController(args.model, seed=args.seed)
    path = controller.train(args.timesteps)
    print(f"Saved PPO policy to {path}.zip")


if __name__ == "__main__":
    main()
