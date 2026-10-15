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
    parser.add_argument("--initial-lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--final-lr", type=float, default=3e-5, help="Final learning rate")
    parser.add_argument("--n-steps", type=int, default=128, help="Steps per rollout")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    args = parser.parse_args()
    controller = DepthPPOController(args.model, seed=args.seed)
    path = controller.train(args.timesteps, initial_lr=args.initial_lr, final_lr=args.final_lr,
                            n_steps=args.n_steps, batch_size=args.batch_size,
                            gamma=args.gamma, ent_coef=args.ent_coef)
    print(f"Saved PPO policy to {path}.zip")


if __name__ == "__main__":
    main()
