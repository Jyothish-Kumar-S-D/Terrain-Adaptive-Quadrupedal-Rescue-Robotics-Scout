# Terrain-Adaptive Quadrupedal Rescue Robotics Scout

Simulation and learning toolkit for a Unitree A1 quadruped navigating procedurally generated terrain. The project combines analytic leg inverse kinematics and crawl gait control with PyBullet simulation, depth-camera perception, point-cloud reconstruction, and an optional PPO policy for terrain-aware locomotion.

<p align="center">
  <strong>PyBullet simulation · Path following · Depth perception · Reinforcement learning</strong>
</p>

## Project highlights

- Simulate a Unitree A1 robot with the included URDF and mesh assets.
- Generate randomized terrain, smooth paths, and obstacle gates for navigation tasks.
- Control the four-legged crawl gait with analytic inverse kinematics.
- Capture metric depth images and reconstruct colored, world-frame point clouds.
- Train a Stable-Baselines3 PPO policy using depth and robot-state observations.
- Run path-following environments through the Gymnasium API.

## Requirements

- Python **3.10** (recommended; the project includes a local `venv` configured for Python 3.10).
- A desktop session with OpenGL support for the interactive PyBullet simulation.
- PyTorch-compatible hardware for PPO training. A CUDA GPU is optional; CPU training is supported but can be slow.

Install the Python dependencies in a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. If your system requires a specific PyTorch build, install the matching PyTorch package for your operating system and hardware before installing the remaining requirements.

## Run the simulation

From the repository root, with the virtual environment active:

```bash
python main.py
```

The GUI simulation creates a randomized terrain course and path, places two gates along the route, and starts the A1 robot. It attempts to load `models/a1_depth_ppo.zip`; if no compatible checkpoint is available, it uses the analytic crawl gait. During the run it captures depth frames, accumulates a point cloud, and records an orbit-camera video under `videos/`. Close the PyBullet window or press `Ctrl+C` to stop; the accumulated point cloud is displayed after the simulation ends.

The project has no separate headless option for `main.py`. For headless environment rollouts, use the training script or environment test commands below.

## Train a depth PPO policy

```bash
python train_deep_rl.py --timesteps 100000
```

Training runs PyBullet in `DIRECT` mode and writes a Stable-Baselines3 checkpoint, normally `models/a1_depth_ppo.zip`. Useful options include:

```bash
python train_deep_rl.py --help
python train_deep_rl.py --timesteps 250000 --seed 42 --model models/experiment
```

The policy receives a 64 × 48 depth image and a 15-value robot/path state vector. Its four normalized actions control stride length, turning, swing height, and gait cycle timing. Training can take substantial time, depending on hardware and the number of requested timesteps.

## Run the path-following environment

The Gymnasium environment is registered as `QuadrupedPathTerrain-v0` when `path_training` is imported. For example:

```python
import gymnasium as gym
import path_training

env = gym.make("QuadrupedPathTerrain-v0", render_mode=None)
observation, info = env.reset(seed=42)
```

The environment can include depth observations and procedurally generated terrain. See `path_training/task.py` and `path_training/gym_env.py` for task configuration, observation, action, reward, and termination details.

## Tests and demos

The repository includes focused inverse-kinematics, gait, terrain-walking, and Gymnasium environment demos. Run them from the repository root:

```bash
python tests/IK_module_test.py --help
python tests/circle_wlk_test.py --help
python tests/terrain_circle_walk_test.py --help
python tests/test_path_terrain_env.py --help
```

Example environment smoke run:

```bash
python tests/test_path_terrain_env.py --policy nominal --episodes 1 --max-steps 50 --direct
```

Gymnasium API check:

```bash
python tests/test_path_terrain_env.py --check-env --direct
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `main.py` | Interactive A1 simulation, path following, depth capture, point-cloud display, and video recording |
| `train_deep_rl.py` | Command-line PPO training entry point |
| `Inverse_Kinematics/` | Analytic leg inverse kinematics and crawl-gait generator |
| `path_training/` | Path task generation and Gymnasium terrain environment |
| `surrounding_building/` | Procedural terrain generation |
| `depth_processing/` | PyBullet depth camera and metric depth conversion |
| `point_cloud/` | Depth-to-point-cloud projection, accumulation, and visualization |
| `deep_rl/` | PPO policy adapter and depth-frame storage |
| `urdf/`, `meshes/` | Robot model and visual assets |
| `models/` | Included or locally trained policy checkpoints |
| `tests/` | Simulation demos and environment checks |
| `Documentations/` | Inverse-kinematics module documentation |

## Data and generated files

Depth captures are written to `depth_frames/`; videos are written to `videos/`; PPO checkpoints are written to `models/`. These paths are ignored by Git, so generated data and newly trained models are not automatically included in commits. The repository currently includes `models/a1_depth_ppo.zip` as a checkpoint asset.

## License

The source file `depth_processing/Depth_Processing.py` contains a permissive copyright notice for that module. No repository-wide license file is currently present. Confirm the applicable license before redistributing or reusing the project as a whole.
