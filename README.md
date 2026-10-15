# Terrain-Adaptive Quadrupedal Rescue Robotics Scout

A reinforcement learning project for training a Unitree A1 quadruped robot to navigate complex terrains using depth camera observations and PPO.

## Requirements

- Python 3.10
- PyBullet
- Stable-Baselines3
- PyTorch
- OpenCV
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── train_deep_rl.py      # Training script for PPO policy
├── main.py               # Simulation with live GUI, video recording, point cloud
├── scripts/
│   └── view_pointcloud.py  # View saved point clouds
├── deep_rl/              # RL controller and policy
├── path_training/        # Path generation and rewards
├── depth_processing/     # Depth camera processing
├── point_cloud/          # Point cloud accumulation and viewing
├── Inverse_Kinematics/   # A1 inverse kinematics
├── surrounding_building/ # Terrain generation
└── urdf/                 # A1 URDF model
```

## Training

Train the depth-based PPO policy:

```bash
# Basic training
python train_deep_rl.py --timesteps 100000

# With custom hyperparameters
python train_deep_rl.py --timesteps 200000 --initial-lr 5e-4 --final-lr 1e-5 --n-steps 256 --batch-size 128
```

### Training Features
- **Learning rate scheduling**: Exponential decay from initial to final LR
- **Best model saving**: Saves model with highest composite score (reward + episode length - loss)
- **Randomized paths**: Each episode gets a new random path with varied curvature
- **Domain randomization**: Terrain, gates, and path vary per episode
- **Reward shaping**: Path following, velocity alignment, stability, progress, fall penalty

Output: `models/a1_depth_ppo.best.zip` (best model) and `models/a1_depth_ppo.zip` (final model)

## Simulation

Run the simulation with live GUI:

```bash
python main.py
```

### Simulation Features
- **Live PyBullet GUI** with robot and terrain
- **Orbiting camera video recording** (saved to `videos/simulation_*.mp4`)
- **Periodic point cloud saving** (saved to `point_clouds/pointcloud_*.npz` every 5 seconds)
- **Final point cloud display** on exit (interactive matplotlib 3D viewer)
- **Graceful shutdown** on Ctrl+C or window close - saves video and point cloud

### Keyboard Controls
- `Ctrl+C` - Graceful shutdown with save
- Close GUI window - Graceful shutdown with save

## Viewing Saved Point Clouds

View previously saved point clouds:

```bash
# View latest point cloud
python scripts/view_pointcloud.py

# View specific file
python scripts/view_pointcloud.py point_clouds/pointcloud_20240101_120000.npz

# List all saved clouds
python scripts/view_pointcloud.py --list
```

## Key Components

### Reward Function (`path_training/task.py`)
- Path adherence (Gaussian)
- Path deviation penalty (linear + quadratic for far off-path)
- Velocity alignment with path tangent
- Stability (roll/pitch penalty)
- Progress along path
- Goal bonus
- Fall penalty (-10.0, terminates episode)
- Living penalty (-0.005/step)

### Environment (`deep_rl/controller.py`)
- Depth + proprioceptive observations (64×48 depth + 15 state)
- Custom CNN+MLP policy network
- Fall detection: height < 0.12m OR roll/pitch > 0.7 rad (~40°)
- Off-path termination with small positive reward
- Randomized gates aligned with path tangent

### Terrain (`surrounding_building/terrain_building.py`)
- Small grippy pyramids (0.08-0.18m size, friction 1.5)
- Covers full path + 2m margin
- Spawn area kept flat

## Output Directories

- `models/` - Trained PPO checkpoints
- `videos/` - Simulation recordings (MP4)
- `point_clouds/` - Accumulated point clouds (NPZ)
- `depth_frames/` - Raw depth frames for dataset

## Notes

- SIGKILL (kill -9) cannot be caught - use Ctrl+C or window close for clean saves
- Point clouds are saved periodically (every 5s) and on exit
- Video recorder uses hardware OpenGL renderer for performance
- Training uses PyBullet DIRECT mode (headless) for speed
