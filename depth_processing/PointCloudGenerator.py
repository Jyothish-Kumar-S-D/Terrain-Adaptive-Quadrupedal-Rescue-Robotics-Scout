'''
Copyright (c) 2026 SivaPA08

Permission is granted to use, copy, modify, and distribute this code,
including for commercial purposes, provided this notice is retained.

This code is provided "as is", without warranty or liability.
'''

import numpy as np
import pybullet as p
import cv2
from typing import Optional, Tuple
from depth_processing.Depth_Processing import DepthCamera


class RGBCamera:
    """Captures RGB images from the robot's camera."""
    
    def __init__(
        self,
        robot_id: int,
        camera_name: str = "cam_L_joint",
        width: int = 640,
        height: int = 640,
        fov: float = 90.0,
        near: float = 0.05,
        far: float = 10.0,
    ):
        self.robot_id = robot_id
        self.width = width
        self.height = height
        self.near = near
        self.far = far

        self.camera_link_index = self._find_link(camera_name)

        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=width / height,
            nearVal=near,
            farVal=far,
        )

    def _find_link(self, camera_name: str) -> int:
        for j in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, j)
            name = info[1].decode("utf-8")
            if name == camera_name:
                return j
        raise ValueError(f"Camera joint '{camera_name}' not found")

    def _get_view_matrix(self) -> list:
        state = p.getLinkState(
            self.robot_id,
            self.camera_link_index,
            computeForwardKinematics=True,
        )

        camera_pos = np.array(state[4])
        camera_orn = state[5]

        rotation = np.array(
            p.getMatrixFromQuaternion(camera_orn)
        ).reshape(3, 3)

        forward = rotation @ np.array([1.0, 0.0, 0.0])
        up = rotation @ np.array([0.0, 0.0, 1.0])
        target = camera_pos + forward

        return p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target,
            cameraUpVector=up,
        )

    def get_camera_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns camera position and orientation (as rotation matrix) in world frame."""
        state = p.getLinkState(
            self.robot_id,
            self.camera_link_index,
            computeForwardKinematics=True,
        )
        camera_pos = np.array(state[4])
        camera_orn = state[5]
        rotation = np.array(
            p.getMatrixFromQuaternion(camera_orn)
        ).reshape(3, 3)
        return camera_pos, rotation

    def get_camera_forward_vector(self) -> np.ndarray:
        """Returns the camera's forward unit vector in world frame."""
        _, rotation = self.get_camera_pose()
        forward = rotation @ np.array([1.0, 0.0, 0.0])
        return forward / np.linalg.norm(forward)

    def capture(self) -> np.ndarray:
        """Capture RGB image from camera."""
        view_matrix = self._get_view_matrix()

        image = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        )

        rgb_buffer = np.asarray(image[2], dtype=np.uint8).reshape(
            self.height, self.width, 4
        )

        return rgb_buffer[:, :, :3]

    def capture_segmentation(self) -> np.ndarray:
        """Capture segmentation mask."""
        view_matrix = self._get_view_matrix()

        image = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        )

        seg_buffer = np.asarray(image[4], dtype=np.int32).reshape(
            self.height, self.width
        )

        return seg_buffer


class PointCloudGenerator:
    """Generates colored 3D point clouds from depth and RGB cameras."""

    def __init__(
        self,
        robot_id: int,
        camera_name: str = "cam_L_joint",
        width: int = 640,
        height: int = 640,
        fov: float = 90.0,
        near: float = 0.05,
        far: float = 10.0,
        downsample: int = 1,
    ):
        self.depth_camera = DepthCamera(
            robot_id, camera_name, width, height, fov, near, far
        )
        self.rgb_camera = RGBCamera(
            robot_id, camera_name, width, height, fov, near, far
        )
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far
        self.downsample = downsample

        self._fx = width / (2 * np.tan(np.deg2rad(fov) / 2))
        self._fy = self._fx
        self._cx = width / 2
        self._cy = height / 2

        self._compute_pixel_directions()

    def _compute_pixel_directions(self):
        """Pre-compute ray directions for each pixel in camera frame."""
        u, v = np.meshgrid(
            np.arange(self.width),
            np.arange(self.height),
            indexing='xy'
        )
        u = u.astype(np.float32)
        v = v.astype(np.float32)

        x = (u - self._cx) / self._fx
        y = (v - self._cy) / self._fy
        z = np.ones_like(x)

        rays = np.stack([x, y, z], axis=-1)
        norms = np.linalg.norm(rays, axis=-1, keepdims=True)
        self._ray_directions = rays / np.maximum(norms, 1e-9)

    def capture_point_cloud(
        self,
        world_frame: bool = True,
        max_points: Optional[int] = None,
        depth_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Capture depth + RGB and generate colored point cloud.

        Args:
            world_frame: If True, transform points to world frame
            max_points: Maximum number of points to return (randomly sampled)
            depth_threshold: Maximum depth to include (meters)

        Returns:
            points: (N, 3) array of 3D points
            colors: (N, 3) array of RGB colors [0, 255]
            depths: (N,) array of depth values
        """
        depth = self.depth_camera.capture()
        rgb = self.rgb_camera.capture()
        camera_pos, camera_rot = self.rgb_camera.get_camera_pose()

        if self.downsample > 1:
            depth = depth[::self.downsample, ::self.downsample]
            rgb = rgb[::self.downsample, ::self.downsample]
            ray_dirs = self._ray_directions[::self.downsample, ::self.downsample]
        else:
            ray_dirs = self._ray_directions

        valid = (depth > self.near) & (depth < self.far)
        if depth_threshold is not None:
            valid &= (depth < depth_threshold)

        valid_indices = np.where(valid)
        if len(valid_indices[0]) == 0:
            return np.empty((0, 3)), np.empty((0, 3)), np.empty((0,))

        depths = depth[valid_indices]
        colors = rgb[valid_indices]
        rays = ray_dirs[valid_indices]

        points_camera = rays * depths[:, np.newaxis]

        if world_frame:
            points_world = (camera_rot @ points_camera.T).T + camera_pos
            points = points_world
        else:
            points = points_camera

        if max_points is not None and len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            points = points[idx]
            colors = colors[idx]
            depths = depths[idx]

        return points, colors, depths

    def get_camera_info(self) -> dict:
        """Get current camera pose and orientation info."""
        pos, rot = self.rgb_camera.get_camera_pose()
        forward = self.rgb_camera.get_camera_forward_vector()
        return {
            'position': pos,
            'rotation': rot,
            'forward_vector': forward,
            'width': self.width,
            'height': self.height,
            'fov': self.fov,
            'near': self.near,
            'far': self.far,
        }

    def close(self):
        self.depth_camera.close()
        cv2.destroyAllWindows()


class PointCloudVisualizer:
    """Interactive 3D point cloud visualizer using Open3D or matplotlib."""

    def __init__(self, window_name: str = "Point Cloud", point_size: float = 2.0):
        self.window_name = window_name
        self.point_size = point_size
        self._use_open3d = False
        try:
            import open3d as o3d
            self._use_open3d = True
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name=window_name, width=1280, height=720)
            self.pcd = o3d.geometry.PointCloud()
            self.vis.add_geometry(self.pcd)
            render_option = self.vis.get_render_option()
            render_option.point_size = point_size
            render_option.background_color = np.array([0.1, 0.1, 0.1])
        except ImportError:
            print("Open3D not available, using matplotlib fallback")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            plt.ion()
            plt.show()

    def update(self, points: np.ndarray, colors: np.ndarray):
        """Update point cloud visualization."""
        if len(points) == 0:
            return

        if self._use_open3d:
            import open3d as o3d
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            self.vis.update_geometry(self.pcd)
            self.vis.poll_events()
            self.vis.update_renderer()
        else:
            self.ax.clear()
            self.ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=colors / 255.0, s=self.point_size, alpha=0.8
            )
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title(self.window_name)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def close(self):
        if self._use_open3d:
            self.vis.destroy_window()
        else:
            import matplotlib.pyplot as plt
            plt.close(self.fig)


def create_point_cloud_generator(
    robot_id: int,
    camera_name: str = "cam_L_joint",
    width: int = 640,
    height: int = 640,
    fov: float = 90.0,
    near: float = 0.05,
    far: float = 10.0,
    downsample: int = 2,
) -> PointCloudGenerator:
    """Factory function to create a PointCloudGenerator."""
    return PointCloudGenerator(
        robot_id, camera_name, width, height, fov, near, far, downsample
    )


if __name__ == "__main__":
    import pybullet as p
    import pybullet_data
    import time

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    robot = p.loadURDF("urdf/a1_unitree.urdf", [0, 0, 0.5])

    pc_gen = create_point_cloud_generator(robot, downsample=4)
    visualizer = PointCloudVisualizer("Robot Point Cloud")

    try:
        while True:
            p.stepSimulation()
            points, colors, depths = pc_gen.capture_point_cloud(
                world_frame=True,
                max_points=50000,
                depth_threshold=5.0
            )
            if len(points) > 0:
                visualizer.update(points, colors)
                info = pc_gen.get_camera_info()
                print(f"Points: {len(points)}, Camera pos: {info['position']}, "
                      f"Forward: {info['forward_vector']}")
            time.sleep(1/30)
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.close()
        pc_gen.close()
        p.disconnect()