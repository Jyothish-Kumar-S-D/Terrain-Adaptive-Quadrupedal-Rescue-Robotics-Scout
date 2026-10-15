'''
Copyright (c) 2026 SivaPA08

Permission is granted to use, copy, modify, and distribute this code,
including for commercial purposes, provided this notice is retained.

This code is provided "as is", without warranty or liability.
'''

try:
    import pybullet as p
except ImportError:
    p = None


class DepthCamera:
    def __init__(
        self,
        robot_id,
        camera_name="cam_L_joint",
        width=640,
        height=640,
        fov=90.0,
        near=0.05,
        far=10.0,
    ):
        self.robot_id = robot_id
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far

        # Find camera joint automatically
        self.camera_link_index = self._find_link(camera_name)

        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=width / height,
            nearVal=near,
            farVal=far,
        )

    def _find_link(self, camera_name):
        for j in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, j)
            name = info[1].decode("utf-8")

            if name == camera_name:
                return j

        raise ValueError(
            f"Camera joint '{camera_name}' not found"
        )

    def _get_view_matrix(self):
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

        # Camera looks along local +X
        forward = rotation @ np.array([
            1.0,
            0.0,
            0.0
        ])

        # Camera up direction
        up = rotation @ np.array([
            0.0,
            0.0,
            1.0
        ])

        # Look forward from the camera
        target = camera_pos + forward

        return p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target,
            cameraUpVector=up,
        )

    def capture(self):
        depth, _ = self.capture_frame()
        return depth

    def capture_frame(self):
        """Return metric depth and the PyBullet object segmentation mask."""
        view_matrix = self._get_view_matrix()

        image = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_TINY_RENDERER,
        )

        depth_buffer = np.asarray(
            image[3],
            dtype=np.float32,
        ).reshape(
            self.height,
            self.width
        )

        # Convert depth buffer to metric depth
        depth = (
            self.near * self.far
            / (
                self.far
                - depth_buffer * (
                    self.far - self.near
                )
            )
        )

        segmentation = np.asarray(
            image[4],
            dtype=np.int32,
        ).reshape(self.height, self.width)

        return depth, segmentation

    def show(self, depth):
        display = np.clip(
            depth,
            self.near,
            self.far,
        )

        display = (
            255
            * (self.far - display)
            / (self.far - self.near)
        ).astype(np.uint8)

        cv2.imshow(
            "Depth Camera",
            display
        )

        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
        
