"""Build and display a 3D point cloud from the robot's depth camera."""

import numpy as np

OBJECT_PALETTE = np.asarray([
    (0.55, 0.58, 0.62),  # ground
    (0.95, 0.20, 0.14),
    (0.14, 0.40, 0.96),
    (1.00, 0.62, 0.08),
    (0.12, 0.76, 0.34),
    (0.76, 0.24, 0.88),
    (0.08, 0.76, 0.78),
    (0.95, 0.38, 0.64),
    (0.67, 0.78, 0.12),
    (0.48, 0.31, 0.15),
    (0.98, 0.82, 0.65),
    (0.35, 0.36, 0.92),
], dtype=np.float32)


class PointCloudAccumulator:
    """Keep every distinct world-frame point from successive scans."""

    def __init__(self):
        self._points = {}

    def add(self, points, colors=None):
        """Add a scan without discarding points from earlier frames."""
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if colors is None:
            colors = np.tile((0.15, 0.65, 0.95), (len(points), 1))
        colors = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
        if len(colors) != len(points):
            raise ValueError("colors must contain one RGB value per point")
        finite = np.all(np.isfinite(points), axis=1)
        points, colors = points[finite], colors[finite]
        rounded = np.round(points, decimals=4)
        for point, color in zip(rounded, colors):
            self._points.setdefault(tuple(point.tolist()), tuple(color.tolist()))
        return len(self._points)

    def points(self):
        if not self._points:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(list(self._points), dtype=np.float32)

    def colors(self):
        if not self._points:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(list(self._points.values()), dtype=np.float32)


def depth_to_point_cloud(depth, camera, stride=8, max_depth=None, segmentation=None):
    """Convert a metric depth image to world-frame XYZ points.

    ``depth`` is the metric image returned by ``DepthCamera.capture``. The
    camera's forward, right, and up axes follow the orientation used by the
    depth camera renderer (local +X, -Y, and +Z respectively).
    """
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth must be a 2D image")
    if stride < 1:
        raise ValueError("stride must be at least 1")

    height, width = depth.shape
    rows = np.arange(0, height, stride)
    cols = np.arange(0, width, stride)
    sampled = depth[np.ix_(rows, cols)]

    far = camera.far if max_depth is None else min(float(max_depth), camera.far)
    valid = np.isfinite(sampled) & (sampled >= camera.near) & (sampled < far)
    if not np.any(valid):
        if segmentation is not None:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
        return np.empty((0, 3), dtype=np.float32)

    fov_y = np.deg2rad(camera.fov)
    aspect = camera.width / camera.height
    tan_half_y = np.tan(fov_y / 2.0)
    tan_half_x = tan_half_y * aspect
    pixel_x = (cols.astype(np.float32) + 0.5) / width * 2.0 - 1.0
    pixel_y = 1.0 - (rows.astype(np.float32) + 0.5) / height * 2.0
    xx, yy = np.meshgrid(pixel_x * tan_half_x, pixel_y * tan_half_y)

    # PyBullet's depth value is distance along the camera forward axis.
    z = sampled[valid]
    local = np.column_stack((
        z,
        -xx[valid] * z,
        yy[valid] * z,
    ))

    import pybullet as p

    state = p.getLinkState(
        camera.robot_id,
        camera.camera_link_index,
        computeForwardKinematics=True,
    )
    origin = np.asarray(state[4], dtype=np.float32)
    rotation = np.asarray(p.getMatrixFromQuaternion(state[5]), dtype=np.float32).reshape(3, 3)
    points = local @ rotation.T + origin
    if segmentation is None:
        return points

    segmentation = np.asarray(segmentation, dtype=np.int32)
    if segmentation.shape != depth.shape:
        raise ValueError("segmentation must have the same dimensions as depth")
    labels = segmentation[np.ix_(rows, cols)][valid]
    has_object = labels >= 0
    points = points[has_object]
    labels = labels[has_object]
    # PyBullet packs the body id in the low 24 bits and link id above it.
    body_ids = np.bitwise_and(labels, (1 << 24) - 1)
    colors = OBJECT_PALETTE[body_ids % len(OBJECT_PALETTE)]
    return points, colors


class PointCloudViewer:
    """Matplotlib 3D scatter view for a completed camera scan."""

    def __init__(self):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.figure = plt.figure("Robot Point Cloud", figsize=(8, 6))
        self.axes = self.figure.add_subplot(111, projection="3d")
        self.axes.set_title("Accumulated point cloud")
        self.axes.set_xlabel("X (m)")
        self.axes.set_ylabel("Y (m)")
        self.axes.set_zlabel("Z (m)")
        self.axes.set_box_aspect((1, 1, 0.7))
        self.scatter = self.axes.scatter([], [], [], s=2, c="#28a7e8", depthshade=False)
        self.axes.set_xlim(0, 8)
        self.axes.set_ylim(-4, 4)
        self.axes.set_zlim(-2, 4)
        self.figure.tight_layout()

    def update(self, points, colors=None):
        """Display the complete accumulated cloud."""
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        self.scatter._offsets3d = (
            points[:, 0] if len(points) else [],
            points[:, 1] if len(points) else [],
            points[:, 2] if len(points) else [],
        )
        if colors is not None and len(points):
            colors = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
            if len(colors) != len(points):
                raise ValueError("colors must contain one RGB value per point")
            self.scatter.set_color(colors)
        if len(points):
            lower = points.min(axis=0)
            upper = points.max(axis=0)
            margin = np.maximum((upper - lower) * 0.05, 0.25)
            self.axes.set_xlim(lower[0] - margin[0], upper[0] + margin[0])
            self.axes.set_ylim(lower[1] - margin[1], upper[1] + margin[1])
            self.axes.set_zlim(lower[2] - margin[2], upper[2] + margin[2])
        self.axes.set_title(f"Accumulated point cloud ({len(points)} points)")
        self.figure.canvas.draw()

    def show(self):
        """Open the final plot and wait for the user to close it."""
        self.plt.show(block=True)

    def is_open(self):
        return self.plt.fignum_exists(self.figure.number)
