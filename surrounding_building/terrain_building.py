import pybullet as p


def create_pyramid(position, size=1.5, height=0.5, color=[0.8, 0.3, 0.1, 1.0]):
    half = size / 2.0

    vertices = [
        (-half, -half, 0.0),
        ( half, -half, 0.0),
        ( half,  half, 0.0),
        (-half,  half, 0.0),
        (0.0, 0.0, height),
    ]

    indices = [
        0, 1, 4,
        1, 2, 4,
        2, 3, 4,
        3, 0, 4,

        0, 3, 2,
        0, 2, 1,
    ]

    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        vertices=vertices,
        indices=indices
    )

    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        vertices=vertices,
        indices=indices,
        rgbaColor=color
    )

    pyramid_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position
    )

    return pyramid_id


def create_surroundings():
    """Creates 10 pyramids in the surroundings of the robot."""
    configs = [
        # (position, size, height, color)
        (( 3.0,  0.0, 0.01), 1.5, 0.50, [0.85, 0.35, 0.15, 1.0]),  # Directly in front
        (( 2.5,  2.2, 0.01), 1.2, 0.40, [0.75, 0.45, 0.20, 1.0]),  # Front-left
        (( 2.8, -2.0, 0.01), 1.3, 0.45, [0.80, 0.30, 0.25, 1.0]),  # Front-right
        (( 5.0,  1.2, 0.01), 1.8, 0.60, [0.90, 0.40, 0.10, 1.0]),  # Mid-distance left
        (( 5.5, -1.5, 0.01), 1.6, 0.55, [0.70, 0.35, 0.30, 1.0]),  # Mid-distance right
        (( 7.5,  0.5, 0.01), 2.0, 0.70, [0.85, 0.50, 0.20, 1.0]),  # Far-front center
        (( 7.0, -3.0, 0.01), 1.4, 0.50, [0.65, 0.40, 0.35, 1.0]),  # Far-right
        (( 6.8,  3.5, 0.01), 1.5, 0.50, [0.80, 0.45, 0.15, 1.0]),  # Far-left
        ((-2.0,  2.5, 0.01), 1.2, 0.40, [0.75, 0.30, 0.20, 1.0]),  # Rear-left
        ((-2.5, -2.2, 0.01), 1.3, 0.45, [0.70, 0.40, 0.25, 1.0]),  # Rear-right
    ]

    pyramid_ids = []
    for pos, size, height, color in configs:
        pid = create_pyramid(position=pos, size=size, height=height, color=color)
        pyramid_ids.append(pid)

    return pyramid_ids


def create_terrain():
    return create_surroundings()