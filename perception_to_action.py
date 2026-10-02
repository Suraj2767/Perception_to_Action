import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
import random


# ============================================================
# CONNECT
# ============================================================

p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)


# ============================================================
# WORLD
# ============================================================

p.loadURDF("plane.urdf")

TABLE_TOP_Z = 0.60

BALL_RADIUS = 0.045

# Ball top surface
BALL_TOUCH_Z = TABLE_TOP_Z + (2 * BALL_RADIUS) + 0.005


# ============================================================
# TABLE
# ============================================================

table_collision = p.createCollisionShape(
    p.GEOM_BOX,
    halfExtents=[0.80, 0.65, 0.30]
)

table_visual = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=[0.80, 0.65, 0.30],
    rgbaColor=[0.45, 0.45, 0.48, 1]
)

p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=table_collision,
    baseVisualShapeIndex=table_visual,
    basePosition=[0.35, 0, 0.30]
)


# ============================================================
# KUKA IIWA
# ============================================================

robot = p.loadURDF(
    "kuka_iiwa/model.urdf",
    [0, 0, TABLE_TOP_Z],
    useFixedBase=True
)

print("KUKA joints:", p.getNumJoints(robot))

END_EFFECTOR = 6


# ============================================================
# CAMERA
# ============================================================

WIDTH = 640
HEIGHT = 480

camera_pos = [
    0.55,
    -1.55,
    1.65
]

camera_target = [
    0.55,
    0,
    TABLE_TOP_Z
]

view_matrix = p.computeViewMatrix(
    cameraEyePosition=camera_pos,
    cameraTargetPosition=camera_target,
    cameraUpVector=[0, 0, 1]
)

projection_matrix = p.computeProjectionMatrixFOV(
    fov=60,
    aspect=WIDTH / HEIGHT,
    nearVal=0.1,
    farVal=3.0
)


# ============================================================
# CREATE RANDOM RED BALL
# ============================================================

def create_ball():

    x = random.uniform(-0.10, 0.75)
    y = random.uniform(-0.35, 0.35)

    collision = p.createCollisionShape(
        p.GEOM_SPHERE,
        radius=BALL_RADIUS
    )

    visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=BALL_RADIUS,
        rgbaColor=[1, 0, 0, 1]
    )

    # Static ball so it does not roll away
    ball = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[
            x,
            y,
            TABLE_TOP_Z + BALL_RADIUS
        ]
    )

    p.changeDynamics(
        ball,
        -1,
        lateralFriction=0.8,
        rollingFriction=1.0,
        spinningFriction=1.0
    )

    return ball, np.array([
        x,
        y,
        TABLE_TOP_Z
    ])


# ============================================================
# PIXEL → WORLD
# ============================================================

def pixel_to_world(u, v):

    x_ndc = (u / (WIDTH - 1)) * 2.0 - 1.0
    y_ndc = 1.0 - (v / (HEIGHT - 1)) * 2.0

    projection = np.array(
        projection_matrix
    ).reshape(4, 4).T

    view = np.array(
        view_matrix
    ).reshape(4, 4).T

    inverse_matrix = np.linalg.inv(
        projection @ view
    )

    near = inverse_matrix @ np.array([
        x_ndc,
        y_ndc,
        -1,
        1
    ])

    far = inverse_matrix @ np.array([
        x_ndc,
        y_ndc,
        1,
        1
    ])

    near = near[:3] / near[3]
    far = far[:3] / far[3]

    ray = far - near

    t = (
        TABLE_TOP_Z - near[2]
    ) / ray[2]

    return near + t * ray


# ============================================================
# OPENCV RED DETECTION
# ============================================================

def detect_red_object():

    if not p.isConnected():
        return None, None

    camera_image = p.getCameraImage(
        WIDTH,
        HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    rgba = np.array(
        camera_image[2],
        dtype=np.uint8
    )

    rgba = rgba.reshape(
        HEIGHT,
        WIDTH,
        4
    )

    frame = cv2.cvtColor(
        rgba,
        cv2.COLOR_RGBA2BGR
    )

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    # Red HSV range
    lower_red1 = np.array([
        0,
        100,
        80
    ])

    upper_red1 = np.array([
        10,
        255,
        255
    ])

    lower_red2 = np.array([
        170,
        100,
        80
    ])

    upper_red2 = np.array([
        180,
        255,
        255
    ])

    mask1 = cv2.inRange(
        hsv,
        lower_red1,
        upper_red1
    )

    mask2 = cv2.inRange(
        hsv,
        lower_red2,
        upper_red2
    )

    mask = mask1 | mask2

    # Noise removal
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, frame

    contour = max(
        contours,
        key=cv2.contourArea
    )

    if cv2.contourArea(contour) < 50:
        return None, frame

    # Centroid
    M = cv2.moments(contour)

    if M["m00"] == 0:
        return None, frame

    cx = int(
        M["m10"] / M["m00"]
    )

    cy = int(
        M["m01"] / M["m00"]
    )

    # Convert pixel to 3D world position
    world_position = pixel_to_world(
        cx,
        cy
    )

    # Detection marker
    cv2.circle(
        frame,
        (cx, cy),
        8,
        (0, 255, 0),
        -1
    )

    cv2.putText(
        frame,
        f"RED OBJECT ({cx},{cy})",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )

    return world_position, frame


# ============================================================
# MOVE KUKA
# ============================================================

def move_arm(target, steps=240):

    if not p.isConnected():
        return False

    target = [
        float(target[0]),
        float(target[1]),
        float(target[2])
    ]

    # End effector pointing downward
    orientation = p.getQuaternionFromEuler([
        0,
        np.pi,
        0
    ])

    try:

        joint_positions = p.calculateInverseKinematics(
            robot,
            END_EFFECTOR,
            target,
            orientation
        )

    except Exception as e:

        print("IK error:", e)

        return False

    # Move all 7 joints
    for i in range(7):

        if not p.isConnected():
            return False

        p.setJointMotorControl2(
            robot,
            i,
            p.POSITION_CONTROL,
            targetPosition=joint_positions[i],
            force=300,
            maxVelocity=1.5
        )

    # Wait for movement
    for _ in range(steps):

        if not p.isConnected():
            return False

        p.stepSimulation()

        time.sleep(1 / 240)

    return True


# ============================================================
# CONTINUOUS LOOP
# ============================================================

run = 1

print()
print("========================================")
print("KUKA PERCEPTION-TO-ACTION SYSTEM")
print("========================================")
print()
print("Camera → OpenCV → 3D Position")
print("→ KUKA → TOUCH → COLLECT")
print()
print("Running continuously...")
print("Close PyBullet or press Ctrl+C to stop.")
print()


try:

    while p.isConnected():

        print()
        print("================================")
        print(f"RUN {run}")
        print("================================")

        # ----------------------------------------------------
        # CREATE RANDOM BALL
        # ----------------------------------------------------

        ball, actual_position = create_ball()

        print(
            "Random ball position:",
            np.round(
                actual_position,
                3
            )
        )

        # Let simulation settle
        for _ in range(60):

            if not p.isConnected():
                break

            p.stepSimulation()

            time.sleep(1 / 240)

        if not p.isConnected():
            break

        # ----------------------------------------------------
        # CAMERA DETECTION
        # ----------------------------------------------------

        detected_position, frame = detect_red_object()

        if not p.isConnected():
            break

        if detected_position is None:

            print(
                "WARNING: Ball not detected."
            )

            if frame is not None:

                cv2.imshow(
                    "Camera - OpenCV Detection",
                    frame
                )

                cv2.waitKey(300)

            if p.isConnected():

                p.removeBody(ball)

            print(
                "Retrying..."
            )

            continue

        print(
            "Camera detected red ball."
        )

        print(
            "Detected position:",
            np.round(
                detected_position,
                3
            )
        )

        # Show camera
        cv2.imshow(
            "Camera - OpenCV Detection",
            frame
        )

        cv2.waitKey(300)

        # ----------------------------------------------------
        # MOVE ABOVE BALL
        # ----------------------------------------------------

        above_object = [
            detected_position[0],
            detected_position[1],
            TABLE_TOP_Z + 0.30
        ]

        print(
            "KUKA moving above ball..."
        )

        if not move_arm(
            above_object,
            steps=240
        ):

            break

        # ----------------------------------------------------
        # DESCEND TOWARDS BALL
        # ----------------------------------------------------

        # Target is near the TOP of the ball.
        # Ball center = 0.645
        # Ball top    = 0.690
        #
        # Small 5 mm offset prevents hard collision.

        touch_object = [
            detected_position[0],
            detected_position[1],
            BALL_TOUCH_Z
        ]

        print(
            "KUKA descending to ball..."
        )

        if not move_arm(
            touch_object,
            steps=300
        ):

            break

        # ----------------------------------------------------
        # BALL TOUCH / COLLECTION
        # ----------------------------------------------------

        print(
            "SUCCESS: KUKA touched the ball!"
        )

        # Immediately remove ball
        if p.isConnected():

            p.removeBody(ball)

            print(
                "Ball collected!"
            )

        else:

            break

        # ----------------------------------------------------
        # MOVE ARM BACK UP
        # ----------------------------------------------------

        print(
            "KUKA moving back..."
        )

        if not move_arm(
            above_object,
            steps=240
        ):

            break

        print(
            f"RUN {run} COMPLETE"
        )

        run += 1

        time.sleep(0.5)


# ============================================================
# STOP
# ============================================================

except KeyboardInterrupt:

    print()
    print("========================================")
    print("PROGRAM STOPPED BY USER")
    print("========================================")


# ============================================================
# CLEANUP
# ============================================================

finally:

    cv2.destroyAllWindows()

    if p.isConnected():

        p.disconnect()

    print()
    print("Robot simulation closed.")