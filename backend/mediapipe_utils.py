import numpy as np

class MockHolistic:
    def process(self, image):
        class MockResults:
            pose_landmarks = None
            left_hand_landmarks = None
            right_hand_landmarks = None
        return MockResults()
    def close(self):
        pass

def extract_keypoints(results) -> np.ndarray:
    """
    Extract pose (33×4), left hand (21×3), right hand (21×3) from Holistic results.
    Returns flat array of shape (258,). Face landmarks are excluded.
    """
    pose_flat = np.zeros(132, dtype=np.float32)
    if getattr(results, "pose_landmarks", None):
        lm = results.pose_landmarks.landmark
        for i, p in enumerate(lm):
            base = i * 4
            pose_flat[base] = p.x
            pose_flat[base + 1] = p.y
            pose_flat[base + 2] = p.z
            pose_flat[base + 3] = p.visibility

    left_hand_flat = np.zeros(63, dtype=np.float32)
    if getattr(results, "left_hand_landmarks", None):
        lm = results.left_hand_landmarks.landmark
        for i, p in enumerate(lm):
            base = i * 3
            left_hand_flat[base] = p.x
            left_hand_flat[base + 1] = p.y
            left_hand_flat[base + 2] = p.z

    right_hand_flat = np.zeros(63, dtype=np.float32)
    if getattr(results, "right_hand_landmarks", None):
        lm = results.right_hand_landmarks.landmark
        for i, p in enumerate(lm):
            base = i * 3
            right_hand_flat[base] = p.x
            right_hand_flat[base + 1] = p.y
            right_hand_flat[base + 2] = p.z

    return np.concatenate([pose_flat, left_hand_flat, right_hand_flat])

def get_mediapipe_model():
    try:
        import mediapipe as mp
        return mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except (ImportError, AttributeError):
        return MockHolistic()
