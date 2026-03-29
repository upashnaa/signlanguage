import os
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "data"

SIGNS = [
    "hello",
    "thanks",
    "yes",
    "no",
    "please",
    "sorry",
    "help",
    "good",
    "bad",
    "more",
    "stop",
    "love",
    "what",
    "where",
    "who",
    "how",
    "name",
    "friend",
    "eat",
    "drink",
    "water",
    "home",
    "work",
    "learn",
]

SEQUENCE_LENGTH = 30
NUM_SEQUENCES = 40
COUNTDOWN_SEC = 2


def extract_keypoints(results) -> np.ndarray:
    pose_flat = np.zeros(132, dtype=np.float32)
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        for i, p in enumerate(lm):
            base = i * 4
            pose_flat[base] = p.x
            pose_flat[base + 1] = p.y
            pose_flat[base + 2] = p.z
            pose_flat[base + 3] = p.visibility

    left_hand_flat = np.zeros(63, dtype=np.float32)
    if results.left_hand_landmarks:
        lm = results.left_hand_landmarks.landmark
        for i, p in enumerate(lm):
            base = i * 3
            left_hand_flat[base] = p.x
            left_hand_flat[base + 1] = p.y
            left_hand_flat[base + 2] = p.z

    right_hand_flat = np.zeros(63, dtype=np.float32)
    if results.right_hand_landmarks:
        lm = results.right_hand_landmarks.landmark
        for i, p in enumerate(lm):
            base = i * 3
            right_hand_flat[base] = p.x
            right_hand_flat[base + 1] = p.y
            right_hand_flat[base + 2] = p.z

    return np.concatenate([pose_flat, left_hand_flat, right_hand_flat])


def draw_hands(frame, results, mp_holistic, mp_drawing, mp_styles):
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )


def record_sequence(cap, holistic, sign, mp_holistic, mp_drawing, mp_styles):
    start = cv2.getTickCount()
    countdown_done = False
    recording = False
    frames_buffer = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            return False

        elapsed = (cv2.getTickCount() - start) / cv2.getTickFrequency()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        rgb.flags.writeable = True
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        draw_hands(frame, results, mp_holistic, mp_drawing, mp_styles)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return None

        if not countdown_done:
            cv2.putText(
                frame,
                f"Get ready: {sign}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (200, 241, 71),
                2,
                cv2.LINE_AA,
            )
            if elapsed >= COUNTDOWN_SEC:
                countdown_done = True
                recording = True
                frame_idx = 0
                frames_buffer = []
        elif recording:
            kp = extract_keypoints(results)
            frames_buffer.append(kp)
            cv2.putText(
                frame,
                f"Recording {sign}: frame {frame_idx + 1}/{SEQUENCE_LENGTH}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 241, 71),
                2,
                cv2.LINE_AA,
            )
            frame_idx += 1
            if len(frames_buffer) >= SEQUENCE_LENGTH:
                arr = np.stack(frames_buffer[:SEQUENCE_LENGTH], axis=0)
                return arr
        else:
            cv2.putText(
                frame,
                f"Get ready: {sign}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (200, 241, 71),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Handsign — Collect Data", frame)

    return False


def prompt_continue_or_quit(cap, sign):
    msg = f"Done with {sign}, press Q to stop or any other key to continue"
    while True:
        ret, frame = cap.read()
        if not ret:
            return "quit"
        cv2.putText(
            frame,
            msg,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Handsign — Collect Data", frame)
        k = cv2.waitKey(0) & 0xFF
        if k == ord("q"):
            return "quit"
        return "continue"


def main():
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.", file=sys.stderr)
        sys.exit(1)

    collected_signs = []

    try:
        for sign in SIGNS:
            sign_dir = DATA_ROOT / sign
            sign_dir.mkdir(parents=True, exist_ok=True)

            for seq_num in range(NUM_SEQUENCES):
                result = record_sequence(
                    cap,
                    holistic,
                    sign,
                    mp_holistic,
                    mp_drawing,
                    mp_styles,
                )
                if result is None:
                    _write_labels(collected_signs)
                    return
                if result is False:
                    _write_labels(collected_signs)
                    return
                out_path = sign_dir / f"{seq_num}.npy"
                np.save(out_path, result)

            collected_signs.append(sign)
            action = prompt_continue_or_quit(cap, sign)
            if action == "quit":
                _write_labels(collected_signs)
                return

        _write_labels(collected_signs)
    finally:
        holistic.close()
        cap.release()
        cv2.destroyAllWindows()


def _write_labels(signs: list) -> None:
    labels_path = SCRIPT_DIR / "labels.txt"
    with open(labels_path, "w", encoding="utf-8") as f:
        for s in signs:
            f.write(f"{s}\n")
    print(f"Wrote {labels_path}")


if __name__ == "__main__":
    os.chdir(SCRIPT_DIR)
    main()
