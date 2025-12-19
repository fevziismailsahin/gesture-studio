import cv2
import mediapipe as mp
import numpy as np
import time
import os

# ================= CONFIG =================
class Config:
    CAM_W, CAM_H = 1280, 720

    # Colors
    COLOR_BAR_BG = (10, 10, 10)
    COLOR_TEXT_PASSIVE = (180, 180, 180)
    COLOR_TEXT_ACTIVE = (0, 255, 255)
    COLOR_GUIDE = (255, 255, 0)
    COLOR_ALERT = (50, 50, 255)

    # Control speed (slow & precise)
    BASE_SPEED = 0.005
    MAX_STEP_ZOOM = 0.01
    MAX_STEP_BRIGHT = 1.0
    MAX_STEP_COLOR = 0.02

    DEADZONE = 40

    # Fist stability
    FIST_RELEASE_TIME = 0.25  # seconds


if not os.path.exists("Studio_Photos_Final"):
    os.makedirs("Studio_Photos_Final")


# ================= TEXT HELPER =================
def draw_text(img, text, x, y, size=0.7, color=(255, 255, 255), thickness=2):
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, size,
                (0, 0, 0), thickness + 3)
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, size,
                color, thickness)


# ================= HUD =================
def draw_hud(img, vals, active_mode):
    h, w, _ = img.shape
    bar_h = 50

    sub = img[0:bar_h, 0:w]
    bg = np.full(sub.shape, Config.COLOR_BAR_BG, dtype=np.uint8)
    cv2.addWeighted(bg, 0.85, sub, 0.15, 0, sub)

    sections = ["ZOOM", "BRIGHTNESS", "CONTRAST", "SATURATION"]
    step = w // 4

    for i, key in enumerate(sections):
        cx = int(step * i + step / 2)
        active = key == active_mode
        color = Config.COLOR_TEXT_ACTIVE if active else Config.COLOR_TEXT_PASSIVE

        val = vals[key]
        val_str = f"{int(val)}" if key == "BRIGHTNESS" else f"{val:.2f}"
        label = "BRIGHT" if key == "BRIGHTNESS" else key

        draw_text(img, f"{label}: {val_str}",
                  cx - 70, 35,
                  0.7 if active else 0.5, color)

        if active:
            cv2.line(img, (cx - 60, 45), (cx + 60, 45), color, 3)


# ================= GUIDE ARROWS =================
def draw_guides(img, cx, cy, mode=None):
    if mode is None:
        draw_text(img, "MOVE HAND", cx - 70, cy + 140, 0.6, (200, 200, 200))

        cv2.arrowedLine(img, (cx, cy - 80), (cx, cy - 130), Config.COLOR_GUIDE, 4)
        draw_text(img, "BRIGHTNESS", cx - 70, cy - 140, 0.6, Config.COLOR_GUIDE)

        cv2.arrowedLine(img, (cx, cy + 80), (cx, cy + 130), Config.COLOR_GUIDE, 4)
        draw_text(img, "SATURATION", cx - 70, cy + 160, 0.6, Config.COLOR_GUIDE)

        cv2.arrowedLine(img, (cx + 80, cy), (cx + 130, cy), Config.COLOR_GUIDE, 4)
        draw_text(img, "ZOOM", cx + 140, cy + 10, 0.6, Config.COLOR_GUIDE)

        cv2.arrowedLine(img, (cx - 80, cy), (cx - 130, cy), Config.COLOR_GUIDE, 4)
        draw_text(img, "CONTRAST", cx - 230, cy + 10, 0.6, Config.COLOR_GUIDE)
    else:
        draw_text(img, mode, cx - 50, cy - 60, 1.0, Config.COLOR_TEXT_ACTIVE)


# ================= IMAGE ENGINE =================
class ControlledEngine:
    def __init__(self):
        self.reset()

    def reset(self):
        self.vals = {
            "ZOOM": 1.0,
            "BRIGHTNESS": 0.0,
            "CONTRAST": 1.0,
            "SATURATION": 1.0
        }

    def update(self, mode, dx, dy):
        if mode == "ZOOM":
            step = np.clip(dx * Config.BASE_SPEED,
                           -Config.MAX_STEP_ZOOM, Config.MAX_STEP_ZOOM)
            self.vals["ZOOM"] = np.clip(self.vals["ZOOM"] + step, 1.0, 3.0)

        elif mode == "BRIGHTNESS":
            step = np.clip(-dy * Config.BASE_SPEED * 50,
                           -Config.MAX_STEP_BRIGHT, Config.MAX_STEP_BRIGHT)
            self.vals["BRIGHTNESS"] = np.clip(self.vals["BRIGHTNESS"] + step, -100, 100)

        elif mode == "CONTRAST":
            step = np.clip(-dx * Config.BASE_SPEED,
                           -Config.MAX_STEP_COLOR, Config.MAX_STEP_COLOR)
            self.vals["CONTRAST"] = np.clip(self.vals["CONTRAST"] + step, 0.5, 2.5)

        elif mode == "SATURATION":
            step = np.clip(dy * Config.BASE_SPEED,
                           -Config.MAX_STEP_COLOR, Config.MAX_STEP_COLOR)
            self.vals["SATURATION"] = np.clip(self.vals["SATURATION"] + step, 0.0, 3.0)

    def apply(self, img):
        v = self.vals

        if v["ZOOM"] > 1.01:
            h, w = img.shape[:2]
            nw, nh = int(w / v["ZOOM"]), int(h / v["ZOOM"])
            x1, y1 = (w - nw) // 2, (h - nh) // 2
            crop = img[y1:y1 + nh, x1:x1 + nw]
            if crop.size:
                img = cv2.resize(crop, (w, h))

        img = img.astype(np.float32)
        img += v["BRIGHTNESS"]

        mean = np.mean(img)
        img = (img - mean) * v["CONTRAST"] + mean

        img = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= v["SATURATION"]

        return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8),
                            cv2.COLOR_HSV2BGR)


# ================= MAIN =================
def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, Config.CAM_W)
    cap.set(4, Config.CAM_H)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

    engine = ControlledEngine()

    anchor = None
    active_mode = None

    photo_timer = 0
    message = ""
    msg_time = 0

    left_open = True
    left_clicks = 0
    left_time = 0

    # Right hand fist stability
    right_fist_locked = False
    right_fist_last_seen = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        clean = engine.apply(frame.copy())
        display = clean.copy()

        right_active = False

        if result.multi_hand_landmarks:
            for i, lm in enumerate(result.multi_hand_landmarks):
                hand = result.multi_handedness[i].classification[0].label

                fingers = []
                for t in [8, 12, 16, 20]:
                    fingers.append(1 if lm.landmark[t].y < lm.landmark[t - 2].y else 0)

                cx = int(lm.landmark[9].x * w)
                cy = int(lm.landmark[9].y * h)

                is_fist = sum(fingers) == 0
                is_peace = fingers == [1, 1, 0, 0]

                # ---------- PHOTO ----------
                if is_peace:
                    if photo_timer == 0:
                        photo_timer = time.time()
                    if time.time() - photo_timer >= 3:
                        cv2.imwrite(f"Studio_Photos_Final/{int(time.time())}.jpg", clean)
                        message = "PHOTO SAVED"
                        msg_time = time.time()
                        photo_timer = 0
                    else:
                        draw_text(display, "HOLD FOR PHOTO",
                                  cx - 90, cy - 60, 0.8, (255, 0, 255))
                else:
                    photo_timer = 0

                # ---------- LEFT HAND RESET ----------
                if hand == "Left":
                    if is_fist and left_open:
                        now = time.time()
                        left_clicks = left_clicks + 1 if now - left_time < 0.7 else 1
                        left_time = now
                        left_open = False

                        if left_clicks >= 2:
                            engine.reset()
                            message = "RESET"
                            msg_time = time.time()
                            left_clicks = 0
                    else:
                        left_open = True

                # ---------- RIGHT HAND CONTROL (STABLE FIST) ----------
                elif hand == "Right":
                    if is_fist:
                        right_fist_last_seen = time.time()
                        right_fist_locked = True

                    if right_fist_locked:
                        if time.time() - right_fist_last_seen < Config.FIST_RELEASE_TIME:
                            right_active = True
                        else:
                            right_fist_locked = False
                            anchor = None
                            active_mode = None
                    else:
                        draw_text(display, "MAKE FIST TO CONTROL",
                                  cx - 120, cy - 80, 0.8, Config.COLOR_ALERT)

                    if right_active:
                        if anchor is None:
                            anchor = (cx, cy)
                            active_mode = None

                        dx = cx - anchor[0]
                        dy = cy - anchor[1]

                        if active_mode is None:
                            draw_guides(display, anchor[0], anchor[1])
                            if abs(dx) > Config.DEADZONE or abs(dy) > Config.DEADZONE:
                                if abs(dx) > abs(dy):
                                    active_mode = "ZOOM" if dx > 0 else "CONTRAST"
                                else:
                                    active_mode = "SATURATION" if dy > 0 else "BRIGHTNESS"
                                anchor = (cx, cy)
                        else:
                            engine.update(active_mode, dx, dy)
                            draw_guides(display, cx, cy, active_mode)

        if not right_active:
            anchor = None
            active_mode = None

        draw_hud(display, engine.vals, active_mode)

        if time.time() - msg_time < 1.5:
            draw_text(display, message,
                      w // 2 - 100, h // 2, 2, (0, 255, 0), 4)

        cv2.imshow("Studio UX Pro", display)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
