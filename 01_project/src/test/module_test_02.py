# YOLO + DoorController (기본 4/7/4, '사람 전혀 없음'이면 4/4/4)
# - OPENING: 4초
# - OPEN_HOLD: 7초(사람을 4초 안에 한 번이라도 보면), 아니면 4초
# - CLOSING: 4초
# - 블로킹 없음(time.sleep 미사용)

import cv2
import time
from ultralytics import YOLO

# =========================
# 사용자 설정
# =========================
MODEL_CANDIDATES = ['yolo11n.pt', 'yolov8n.pt']
CONFIDENCE = 0.3
SHOW_FPS = True
CAM_INDEX = 0
TARGET_CLASS_IDS = [0, 16]         # person=0, dog=16 (표시만)
CLASS_NAME = {0: 'person', 16: 'dog'}

# 타이밍(초)
OPENING_SEC = 4
HOLD_DEFAULT = 7                   # 기본 열린유지 목표
HOLD_SHORT = 4                     # '사람 전혀 없음'이면 4초에 닫기
CLOSING_SEC = 4

START_ON_BOOT = True               # 실행 즉시 한 사이클 시작(테스트 편의용)

# =========================
# 유틸
# =========================
def load_model():
    last_err = None
    for w in MODEL_CANDIDATES:
        try:
            model = YOLO(w)
            print(f"[INFO] Loaded weights: {w}")
            return model
        except Exception as e:
            last_err = e
            print(f"[WARN] Failed to load {w}: {e}")
    raise RuntimeError(f"Failed to load any weights: {MODEL_CANDIDATES}\nLast: {last_err}")

def draw_caution_banner(frame, kinds):
    h, w = frame.shape[:2]
    label = " / ".join(sorted(kinds)) if kinds else "TARGET"
    text = f"CAUTION: {label.upper()}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = (w - tw) // 2
    y = 40
    pad = 8
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + pad), (0, 0, 255), -1)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

def draw_state_overlay(frame, state, remain_sec, mode_text):
    x, y = 10, 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.6, 2
    cv2.putText(frame, f"STATE: {state}", (x, y), font, scale, (255,255,255), thick, cv2.LINE_AA)
    cv2.putText(frame, f"REMAIN: {max(0, int(remain_sec))}s", (x, y+22), font, scale, (255,255,255), thick, cv2.LINE_AA)
    if mode_text:
        cv2.putText(frame, mode_text, (x, y+44), font, scale, (255,255,255), thick, cv2.LINE_AA)

def draw_people_count(frame, count):
    # 목적: 우상단에 사람 수 표시
    h, w = frame.shape[:2]
    text = f"PEOPLE: {count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.8, 2
    # 텍스트 크기 계산 후, 오른쪽 여백 10px, 위쪽 여백 10px
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = w - tw - 10
    y = 10 + th
    # 반투명 배경 박스
    bg_pad = 6
    cv2.rectangle(frame, (x - bg_pad, y - th - bg_pad),
                         (x + tw + bg_pad, y + bg_pad),
                         (0, 0, 0), -1)
    # 텍스트
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

# =========================
# 비블로킹 도어 상태머신
# =========================
class DoorController:
    IDLE="IDLE"; OPENING="OPENING"; OPEN_HOLD="OPEN_HOLD"; CLOSING="CLOSING"

    def __init__(self):
        self.state = self.IDLE
        self.phase_start = 0.0
        self.last_print_sec = 0

        # OPEN_HOLD 판단용
        self.seen_in_hold = False     # OPEN_HOLD 내 '사람을 한 번이라도 봤는가'
        self.mode_text = ""           # 4/7/4 또는 4/4/4 모드 표기용

    def start_cycle(self, now: float):
        self.state = self.OPENING
        self.phase_start = now
        self.last_print_sec = 0
        print("문이 열립니다")

    def _print_sec(self, elapsed: float, duration_cap: int):
        # 초당 1,2,... 출력(최대 duration_cap까지만)
        sec_to_print = min(int(elapsed) + 1, duration_cap)
        if sec_to_print != self.last_print_sec:
            print(f"{sec_to_print} sec")
            self.last_print_sec = sec_to_print

    def update(self, now: float, person_present: bool):
        remain = 0.0
        self.mode_text = ""

        if self.state == self.IDLE:
            if START_ON_BOOT and not getattr(self, "_boot_started", False):
                self._boot_started = True
                self.start_cycle(now)

        elif self.state == self.OPENING:
            elapsed = now - self.phase_start
            remain = OPENING_SEC - elapsed
            self._print_sec(elapsed, OPENING_SEC)

            if elapsed >= OPENING_SEC:
                self.state = self.OPEN_HOLD
                self.phase_start = now
                self.last_print_sec = 0
                self.seen_in_hold = person_present  # 진입 순간 감지되면 True
                print("문이 열린 상태 유지")

        elif self.state == self.OPEN_HOLD:
            elapsed = now - self.phase_start

            # 4초 안에 한 번이라도 사람을 보면 7초 유지 확정
            if person_present:
                self.seen_in_hold = True

            # 모드/남은시간 계산
            if self.seen_in_hold:
                # 7초 모드(4/7/4)
                duration_target = HOLD_DEFAULT
                self.mode_text = "MODE: 4/7/4"
            else:
                # 4초 모드(4/4/4)
                duration_target = HOLD_SHORT
                self.mode_text = "MODE: 4/4/4 (no person)"

            remain = duration_target - elapsed
            self._print_sec(elapsed, duration_target)

            if elapsed >= duration_target:
                self.state = self.CLOSING
                self.phase_start = now
                self.last_print_sec = 0
                print("문이 닫힙니다")

        elif self.state == self.CLOSING:
            elapsed = now - self.phase_start
            remain = CLOSING_SEC - elapsed
            self._print_sec(elapsed, CLOSING_SEC)

            if elapsed >= CLOSING_SEC:
                self.state = self.IDLE
                self.phase_start = now
                self.last_print_sec = 0
                print("문이 닫혔습니다")

        return self.state, max(0.0, remain), self.mode_text

# =========================
# 메인 루프
# =========================
def main():
    # 카메라
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    # 모델
    model = load_model()

    # 도어 컨트롤러
    door = DoorController()
    last_time = time.time()

    print("[INFO] Press 'q' to quit.")
    window = "YOLO + DoorController (4/7/4 vs 4/4/4)"
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame.")
            break

        # 감지
        results = model(frame, conf=CONFIDENCE, classes=TARGET_CLASS_IDS, verbose=False)[0]
        person_present = False
        person_count = 0
        kinds = set()

        if results.boxes is not None and len(results.boxes) > 0:
            # 안전한 zip 순회
            for xyxy, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                x1, y1, x2, y2 = xyxy.int().tolist()
                conf = float(conf)
                cls_id = int(cls)
                name = CLASS_NAME.get(cls_id, str(cls_id))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

                if cls_id == 0:
                    person_present = True
                    person_count += 1

                if cls_id in CLASS_NAME:
                    kinds.add(name)

        if kinds:
            draw_caution_banner(frame, kinds)

        # 상태 업데이트 + 오버레이
        now = time.time()
        state, remain, mode_text = door.update(now, person_present)
        draw_state_overlay(frame, state, remain, mode_text)

        # FPS
        if SHOW_FPS:
            fps = 1.0 / (now - last_time) if (now - last_time) > 0 else 0.0
            last_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
            
        if kinds:
            draw_caution_banner(frame, kinds)

        # 상태 업데이트 + 상태/잔여시간 오버레이
        now = time.time()
        state, remain, mode_text = door.update(now, person_present)
        draw_state_overlay(frame, state, remain, mode_text)

        # ★ 사람 수 표시
        draw_people_count(frame, person_count) 

        # 출력/입력
        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
