# -*- coding: utf-8 -*-
# 한글 주석, 영문 코드

import cv2
import time
import threading
from dataclasses import dataclass, field
from typing import Set
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk

# =========================
# 사용자 설정 (module_test_02.py 기반)
# =========================
MODEL_CANDIDATES = ['yolo11n.pt', 'yolov8n.pt']
CONFIDENCE = 0.3
CAM_INDEX = 0
SHOW_FPS = True

# 관심 클래스: 사람(0), 개(16) — 필요 시 수정
TARGET_CLASS_IDS = [0, 16]
CLASS_NAME = {0: 'person', 16: 'dog'}

# 타이밍(초): 4/7/4 기본, '사람 전혀 없음'이면 4/4/4
OPENING_SEC = 4
HOLD_DEFAULT = 7
HOLD_SHORT = 4
CLOSING_SEC = 4
START_ON_BOOT = False  # Tk 쪽 [Start] 버튼으로 시작하게 함

# =========================
# 공유 상태
# =========================
@dataclass
class Shared:
    lock: threading.Lock = field(default_factory=threading.Lock)
    # 카메라/YOLO → 컨트롤러
    person_present: bool = False
    # 컨트롤러 → 카메라/Tk overlay
    state: str = "IDLE"
    remain: float = 0.0
    mode_text: str = ""  # "MODE: 4/7/4" or "MODE: 4/4/4 ..."
    person_count: int = 0
    kinds: Set[str] = field(default_factory=set)

shared = Shared()
stop_event = threading.Event()

# =========================
# 유틸 (module_test_02.py에서 사용)
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
    if not kinds:
        return
    h, w = frame.shape[:2]
    label = " / ".join(sorted(kinds))
    text = f"CAUTION: {label.upper()}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 1.0, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = (w - tw) // 2
    y = 40
    pad = 8
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + pad), (0, 0, 255), -1)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

def draw_people_count(frame, count):
    h, w = frame.shape[:2]
    text = f"PEOPLE: {count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.8, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = w - tw - 10
    y = 10 + th
    pad = 6
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + pad), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

def draw_state_overlay(frame, state, remain_sec, mode_text):
    x, y = 10, 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.6, 2
    cv2.putText(frame, f"STATE: {state}", (x, y), font, scale, (255,255,255), thick, cv2.LINE_AA)
    cv2.putText(frame, f"REMAIN: {max(0, int(remain_sec))}s", (x, y+22), font, scale, (255,255,255), thick, cv2.LINE_AA)
    if mode_text:
        cv2.putText(frame, mode_text, (x, y+44), font, scale, (255,255,255), thick, cv2.LINE_AA)

# =========================
# 도어 상태머신 (module_test_02.py 기반)
# =========================
class DoorController:
    IDLE="IDLE"; OPENING="OPENING"; OPEN_HOLD="OPEN_HOLD"; CLOSING="CLOSING"
    def __init__(self):
        self.state = self.IDLE
        self.phase_start = 0.0
        self.last_print_sec = 0
        self.seen_in_hold = False
        self.mode_text = ""
        self.hold_target = HOLD_DEFAULT  # 현재 홀드 목표(7 또는 4)

    def start_cycle(self, now: float):
        self.state = self.OPENING
        self.phase_start = now
        self.last_print_sec = 0
        print("문이 열립니다")

    def _print_sec(self, elapsed: float, duration_cap: int):
        sec_to_print = min(int(elapsed) + 1, duration_cap)
        if sec_to_print != self.last_print_sec:
            print(f"{sec_to_print} sec")
            self.last_print_sec = sec_to_print

    def update(self, now: float, person_present: bool):
        remain = 0.0
        self.mode_text = ""

        if self.state == self.IDLE:
            # START_ON_BOOT 시 자동 시작(이번 통합에서는 False)
            pass

        elif self.state == self.OPENING:
            elapsed = now - self.phase_start
            remain = OPENING_SEC - elapsed
            self._print_sec(elapsed, OPENING_SEC)
            if elapsed >= OPENING_SEC:
                self.state = self.OPEN_HOLD
                self.phase_start = now
                self.last_print_sec = 0
                self.seen_in_hold = person_present
                print("문이 열린 상태 유지")

        elif self.state == self.OPEN_HOLD:
            elapsed = now - self.phase_start
            if person_present:
                self.seen_in_hold = True

            if self.seen_in_hold:
                self.hold_target = HOLD_DEFAULT
                self.mode_text = "MODE: 4/7/4"
            else:
                self.hold_target = HOLD_SHORT
                self.mode_text = "MODE: 4/4/4 (no person)"

            remain = self.hold_target - elapsed
            self._print_sec(elapsed, self.hold_target)

            if elapsed >= self.hold_target:
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
# Tkinter 문 애니메이션 (module_test_03.py 기반, 컨트롤러 연동)
# =========================
class DoorSimApp:
    def __init__(self, root, door: DoorController, shared: Shared):
        self.root = root
        self.door = door
        self.shared = shared
        self.root.title("Elevator Door 4/7/4 Simulation (Tkinter)")



        # 캔버스 배치/크기
        self.canvas_w = 520
        self.canvas_h = 300
        self.door_gap = 6

        # 문 전체 너비: 캔버스의 60%로 축소 중앙 배치
        door_scale = 0.6
        self.door_width_total = int(self.canvas_w * door_scale)
        self.door_margin = (self.canvas_w - self.door_width_total) // 2

        self.panel_half_width = (self.door_width_total - self.door_gap) // 2
        self.panel_height = 220
        self.panel_top = 40
        self.panel_bottom = self.panel_top + self.panel_height

        # 0=닫힘, 1=완전개방
        self.progress = 0.0
       
        # 반복 실행 옵션
        self.loop_enabled = tk.BooleanVar(value=False)
       
        # UI
        self.build_ui()


        # 초기 그리기
        self.draw_static()
        self.draw_doors(0.0)
        self.update_status_text("IDLE", 0.0, "cycle: 4/7/4")

        # 틱 루프
        self.root.after(int(1000/30), self.tick)

    def build_ui(self):
        top = tk.Frame(self.root)
        top.pack(padx=10, pady=10, fill="x")

        self.status_var = tk.StringVar(value="state: IDLE | remain: 0s | cycle: 4/7/4")
        tk.Label(top, textvariable=self.status_var, font=("Arial", 12)).pack(anchor="w")

        self.pbar = ttk.Progressbar(top, orient="horizontal", mode="determinate", maximum=100, value=0)
        self.pbar.pack(fill="x", pady=(6,6))

        ctrl = tk.Frame(top); ctrl.pack(fill="x")
        tk.Button(ctrl, text="Start", width=10, command=self.on_start).pack(side="left", padx=(0,8))
        tk.Checkbutton(ctrl, text="Loop", variable=self.loop_enabled).pack(side="left")

        self.canvas = tk.Canvas(self.root, width=self.canvas_w, height=self.canvas_h, bg="#111")
        self.canvas.pack(padx=10, pady=(0,10))

    def on_start(self):
        self.door.start_cycle(time.time())

    def tick(self):
        now = time.time()

        # YOLO에서 온 사람 감지 플래그 읽기
        with self.shared.lock:
            person_present = self.shared.person_present

        # 컨트롤러 진행
        state, remain, mode_text = self.door.update(now, person_present)

        # 상태를 공유(카메라 오버레이에서 사용)
        with self.shared.lock:
            self.shared.state = state
            self.shared.remain = remain
            self.shared.mode_text = mode_text

        # 애니메이션 갱신
        self.animate(state, remain)

        # 상태 텍스트
        cycle_txt = "4/7/4" if (state != "OPEN_HOLD" or self.door.hold_target == HOLD_DEFAULT) else "4/4/4"
        self.update_status_text(state.replace("OPEN_HOLD", "DWELL"), remain, f"cycle: {cycle_txt} | {mode_text}")

        # 루프 옵션: IDLE이면 자동 재시작
        if state == DoorController.IDLE and self.loop_enabled.get():
            self.door.start_cycle(now)

        if not stop_event.is_set():
            self.root.after(int(1000/30), self.tick)
        else:
            try:
                self.root.destroy()
            except:
                pass

    def draw_static(self):
        self.canvas.create_rectangle(
            self.door_margin-6, self.panel_top-6,
            self.canvas_w - self.door_margin + 6, self.panel_bottom + 6,
            outline="#888", width=2
        )
        self.canvas.create_text(self.canvas_w//2, 18, text="Elevator Doors",
                                fill="#ddd", font=("Arial", 12, "bold"))

    def draw_doors(self, p: float):
        cx = self.canvas_w // 2
        half_gap = self.door_gap // 2

        left_panel_right = cx - half_gap - int(p * (self.panel_half_width))
        left_panel_left  = left_panel_right - self.panel_half_width

        right_panel_left  = cx + half_gap + int(p * (self.panel_half_width))
        right_panel_right = right_panel_left + self.panel_half_width

        self.canvas.delete("door")
        self.canvas.create_rectangle(
            left_panel_left, self.panel_top, left_panel_right, self.panel_bottom,
            fill="#1e90ff", outline="#0c5ba8", width=2, tags="door"
        )
        self.canvas.create_rectangle(
            right_panel_left, self.panel_top, right_panel_right, self.panel_bottom,
            fill="#1e90ff", outline="#0c5ba8", width=2, tags="door"
        )

    def animate(self, state: str, remain: float):
        # 상태별 진행도 p 계산
        if state == DoorController.OPENING:
            elapsed = OPENING_SEC - remain
            p = min(max(elapsed / max(OPENING_SEC, 1e-6), 0.0), 1.0)
            self.progress = p
            self.draw_doors(self.progress)
            self.pbar["value"] = int(p * 100)

        elif state == DoorController.OPEN_HOLD:
            self.progress = 1.0
            self.draw_doors(self.progress)
            self.pbar["value"] = 100

        elif state == DoorController.CLOSING:
            elapsed = CLOSING_SEC - remain
            p = max(1.0 - (elapsed / max(CLOSING_SEC, 1e-6)), 0.0)
            self.progress = p
            self.draw_doors(self.progress)
            self.pbar["value"] = int(p * 100)

        else:  # IDLE
            self.progress = 0.0
            self.draw_doors(self.progress)
            self.pbar["value"] = 0

    def update_status_text(self, state: str, remain: float, extra: str):
        self.status_var.set(f"state: {state} | remain: {remain:.1f}s | {extra}")

# =========================
# 카메라 + YOLO 스레드
# =========================
def camera_worker(shared: Shared, stop_event: threading.Event):
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        stop_event.set()
        return

    model = load_model()
    window = "YOLO + DoorController (Linked)"
    print("[INFO] Press 'q' to quit (OpenCV window).")

    last_time = time.time()
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame.")
            break

        # YOLO 추론
        results = model(frame, conf=CONFIDENCE, classes=TARGET_CLASS_IDS, verbose=False)[0]
        person_present = False
        person_count = 0
        kinds = set()

        if results.boxes is not None and len(results.boxes) > 0:
            for xyxy, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                x1, y1, x2, y2 = xyxy.int().tolist()
                conf = float(conf)
                cls_id = int(cls)
                name = CLASS_NAME.get(cls_id, str(cls_id))

                # 박스/라벨
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(0, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

                if cls_id == 0:
                    person_present = True
                    person_count += 1
                if cls_id in CLASS_NAME:
                    kinds.add(name)

        # 공유 플래그 갱신
        with shared.lock:
            shared.person_present = person_present
            shared.person_count = person_count
            shared.kinds = kinds.copy()
            s_state = shared.state
            s_remain = shared.remain
            s_mode = shared.mode_text

        # 오버레이
        draw_state_overlay(frame, s_state.replace("OPEN_HOLD", "DWELL"), s_remain, s_mode)
        draw_people_count(frame, person_count)
        draw_caution_banner(frame, kinds)

        # FPS
        if SHOW_FPS:
            now = time.time()
            fps = 1.0 / (now - last_time) if (now - last_time) > 0 else 0.0
            last_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

        # 표시/키입력
        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()

# =========================
# 엔트리 포인트
# =========================
def main():
    # 카메라/YOLO 스레드 시작
    t = threading.Thread(target=camera_worker, args=(shared, stop_event), daemon=True)
    t.start()

    # Tkinter 메인 루프
    root = tk.Tk()
    app = DoorSimApp(root, DoorController(), shared)
    try:
        root.mainloop()
    finally:
        stop_event.set()
        t.join(timeout=2)

if __name__ == "__main__":
    main()
