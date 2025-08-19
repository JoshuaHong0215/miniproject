# -*- coding: utf-8 -*-
# 한글 주석, 영문 코드

import cv2, time, threading, numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Set, Optional

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO

# =========================
# YOLO & 카메라 설정
# =========================
MODEL_CANDIDATES = ['yolo11n.pt', 'yolov8n.pt']  # 존재하는 첫 가중치 로드
CONFIDENCE = 0.3
CAM_INDEX = 0
SHOW_FPS = True
TARGET_CLASS_IDS = [0, 16]  # person=0, dog=16 (예시)
CLASS_NAME = {0: 'person', 16: 'dog'}

# Tk 카메라 미리보기
CAM_VIEW_W = 520
CAM_VIEW_H = 300
CAM_VIEW_MODE = "fit"       # "fit" | "fill" | "stretch"
PAD_COLOR = (0, 0, 0)       # fit 모드 레터박스 색(BGR)

# =========================
# 도어/이동/개념도 설정
# =========================
OPENING_SEC = 4
HOLD_DEFAULT = 7
HOLD_SHORT = 4
CLOSING_SEC = 4

FLOORS_TOTAL = 15
FLOOR_TIME_SEC = 3.0        # 층간 이동 시간(초)
CONCEPT_W, CONCEPT_H = 260, 300
CAR_W, CAR_H = 18, 12
SHAFT_X_RATIO = 0.50        # 샤프트 X (0~1)
TOP_Y_RATIO = 0.10          # 최상층 Y 비율
BOT_Y_RATIO = 0.88          # 최하층 Y 비율

# 호출 방향
UP = "UP"
DOWN = "DOWN"
ARROW = {UP: "↑", DOWN: "↓"}

# =========================
# 공유 상태
# =========================
@dataclass
class Shared:
    lock: threading.Lock = field(default_factory=threading.Lock)
    person_present: bool = False
    person_count: int = 0
    kinds: Set[str] = field(default_factory=set)
    state: str = "IDLE"
    remain: float = 0.0
    mode_text: str = ""
    frame_bgr: Optional[np.ndarray] = None

shared = Shared()
stop_event = threading.Event()

# =========================
# 유틸(오버레이/모델)
# =========================
def load_model():
    last_err = None
    for w in MODEL_CANDIDATES:
        try:
            m = YOLO(w)
            print(f"[INFO] Loaded weights: {w}")
            return m
        except Exception as e:
            last_err = e
            print(f"[WARN] Failed to load {w}: {e}")
    raise RuntimeError(f"Failed to load any weights: {MODEL_CANDIDATES}\nLast: {last_err}")

def draw_state_overlay(frame, state, remain_sec, mode_text):
    x, y = 10, 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.6, 2
    cv2.putText(frame, f"STATE: {state}", (x, y), font, scale, (255,255,255), thick, cv2.LINE_AA)
    cv2.putText(frame, f"REMAIN: {max(0, int(remain_sec))}s", (x, y+22), font, scale, (255,255,255), thick, cv2.LINE_AA)
    if mode_text:
        cv2.putText(frame, mode_text, (x, y+44), font, scale, (255,255,255), thick, cv2.LINE_AA)

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
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad), (y + pad), (0, 0, 255), -1)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

# =========================
# 도어 상태머신
# =========================
class DoorController:
    IDLE="IDLE"; OPENING="OPENING"; OPEN_HOLD="OPEN_HOLD"; CLOSING="CLOSING"
    def __init__(self):
        self.state = self.IDLE
        self.phase_start = 0.0
        self.last_print_sec = 0
        self.seen_in_hold = False
        self.hold_target = HOLD_DEFAULT
        self.mode_text = ""

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
        if self.state == self.OPENING:
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
# Tk 앱
# =========================
class DoorSimApp:
    def __init__(self, root, door: DoorController, shared: Shared):
        self.root = root
        self.door = door
        self.shared = shared
        self.root.title("Elevator System (Door + Camera + Calls + Concept)")

        # ----- 치수/도어 캔버스 파라미터(먼저 정의) -----
        self.canvas_w = 520
        self.canvas_h = 300
        self.door_gap = 6
        door_scale = 0.6
        self.door_width_total = int(self.canvas_w * door_scale)
        self.door_margin = (self.canvas_w - self.door_width_total) // 2
        self.panel_half_width = (self.door_width_total - self.door_gap) // 2
        self.panel_height = 220
        self.panel_top = 40
        self.panel_bottom = self.panel_top + self.panel_height
        self.progress = 0.0

        # ----- 호출 상태 -----
        self.top_floor = FLOORS_TOTAL
        self.bottom_floor = 1
        self.pending_calls = deque()     # (floor, direction(or None), ts)
        self.call_timeout = 30           # 초

        # ----- 이동 상태 -----
        self.floors_total = FLOORS_TOTAL
        self.car_pos_f = 1.0             # 현재 위치(부동소수, 1층=1.0)
        self.car_target = None           # 목적층
        self._last_tick = time.time()

        # ----- UI 구성 -----
        self.loop_enabled = tk.BooleanVar(value=False)
        self._cam_imgtk = None

        self.build_ui()
        self.draw_static()
        self.draw_doors(0.0)
        self.update_status_text("IDLE", 0.0, "cycle: 4/7/4")

        # 개념도 초기화
        self._shaft_x  = int(CONCEPT_W * SHAFT_X_RATIO)
        self._top_y    = int(CONCEPT_H * TOP_Y_RATIO)
        self._bot_y    = int(CONCEPT_H * BOT_Y_RATIO)
        self._draw_concept_background()
        self._car = self.concept.create_rectangle(0,0,0,0, fill="#e62828", outline="#b81d1d", width=2)
        self._update_car_visual()

        # 루프 시작
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(int(1000/30), self.tick)

    # ---------- UI ----------
    def build_ui(self):
        top = tk.Frame(self.root); top.pack(padx=10, pady=10, fill="x")
        self.status_var = tk.StringVar(value="state: IDLE | remain: 0s | cycle: 4/7/4")
        tk.Label(top, textvariable=self.status_var, font=("Arial", 12)).pack(anchor="w")
        self.pbar = ttk.Progressbar(top, orient="horizontal", mode="determinate", maximum=100, value=0)
        self.pbar.pack(fill="x", pady=(6,6))

        ctrl = tk.Frame(top); ctrl.pack(fill="x")
        tk.Button(ctrl, text="Start", width=10, command=self.on_start).pack(side="left", padx=(0,8))
        tk.Checkbutton(ctrl, text="Loop", variable=self.loop_enabled).pack(side="left")
        tk.Button(ctrl, text="Quit", command=self.on_close).pack(side="right")

        # ===== 본문 =====
        body = tk.Frame(self.root); body.pack(padx=10, pady=(0,10), fill="both")

        # 좌: 도어 캔버스 (그대로)
        left = tk.Frame(body); left.pack(side="left", padx=(0,10))
        self.canvas = tk.Canvas(left, width=self.canvas_w, height=self.canvas_h, bg="#111")
        self.canvas.pack()

        # 중: 카메라 (위) + 호출/개념도 (아래 가로 정렬)
        mid = tk.Frame(body); mid.pack(side="left")

        # --- 카메라 프리뷰 (상단에 단독) ---
        cam_sec = tk.Frame(mid); cam_sec.pack(fill="both", expand=True)
        tk.Label(cam_sec, text="Camera Preview", font=("Arial", 11)).pack(anchor="w")
        self.cam_label = tk.Label(cam_sec, bg="black")
        self.cam_label.pack()

        # --- 하단 가로 레일: Floor Calls | Hall Calls | Concept View ---
        rail = tk.Frame(mid)
        rail.pack(pady=(10, 0), fill="x")

        # 1) Floor Calls (Direct)
        floor_panel = tk.LabelFrame(rail, text="Floor Calls (Direct)", padx=6, pady=6)
        floor_panel.grid(row=0, column=0, sticky="n")
        self.calls_var = tk.StringVar(value="pending: None")
        tk.Label(floor_panel, textvariable=self.calls_var, anchor="w").grid(
            row=0, column=0, columnspan=3, sticky="we", pady=(0,6)
        )
        floors = list(range(self.top_floor, self.bottom_floor - 1, -1))
        for idx, fl in enumerate(floors):
            r = idx // 3 + 1
            c = idx % 3
            tk.Button(floor_panel, text=f"{fl}F", width=8,
                    command=lambda f=fl: self.add_call(f)).grid(row=r, column=c, padx=3, pady=3)

        # 2) Hall Calls (Up/Down)
        hall = tk.LabelFrame(rail, text="Hall Calls (Up/Down)", padx=6, pady=6)
        hall.grid(row=0, column=1, padx=(8, 0), sticky="n")
        row = 0
        for fl in range(self.top_floor, self.bottom_floor - 1, -1):
            tk.Label(hall, text=f"{fl}F", width=4, anchor="e").grid(row=row, column=0, padx=(0,6), pady=2)
            up_btn = tk.Button(hall, text="UP", width=6, command=lambda f=fl: self.add_call_dir(f, UP))
            up_btn.grid(row=row, column=1, padx=2, pady=2)
            if fl == self.top_floor:
                up_btn.configure(state="disabled")
            dn_btn = tk.Button(hall, text="DOWN", width=6, command=lambda f=fl: self.add_call_dir(f, DOWN))
            dn_btn.grid(row=row, column=2, padx=2, pady=2)
            if fl == self.bottom_floor:
                dn_btn.configure(state="disabled")
            row += 1

        # 3) Concept View (우측 칸으로 이동)
        concept_box = tk.Frame(rail)
        concept_box.grid(row=0, column=2, padx=(8, 0), sticky="n")
        tk.Label(concept_box, text="Concept View", font=("Arial", 11)).pack(anchor="w")
        self.concept = tk.Canvas(concept_box, width=CONCEPT_W, height=CONCEPT_H, bg="#111")
        self.concept.pack()

        # grid 폭 균형: 필요 시 가중치 조정(지금은 고정폭들)
        rail.grid_columnconfigure(0, weight=0)
        rail.grid_columnconfigure(1, weight=0)
        rail.grid_columnconfigure(2, weight=0)

    # ---------- 이벤트 ----------
    def on_start(self):
        self.door.start_cycle(time.time())

    def on_close(self):
        stop_event.set()
        self.root.destroy()

    # ---------- 개념도/이동 ----------
    def _draw_concept_background(self):
        sx, ty, by = self._shaft_x, self._top_y, self._bot_y
        self.concept.create_line(sx, ty, sx, by, fill="#888", width=2)
        for f in range(self.floors_total, 0, -1):
            y = self._floor_to_y(float(f))
            self.concept.create_line(sx-10, y, sx+10, y, fill="#444")

    def _floor_to_y(self, floor_f: float) -> int:
        floor_f = max(1.0, min(self.floors_total * 1.0, floor_f))
        return int(self._bot_y + (self._top_y - self._bot_y) * (floor_f - 1.0) / (self.floors_total - 1.0))

    def _update_car_visual(self):
        cx = self._shaft_x
        cy = self._floor_to_y(self.car_pos_f)
        self.concept.coords(self._car, cx - CAR_W//2, cy - CAR_H//2, cx + CAR_W//2, cy + CAR_H//2)

    def start_move_to(self, target_floor: int):
        if target_floor is None: return
        target_floor = int(max(1, min(self.floors_total, target_floor)))
        if int(round(self.car_pos_f)) == target_floor:
            self.door.start_cycle(time.time())
            return
        self.car_target = target_floor

    # ---------- 호출 큐 ----------
    def add_call(self, floor: int):
        self.pending_calls.append((floor, None, time.time()))
        self.update_calls_label()

    def add_call_dir(self, floor: int, direction: str):
        if direction not in (UP, DOWN): return
        self.pending_calls.append((floor, direction, time.time()))
        self.update_calls_label()

    def cleanup_calls(self, timeout_sec: int):
        now = time.time()
        while self.pending_calls:
            item = self.pending_calls[0]
            ts = item[2] if len(item) >= 3 else item[1]
            if now - ts > timeout_sec:
                self.pending_calls.popleft()
            else:
                break

    def get_pending_preview(self, k: int = 8):
        out = []
        for item in list(self.pending_calls)[:k]:
            if len(item) >= 3:
                f, d, _ = item
            else:
                f, _ = item; d = None
            arrow = ARROW.get(d, "") if d else ""
            out.append(f"{f}{arrow}")
        return out

    def update_calls_label(self):
        preview = self.get_pending_preview()
        self.calls_var.set("pending: " + (", ".join(preview) if preview else "None"))

    def pop_next_call(self):
        return self.pending_calls.popleft() if self.pending_calls else None

    def has_pending_calls(self) -> bool:
        return len(self.pending_calls) > 0

    # ---------- 카메라 프리뷰 ----------
    def _make_cam_preview(self, frame_bgr: np.ndarray) -> ImageTk.PhotoImage:
        h, w = frame_bgr.shape[:2]
        tgt_w, tgt_h = CAM_VIEW_W, CAM_VIEW_H
        if CAM_VIEW_MODE == "stretch":
            out = cv2.resize(frame_bgr, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
        elif CAM_VIEW_MODE == "fill":
            scale = max(tgt_w / w, tgt_h / h)
            nw, nh = int(w*scale), int(h*scale)
            resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
            x0 = (nw - tgt_w) // 2; y0 = (nh - tgt_h) // 2
            out = resized[y0:y0+tgt_h, x0:x0+tgt_w]
        else:  # fit
            scale = min(tgt_w / w, tgt_h / h)
            nw, nh = int(w*scale), int(h*scale)
            resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
            canvas = np.full((tgt_h, tgt_w, 3), PAD_COLOR, dtype=np.uint8)
            x0 = (tgt_w - nw) // 2; y0 = (tgt_h - nh) // 2
            canvas[y0:y0+nh, x0:x0+nw] = resized
            out = canvas
        disp = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(disp))

    def update_cam_preview(self):
        frame_bgr = None
        with self.shared.lock:
            if self.shared.frame_bgr is not None:
                frame_bgr = self.shared.frame_bgr.copy()
        if frame_bgr is None: return
        imgtk = self._make_cam_preview(frame_bgr)
        self._cam_imgtk = imgtk
        self.cam_label.configure(image=imgtk)

    # ---------- 도어 애니메이션 ----------
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
        if state == DoorController.OPENING:
            elapsed = OPENING_SEC - remain
            p = min(max(elapsed / max(OPENING_SEC, 1e-6), 0.0), 1.0)
            self.draw_doors(p); self.pbar["value"] = int(p * 100)
        elif state == DoorController.OPEN_HOLD:
            self.draw_doors(1.0); self.pbar["value"] = 100
        elif state == DoorController.CLOSING:
            elapsed = CLOSING_SEC - remain
            p = max(1.0 - (elapsed / max(CLOSING_SEC, 1e-6)), 0.0)
            self.draw_doors(p); self.pbar["value"] = int(p * 100)
        else:
            self.draw_doors(0.0); self.pbar["value"] = 0

    def update_status_text(self, state: str, remain: float, extra: str):
        self.status_var.set(f"state: {state} | remain: {remain:.1f}s | {extra}")

    # ---------- 틱 루프 ----------
    def tick(self):
        now = time.time()

        # 호출 만료/표시 갱신
        self.cleanup_calls(self.call_timeout)
        self.update_calls_label()

        # 이동 애니메이션
        dt = now - self._last_tick
        self._last_tick = now
        moving = False
        move_remain = 0.0

        if self.car_target is not None:
            moving = True
            direction = 1.0 if self.car_target > self.car_pos_f else -1.0
            self.car_pos_f += direction * (dt / FLOOR_TIME_SEC)
            arrived = (direction > 0 and self.car_pos_f >= self.car_target) or \
                      (direction < 0 and self.car_pos_f <= self.car_target)
            if arrived:
                self.car_pos_f = float(self.car_target)
                self.car_target = None
                self.door.start_cycle(now)
            else:
                floors_left = abs(self.car_target - self.car_pos_f)
                move_remain = floors_left * FLOOR_TIME_SEC

        self._update_car_visual()

        # YOLO 감지 상태
        with self.shared.lock:
            person_present = self.shared.person_present

        # 도어 상태 진행
        state, remain, mode_text = self.door.update(now, person_present)

        # 이동 중인 경우 상태 대체
        if moving and self.car_target is not None:
            state = "MOVING"
            remain = move_remain
            mode_text = f"to {int(self.car_target)}F"

        # 공유 상태 갱신
        with self.shared.lock:
            self.shared.state = state
            self.shared.remain = remain
            self.shared.mode_text = mode_text

        # 애니메이션/상태 텍스트
        self.animate(state, remain)
        cycle_txt = "4/7/4" if (state != "OPEN_HOLD" or self.door.hold_target == HOLD_DEFAULT) else "4/4/4"
        self.update_status_text(state.replace("OPEN_HOLD", "DWELL"), remain, f"cycle: {cycle_txt} | {mode_text}")

        # 다음 호출 소비: 문이 닫혀 IDLE이고 이동 목표 없으면 꺼내 이동 시작
        if self.door.state == DoorController.IDLE and self.car_target is None and self.has_pending_calls():
            item = self.pop_next_call()
            target = item[0] if len(item) >= 2 else item
            self.start_move_to(int(target))

        # 카메라 미리보기 갱신
        self.update_cam_preview()

        if not stop_event.is_set():
            self.root.after(int(1000/30), self.tick)

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
    last_time = time.time()

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame."); break

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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(0, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
                if cls_id == 0:
                    person_present = True
                    person_count += 1
                if cls_id in CLASS_NAME:
                    kinds.add(name)

        # 상태 오버레이(문/남은시간/모드)
        with shared.lock:
            s_state = shared.state
            s_remain = shared.remain
            s_mode = shared.mode_text
        draw_state_overlay(frame, s_state.replace("OPEN_HOLD", "DWELL"), s_remain, s_mode)
        draw_people_count(frame, person_count)
        draw_caution_banner(frame, kinds)

        # 공유 상태 갱신 + 프레임 전달
        with shared.lock:
            shared.person_present = person_present
            shared.person_count = person_count
            shared.kinds = kinds.copy()
            shared.frame_bgr = frame

        # FPS 갱신(텍스트로만 계산)
        if SHOW_FPS:
            now = time.time()
            fps = 1.0 / (now - last_time) if (now - last_time) > 0 else 0.0
            last_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

    cap.release()

# =========================
# 엔트리 포인트
# =========================
def main():
    t = threading.Thread(target=camera_worker, args=(shared, stop_event), daemon=True)
    t.start()

    root = tk.Tk()
    app = DoorSimApp(root, DoorController(), shared)
    try:
        root.mainloop()
    finally:
        stop_event.set()
        t.join(timeout=2)

if __name__ == "__main__":
    main()
