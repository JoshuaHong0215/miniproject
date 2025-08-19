# -*- coding: utf-8 -*-
# 한글 주석, 영문 코드

import tkinter as tk
from tkinter import ttk
import time

# --------- 설정값(초) ----------
OPEN_T_DEFAULT  = 4
DWELL_T_DEFAULT = 7
CLOSE_T_DEFAULT = 4

DWELL_T_SHORT   = 4  # Short dwell 옵션일 때의 대기 시간
FPS              = 30  # 애니메이션 프레임레이트 목표

class DoorSimApp:
    # 엘리베이터 문 애니메이션: OPENING -> DWELL -> CLOSING
    def __init__(self, root):
        self.root = root
        self.root.title("Elevator Door 4/7/4 Simulation (Tkinter)")

        # 상태 변수
        self.state = "IDLE"          # IDLE, OPENING, DWELL, CLOSING
        self.state_start = 0.0       # 해당 상태 시작 시각
        self.loop_enabled = tk.BooleanVar(value=False)   # 사이클 반복 여부
        self.short_enabled = tk.BooleanVar(value=False)  # dwell 단축(4/4/4) 여부

        # 현재 사이클 타이밍(초)
        self.open_t  = OPEN_T_DEFAULT
        self.dwell_t = DWELL_T_DEFAULT
        self.close_t = CLOSE_T_DEFAULT

        # UI 레이아웃
        self.build_ui()

        # 캔버스에 문 그리기용 좌표/객체
        self.canvas_w = 520
        self.canvas_h = 300
        self.door_gap = 6  # 문 사이 중앙 간극

        # 문 전체 너비를 캔버스의 60%로 축소하고, 가운데 배치
        door_scale = 0.4                     # 0.5~0.8 사이로 조절
        self.door_width_total = int(self.canvas_w * door_scale)

        # 프레임을 문 너비에 맞춰 중앙 정렬되게 자동 계산
        self.door_margin = (self.canvas_w - self.door_width_total) // 2

        self.panel_half_width = (self.door_width_total - self.door_gap) // 2
        self.panel_height = 220
        self.panel_top = 40
        self.panel_bottom = self.panel_top + self.panel_height
        
        # "닫힘(=중앙)"이 p=0, "완전 개방"이 p=1
        self.progress = 0.0

        # 그리기 초기화
        self.draw_static()
        self.draw_doors(p=0.0)
        self.update_status_text("IDLE", remain=0.0)

        # 애니메이션 루프 시작
        interval = int(1000 / FPS)
        self.root.after(interval, self.tick)

    # ----------------- UI -----------------
    def build_ui(self):
        top = tk.Frame(self.root)
        top.pack(padx=10, pady=10, fill="x")

        # 상태/시간 표시
        self.status_var = tk.StringVar(value="state: IDLE | remain: 0.0s | cycle: 4/7/4")
        self.status_lbl = tk.Label(top, textvariable=self.status_var, font=("Arial", 12))
        self.status_lbl.pack(anchor="w")

        # 프로그레스바(상태의 진행 정도 감각용)
        self.pbar = ttk.Progressbar(top, orient="horizontal", mode="determinate")
        self.pbar.pack(fill="x", pady=(6, 6))
        self.pbar["maximum"] = 100
        self.pbar["value"] = 0

        # 버튼/옵션
        ctrl = tk.Frame(top)
        ctrl.pack(fill="x")

        self.start_btn = tk.Button(ctrl, text="Start", width=10, command=self.start_cycle)
        self.start_btn.pack(side="left", padx=(0,8))

        self.loop_chk = tk.Checkbutton(ctrl, text="Loop", variable=self.loop_enabled)
        self.loop_chk.pack(side="left", padx=(0,8))

        self.short_chk = tk.Checkbutton(ctrl, text="Short dwell (4/4/4)", variable=self.short_enabled, command=self.update_cycle_label)
        self.short_chk.pack(side="left")

        # 캔버스
        self.canvas = tk.Canvas(self.root, width=520, height=300, bg="#111")
        self.canvas.pack(padx=10, pady=(0,10))

    # ----------------- 그리기 -----------------
    def draw_static(self):
        # 프레임/바닥 등 고정 요소
        # 외곽 프레임
        self.canvas.create_rectangle(
            self.door_margin-6, self.panel_top-6,
            self.canvas_w - self.door_margin + 6, self.panel_bottom + 6,
            outline="#888", width=2
        )
        # 상단 안내 텍스트
        self.canvas.create_text(
            self.canvas_w//2, 18,
            text="Elevator Doors", fill="#ddd", font=("Arial", 12, "bold")
        )

    def draw_doors(self, p: float):
        # p: 0.0(완전 닫힘) ~ 1.0(완전 열림)
        # 좌/우 문 두 장을 중앙으로부터 좌우로 이동
        # 중앙 기준
        cx = self.canvas_w // 2
        half_gap = self.door_gap // 2

        # 왼쪽 패널의 오른쪽 가장자리 x 좌표(중앙에서 p만큼 이동)
        left_panel_right = cx - half_gap - int(p * (self.panel_half_width))
        left_panel_left  = left_panel_right - self.panel_half_width

        # 오른쪽 패널의 왼쪽 가장자리 x 좌표
        right_panel_left  = cx + half_gap + int(p * (self.panel_half_width))
        right_panel_right = right_panel_left + self.panel_half_width

        # 기존 문 지우고 다시 그림
        self.canvas.delete("door")
        # 왼쪽 문
        self.canvas.create_rectangle(
            left_panel_left, self.panel_top,
            left_panel_right, self.panel_bottom,
            fill="#1e90ff", outline="#0c5ba8", width=2, tags="door"
        )
        # 오른쪽 문
        self.canvas.create_rectangle(
            right_panel_left, self.panel_top,
            right_panel_right, self.panel_bottom,
            fill="#1e90ff", outline="#0c5ba8", width=2, tags="door"
        )

    # ----------------- 상태/싸이클 -----------------
    def start_cycle(self):
        # Short dwell 여부 반영
        if self.short_enabled.get():
            self.open_t  = OPEN_T_DEFAULT
            self.dwell_t = DWELL_T_SHORT
            self.close_t = CLOSE_T_DEFAULT
        else:
            self.open_t  = OPEN_T_DEFAULT
            self.dwell_t = DWELL_T_DEFAULT
            self.close_t = CLOSE_T_DEFAULT

        # OPENING 상태로 진입
        self.set_state("OPENING")
        self.progress = 0.0
        self.update_cycle_label()

    def set_state(self, s: str):
        self.state = s
        self.state_start = time.time()

    def state_elapsed(self) -> float:
        return time.time() - self.state_start

    def update_cycle_label(self):
        # 상단 상태줄의 cycle 부분만 최신화(체크박스 변화 반영)
        cycle_txt = f"{self.open_t}/{self.dwell_t}/{self.close_t}"
        # 나머지는 tick에서 덮어쓰므로 여기서는 cycle만 반영
        cur = self.status_var.get()
        parts = cur.split("|")
        if len(parts) >= 3:
            parts[2] = f" cycle: {cycle_txt}"
            self.status_var.set("|".join(parts).strip())
        else:
            self.status_var.set(f"state: {self.state} | remain: 0.0s | cycle: {cycle_txt}")

    # ----------------- 메인 루프 -----------------
    def tick(self):
        interval = int(1000 / FPS)
        self.animate()
        self.root.after(interval, self.tick)

    def animate(self):
        # 상태별로 p(문 열림 정도)와 남은 시간, 프로그레스바를 갱신
        if self.state == "IDLE":
            # 대기: 닫힌 상태 유지
            self.progress = 0.0
            self.draw_doors(self.progress)
            self.update_status_text("IDLE", remain=0.0)
            self.pbar["value"] = 0
            return

        if self.state == "OPENING":
            t = self.state_elapsed()
            dur = max(self.open_t, 1e-6)
            # 0 -> 1 로 선형 증가
            p = min(t / dur, 1.0)
            self.progress = p
            self.draw_doors(self.progress)
            remain = max(self.open_t - t, 0.0)
            self.update_status_text("OPENING", remain)
            self.pbar["value"] = int(p * 100)
            if t >= self.open_t:
                self.set_state("DWELL")

        elif self.state == "DWELL":
            t = self.state_elapsed()
            # 열린 상태 유지
            self.progress = 1.0
            self.draw_doors(self.progress)
            remain = max(self.dwell_t - t, 0.0)
            self.update_status_text("DWELL", remain)
            self.pbar["value"] = 100
            if t >= self.dwell_t:
                self.set_state("CLOSING")

        elif self.state == "CLOSING":
            t = self.state_elapsed()
            dur = max(self.close_t, 1e-6)
            # 1 -> 0 로 선형 감소
            p = max(1.0 - (t / dur), 0.0)
            self.progress = p
            self.draw_doors(self.progress)
            remain = max(self.close_t - t, 0.0)
            self.update_status_text("CLOSING", remain)
            self.pbar["value"] = int(p * 100)
            if t >= self.close_t:
                # 사이클 완료
                if self.loop_enabled.get():
                    # 다시 OPENING으로
                    self.start_cycle()
                else:
                    self.set_state("IDLE")

    def update_status_text(self, state: str, remain: float):
        cycle_txt = f"{self.open_t}/{self.dwell_t}/{self.close_t}"
        self.status_var.set(f"state: {state} | remain: {remain:.1f}s | cycle: {cycle_txt}")

def main():
    root = tk.Tk()
    app = DoorSimApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
