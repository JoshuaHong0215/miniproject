# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import messagebox
import random

class ElevatorApp:
    def __init__(self, root):
        # 기본 설정 --------------------------------------------------------
        self.root = root
        self.root.title("엘리베이터 ETA 시뮬레이터 (15층)")

        # 상수 ------------------------------------------------------------
        self.TOTAL_FLOORS = 15            # 총 층 수
        self.HEIGHT_PER_FLOOR_M = 3       # 층당 높이(m)
        self.SPEED_M_PER_SEC = 1          # 속도(m/s) = 60 m/min
        self.TIME_PER_FLOOR_SEC = int(self.HEIGHT_PER_FLOOR_M / self.SPEED_M_PER_SEC)  # 층당 3초

        # 상태값 ----------------------------------------------------------
        self.elevator_floor = random.randint(1, self.TOTAL_FLOORS)  # 엘리베이터 랜덤 배치
        self.moving = False                 # 이동 중 여부
        self.target_floor = None            # 목표(사용자) 층
        self.remaining_floors = 0           # 남은 이동 층 수
        self.direction = 0                  # 이동 방향 (+1 또는 -1)
        self.per_floor_remaining = 0        # 현재 층 이동에 남은 초(3 → 0)
        self.total_remaining_seconds = 0    # 전체 남은 시간(초)
        self._tick_job = None               # after 스케줄러 핸들(전체 카운트다운)
        self._per_floor_job = None          # after 스케줄러 핸들(층별 카운트다운)

        # UI 구성 ---------------------------------------------------------
        self.status_var = tk.StringVar()
        self._update_status_text()
        status_label = tk.Label(self.root, textvariable=self.status_var, font=("Malgun Gothic", 12))
        status_label.grid(row=0, column=0, columnspan=5, padx=10, pady=(10, 6), sticky="w")

        info = (
            f"- 총 {self.TOTAL_FLOORS}층, 층당 {self.HEIGHT_PER_FLOOR_M} m\n"
            f"- 속도: {self.SPEED_M_PER_SEC} m/s (층당 {self.TIME_PER_FLOOR_SEC}초)\n"
            f"- 아래에서 현재(대기) 층을 클릭하세요."
        )
        info_label = tk.Label(self.root, text=info, justify="left", font=("Malgun Gothic", 10))
        info_label.grid(row=1, column=0, columnspan=5, padx=10, pady=(0, 10), sticky="w")

        self.result_var = tk.StringVar(value="예상 도착시간: -")
        result_label = tk.Label(self.root, textvariable=self.result_var, font=("Malgun Gothic", 12, "bold"))
        result_label.grid(row=2, column=0, columnspan=5, padx=10, pady=(0, 6), sticky="w")

        self.per_floor_var = tk.StringVar(value="층 이동 카운트다운: -")
        perfloor_label = tk.Label(self.root, textvariable=self.per_floor_var, font=("Malgun Gothic", 11))
        perfloor_label.grid(row=3, column=0, columnspan=5, padx=10, pady=(0, 10), sticky="w")

        self.arrival_var = tk.StringVar(value="")
        self.arrival_label = tk.Label(self.root, textvariable=self.arrival_var,
                                      font=("Malgun Gothic", 14, "bold"), fg="green")
        self.arrival_label.grid(row=4, column=0, columnspan=5, padx=10, pady=(0, 10), sticky="w")

        # 층 버튼(15층) ---------------------------------------------------
        btn_frame = tk.Frame(self.root)
        btn_frame.grid(row=5, column=0, columnspan=5, padx=10, pady=(0, 10))
        floors = list(range(self.TOTAL_FLOORS, 1-1, -1))  # [15..1]
        self.floor_buttons = []
        for idx, floor in enumerate(floors):
            r = idx // 5
            c = idx % 5
            btn = tk.Button(
                btn_frame,
                text=f"{floor}층",
                width=8,
                command=lambda f=floor: self.on_floor_click(f)
            )
            btn.grid(row=r, column=c, padx=4, pady=4)
            self.floor_buttons.append(btn)

        # 재배치 버튼 -----------------------------------------------------
        reset_btn = tk.Button(self.root, text="엘리베이터 재배치(랜덤)", command=self.reset_elevator)
        reset_btn.grid(row=6, column=0, columnspan=5, padx=10, pady=(0, 12), sticky="we")

    # 상태 라벨 갱신 ------------------------------------------------------
    def _update_status_text(self):
        self.status_var.set(f"현재 엘리베이터 위치: {self.elevator_floor}층")

    # 버튼 클릭(사용자 층 선택) -------------------------------------------
    def on_floor_click(self, user_floor):
        # 이동 중이면 무시 (중복 명령 방지)
        if self.moving:
            return

        # 도착 라벨 초기화
        self.arrival_var.set("")
        self.arrival_label.config(fg="green")

        # 목표/방향/남은층 계산
        self.target_floor = user_floor
        self.remaining_floors = abs(self.elevator_floor - self.target_floor)
        if self.remaining_floors == 0:
            # 이미 같은 층인 경우 즉시 도착 처리
            self.result_var.set("예상 도착시간: 0초 (이미 도착)")
            self._on_arrival()
            return

        self.direction = 1 if self.target_floor > self.elevator_floor else -1
        self.total_remaining_seconds = self.remaining_floors * self.TIME_PER_FLOOR_SEC
        self.result_var.set(f"예상 도착시간: {self.total_remaining_seconds}초")

        # 이동 시작 상태 세팅
        self.moving = True
        self._set_buttons_state(tk.DISABLED)

        # 전체 타이머 시작(초 단위 감소)
        self._tick_total_countdown()

        # 첫 층 이동 시작(3초 카운트다운)
        self._start_move_one_floor()

    # 전체 남은 시간 카운트다운(초 단위) -----------------------------------
    def _tick_total_countdown(self):
        # 이동 종료 시 스케줄 중단
        if not self.moving:
            return
        # 표기 갱신
        self.result_var.set(f"예상 도착시간: {self.total_remaining_seconds}초")
        # 다음 1초 예약
        if self.total_remaining_seconds > 0:
            self.total_remaining_seconds -= 1
            self._tick_job = self.root.after(1000, self._tick_total_countdown)

    # 한 층(3초) 이동 시작 -------------------------------------------------
    def _start_move_one_floor(self):
        # 현재 층에서 다음 층으로 가기 위한 3초 카운트다운 시작
        self.per_floor_remaining = self.TIME_PER_FLOOR_SEC
        self._per_floor_tick()

    # 한 층 이동 카운트다운 틱 --------------------------------------------
    def _per_floor_tick(self):
        if not self.moving:
            return

        # 남은 초 표기
        self.per_floor_var.set(f"층 이동 카운트다운: {self.per_floor_remaining}초")
        if self.per_floor_remaining > 0:
            self.per_floor_remaining -= 1
            self._per_floor_job = self.root.after(1000, self._per_floor_tick)
        else:
            # 3초가 끝나면 실제로 한 층 이동
            self.elevator_floor += self.direction
            self._update_status_text()

            # 남은 층 수 감소
            self.remaining_floors -= 1

            # 다음 층이 남았으면 또 3초 카운트다운
            if self.remaining_floors > 0:
                self._start_move_one_floor()
            else:
                # 도착 처리
                self._on_arrival()

    # 도착 처리 -----------------------------------------------------------
    def _on_arrival(self):
        # 이동 플래그 해제 및 버튼 복구
        self.moving = False
        self._set_buttons_state(tk.NORMAL)

        # 카운트다운 라벨 정리
        self.per_floor_var.set("층 이동 카운트다운: 완료")

        # 도착 라벨/팝업
        self.arrival_var.set(f"도착했습니다! (현재 층: {self.elevator_floor}층)")
        try:
            messagebox.showinfo("도착", f"{self.elevator_floor}층에 도착했습니다.")
        except Exception:
            pass

        # 상태 정리
        self.target_floor = None
        self.direction = 0

    # 버튼 상태 일괄 전환 --------------------------------------------------
    def _set_buttons_state(self, state):
        for btn in self.floor_buttons:
            btn.config(state=state)

    # 랜덤 재배치 ----------------------------------------------------------
    def reset_elevator(self):
        # 이동 중이면 리셋하지 않음(혼선 방지)
        if self.moving:
            return
        # 스케줄 클리어
        self._cancel_scheduled_jobs()

        # 상태 초기화
        self.elevator_floor = random.randint(1, self.TOTAL_FLOORS)
        self._update_status_text()
        self.result_var.set("예상 도착시간: -")
        self.per_floor_var.set("층 이동 카운트다운: -")
        self.arrival_var.set("")
        self.target_floor = None
        self.remaining_floors = 0
        self.total_remaining_seconds = 0
        self.direction = 0

    # 예약된 after 작업 취소 ----------------------------------------------
    def _cancel_scheduled_jobs(self):
        if self._tick_job is not None:
            try:
                self.root.after_cancel(self._tick_job)
            except Exception:
                pass
            self._tick_job = None
        if self._per_floor_job is not None:
            try:
                self.root.after_cancel(self._per_floor_job)
            except Exception:
                pass
            self._per_floor_job = None

if __name__ == "__main__":
    # Tkinter 앱 실행 -----------------------------------------------------
    root = tk.Tk()
    app = ElevatorApp(root)
    root.mainloop()
