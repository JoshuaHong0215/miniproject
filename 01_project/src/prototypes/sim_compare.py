# sim_compare.py
# Elevators: baseline vs applied dispatch policy simulation (headless)
from dataclasses import dataclass
from collections import deque
import random, statistics
from typing import Optional


UP, DOWN = "UP", "DOWN"

# === 시뮬레이터 파라미터(당신 코드와 대체로 일치) ===
FLOORS = 15
FLOOR_TIME = 3.0             # 층간 이동(초)
OPEN = 4.0
HOLD_DEFAULT = 7.0
HOLD_SHORT = 4.0
CLOSE = 4.0
DOOR_NO_PERSON = False       # 시뮬엔 YOLO 없음

@dataclass
class Call:
    t: float
    floor: int
    kind: str  # 'floor' or 'hall'
    direction: Optional[str]       # hall: UP/DOWN, floor: None
    id: int                   # 추적용

class PolicySim:
    """
    두 가지 정책을 스위치로 돌리는 경량 시뮬.
    - mode='baseline' : 개선 전 로직 근사
    - mode='applied'  : 개선 로직(내림차순 모드, FLOOR 포함, requeue 등)
    """
    def __init__(self, mode='baseline', floors=FLOORS):
        self.mode = mode
        self.floors = floors
        self.pos = 1.0
        self.target = None
        self.state = "IDLE"  # IDLE/OPENING/HOLD/CLOSING/MOVING
        self.phase_end = 0.0
        self.pending = deque()      # (floor, direction, ts, id)
        self.down_sweep_active = False
        self.now = 0.0
        self.next_call_id = 0

        # metrics
        self.wait_map = {}     # id -> press_time
        self.served_map = {}   # id -> serve_time
        self.travel_floors = 0.0
        self.completed = 0

    # --- 유틸 ---
    def _door_cycle(self):
        # OPEN -> HOLD -> CLOSE (사람 감지 없음 가정)
        self.state = "OPENING"; self.phase_end = self.now + OPEN
        self._advance_time_to(self.phase_end)
        self.state = "HOLD"; self.phase_end = self.now + (HOLD_DEFAULT if not DOOR_NO_PERSON else HOLD_SHORT)
        self._advance_time_to(self.phase_end)
        self.state = "CLOSING"; self.phase_end = self.now + CLOSE
        self._advance_time_to(self.phase_end)
        self.state = "IDLE"

    def _advance_time_to(self, t):
        self.now = t

    def _push_call(self, floor, direction, ts, cid):
        # 중복 단순 차단
        for f,d,_,_cid in self.pending:
            if f==floor and d==direction: return
        self.pending.append((floor, direction, ts, cid))

    # --- 외부에서 주입할 이벤트 처리 ---
    def inject_call(self, floor, direction, press_time, kind='hall'):
        cid = self.next_call_id; self.next_call_id += 1
        self.wait_map[cid] = press_time
        self._push_call(floor, direction if kind=='hall' else None, press_time, cid)

    # --- 정책 핵심 ---
    def _select_next_call(self):
        if not self.pending: return None
        cur = self.pos
        calls = list(self.pending)

        # 동일층 최우선
        same = [(f,d,ts,cid) for (f,d,ts,cid) in calls if int(round(cur))==f]
        if same:
            target = same[0]
            self.pending.remove(target)
            return target

        if self.mode=='applied' and self.down_sweep_active:
            # 아래쪽의 DOWN 또는 FLOOR(None) 중 '가장 높은 층'
            downs_below = [(f,d,ts,cid) for (f,d,ts,cid) in calls if f < cur and (d==DOWN or d is None)]
            if downs_below:
                target = max(downs_below, key=lambda x: x[0])
                self.pending.remove(target)
                return target
            else:
                self.down_sweep_active = False  # 더 갈 곳 없으면 해제

        # 일반 규칙: (applied/baseline 공통 근사)
        downs_above = [(f,d,ts,cid) for (f,d,ts,cid) in calls if d==DOWN and f>cur]
        if downs_above:
            target = max(downs_above, key=lambda x: x[0])
        else:
            downs_below = [(f,d,ts,cid) for (f,d,ts,cid) in calls if d==DOWN and f<cur]
            if downs_below:
                target = max(downs_below, key=lambda x: x[0])
            else:
                ups_below = [(f,d,ts,cid) for (f,d,ts,cid) in calls if d==UP and f<cur]
                if ups_below:
                    target = min(ups_below, key=lambda x: x[0])
                else:
                    # 가장 가까운 층 (방향 가중치 0)
                    target = min(calls, key=lambda x: abs(x[0]-cur))
        self.pending.remove(target)
        return target

    def _update_moving_target(self):
        if self.target is None: return
        going_up = self.target > self.pos
        if self.mode=='applied' and going_up:
            higher_down = [f for (f,d,ts,cid) in self.pending if d==DOWN and f>self.target]
            if higher_down:
                old_target = int(self.target)
                new_target = max(higher_down)
                if new_target > old_target:
                    # 기존 목표 re-queue (DOWN 또는 FLOOR(None))
                    need = True
                    for f,d,ts,cid in self.pending:
                        if f==old_target and (d==DOWN or d is None):
                            need=False; break
                    if need and self.active_call is not None:
                        of, od, ots, ocid = self.active_call
                        if of==old_target and (od==DOWN or od is None):
                            self._push_call(of, od, self.now, ocid)
                    self.target = new_target

    # --- 주행 루프(이벤트 시간 순으로 처리) ---
    def run_until(self, T, external_events):
        """
        external_events: [(t, floor, direction, kind), ...] 시간순 입력
        시뮬레이터가 시간이 흐르면서 콜을 소화
        """
        self.active_call = None
        i = 0
        while self.now < T:
            # 도착/IDLE 상태에서 다음 목적지 선정
            if self.state == "IDLE" and self.target is None:
                # 외부 이벤트 주입
                while i < len(external_events) and external_events[i][0] <= self.now:
                    t,f,d,kind = external_events[i]; i+=1
                    self.inject_call(f,d,t,kind)
                # 선택
                nxt = self._select_next_call()
                if nxt is not None:
                    f,d,ts,cid = nxt
                    self.active_call = (f,d,ts,cid)
                    # 내림차순 모드 ON 조건
                    if self.mode=='applied' and d == DOWN:
                        self.down_sweep_active = True
                    # 이동 시작
                    self.target = f
                    self.state = "MOVING"

            if self.state == "MOVING" and self.target is not None:
                self._update_moving_target()
                direction = 1.0 if self.target > self.pos else -1.0
                # 다음 외부 이벤트 시각으로 한 번에 점프 (연속 시간 단순화)
                next_event_t = external_events[i][0] if i < len(external_events) else float('inf')
                # 목표까지 남은 시간
                time_to_target = abs(self.target - self.pos) * FLOOR_TIME
                # 다음에 일어날 시각
                dt = min(time_to_target, max(1e-6, next_event_t - self.now))
                self.travel_floors += abs((direction * dt / FLOOR_TIME))
                self.pos += direction * (dt / FLOOR_TIME)
                self.now += dt
                if abs(self.pos - self.target) < 1e-9:  # 도착
                    self.pos = float(self.target)
                    # 도착 시 해당 층 모든 콜 소등/처리
                    self._serve_all_calls_on_floor(int(self.pos))
                    self.target = None
                    self._door_cycle()
                    continue
                else:
                    # 이동 도중 외부 이벤트 주입
                    while i < len(external_events) and external_events[i][0] <= self.now:
                        t,f,d,kind = external_events[i]; i+=1
                        self.inject_call(f,d,t,kind)
                    continue

            # IDLE인데 외부 이벤트만 기다려야 하면 시간을 점프
            if self.state == "IDLE" and self.target is None:
                if i < len(external_events):
                    self.now = max(self.now, external_events[i][0])
                    continue
                else:
                    break  # 끝

        # 끝까지 남은 이벤트는 대기시간 계산에서 미처리로 남음
        return self._metrics()

    def _serve_all_calls_on_floor(self, floor):
        # 해당 층의 모든 콜 처리 (FLOOR / HALL-UP / HALL-DOWN)
        # 대기시간 기록
        for it in list(self.pending):
            f,d,ts,cid = it
            if f == floor:
                self.served_map[cid] = self.now
                self.pending.remove(it)
                self.completed += 1
        # 현재 active_call이 그 층이면 역시 기록
        if self.active_call is not None:
            af, ad, ats, acid = self.active_call
            if af == floor:
                self.served_map[acid] = self.now
                self.completed += 1
                self.active_call = None

    def _metrics(self):
        waits = []
        for cid, t0 in self.wait_map.items():
            if cid in self.served_map:
                waits.append(self.served_map[cid] - t0)
        avg = statistics.mean(waits) if waits else 0.0
        p90 = statistics.quantiles(waits, n=10)[-1] if len(waits) >= 10 else (sorted(waits)[int(0.9*len(waits))] if waits else 0.0)
        p95 = statistics.quantiles(waits, n=20)[-1] if len(waits) >= 20 else (sorted(waits)[int(0.95*len(waits))] if waits else 0.0)
        return {
            "calls": len(waits),
            "avg_wait": avg,
            "p90_wait": p90,
            "p95_wait": p95,
            "travel_floors": self.travel_floors,
            "time_end": self.now
        }

# === 테스트 시나리오 생성 ===
def scenario_from_story():
    # 사용자가 주신 시나리오 타임라인 근사
    ev = []
    t=0.0
    # 1) 1F HALL-UP
    ev.append((t, 1, UP, 'hall'))
    # 2) 1F 탑승 후 내부 5F
    ev.append((t+1.0, 5, None, 'floor'))
    # 4) 이동 중 4F/2F DOWN
    ev.append((t+4.0, 4, DOWN, 'hall'))
    ev.append((t+6.0, 2, DOWN, 'hall'))
    # 7) 4F 도착 후 내부 1F
    ev.append((t+30.0, 1, None, 'floor'))  # 대략적인 시간
    return sorted(ev, key=lambda x:x[0])

def random_scenario(T=600.0, rate_per_min=8.0):
    # 포아송 도착으로 임의 콜 생성 (층/방향/종류 랜덤)
    # 실제 테스트용: 여러 seed로 Monte Carlo
    lam = rate_per_min/60.0
    t=0.0; ev=[]
    rng = random.Random(0)
    while t < T:
        # exp inter-arrival
        dt = -1.0 * (1.0/lam) * (rng.random() and ( (1.0/lam) * -1 ))
        # 위 한 줄은 간단화를 위해 고정 간격도 가능:
        dt = max(1.0, rng.expovariate(lam))
        t += dt
        if t>=T: break
        floor = rng.randint(1, FLOORS)
        kind = 'hall' if rng.random()<0.7 else 'floor'
        if kind=='hall':
            if floor==1: direction=UP
            elif floor==FLOORS: direction=DOWN
            else: direction = UP if rng.random()<0.5 else DOWN
        else:
            direction=None
        ev.append((t, floor, direction, kind))
    ev.sort(key=lambda x:x[0])
    return ev

def run_compare(ev):
    base = PolicySim(mode='baseline')
    app  = PolicySim(mode='applied')

    mb = base.run_until(T=3600, external_events=ev)
    ma = app.run_until(T=3600, external_events=ev)

    def pretty(m):
        return (f"calls={m['calls']}, avg_wait={m['avg_wait']:.1f}s, "
                f"p90={m['p90_wait']:.1f}s, p95={m['p95_wait']:.1f}s, "
                f"travel_floors={m['travel_floors']:.1f}, time_end={m['time_end']:.1f}s")

    print("[Baseline] ", pretty(mb))
    print("[Applied ] ", pretty(ma))
    if mb['avg_wait']>0:
        gain = 100.0*(mb['avg_wait']-ma['avg_wait'])/mb['avg_wait']
        print(f"Avg wait reduction: {gain:.1f}%")

if __name__ == "__main__":
    # ① 시나리오 재현
    ev = scenario_from_story()
    print("=== Story Scenario ===")
    run_compare(ev)

    # ② 랜덤 시나리오 비교(원하면 주석 해제)
    # ev2 = random_scenario(T=1200, rate_per_min=10.0)
    # print("\n=== Random Scenario ===")
    # run_compare(ev2)
