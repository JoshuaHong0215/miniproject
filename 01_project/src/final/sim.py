# sim_compare.py
# 09:00 ~ 19:00 동안의 엘리베이터 정차 시퀀스를 생성하고
# 4/7/4 vs 4/3/4 정책을 같은 시퀀스에 적용해 총 소요시간/절감율을 비교
# 출력은 '분' 단위(방법 1: 출력부만 /60 변환)

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

random.seed(42)  # 재현성 고정

# --------- 빌딩/동작 파라미터 ---------
N_FLOORS = 15
FLOOR_TIME = 3.0        # 층간 이동 시간(초)
OPEN_T = 4.0
CLOSE_T = 4.0
HOLD_BASE = 7.0         # 4/7/4
HOLD_SHORT = 3.0        # 4/3/4
DAY_START = 9 * 3600    # 09:00 -> seconds
DAY_END = 19 * 3600     # 19:00 -> seconds
SIM_T = DAY_END - DAY_START

# '다른 층 콜이 대기 중' 간주 기준: 다음 이벤트가 문열림~유지 시간 내에 충분히 근접해 있으면 True
OTHER_CALL_WINDOW = 60.0  # 초 (문열림~유지 중에는 대기콜이 있다고 보기에 충분한 간격)

@dataclass
class StopEvent:
    t: float       # "요청 발생 시각" (초, 00:00 기준)
    floor: int

# --------- 시간대별 현실적 패턴(분당 정차 기대값, IN→OUT&OUT-clear 단축 성사 확률) ---------
def pattern_per_hour(hh: int) -> Tuple[float, float]:
    """
    반환: (rate_per_min, p_short_condition)
    - rate_per_min: 분당 정차 기대값(포아송)
    - p_short_condition: IN→OUT 발생 + OUT영역 3초 클리어가 될 확률(시간대별 가중)
    """
    # 9-10 출근 막바지(업피크), 12-13 점심(왕복 활발), 17-19 하행 피크
    if 9 <= hh < 10:
        return (6.0, 0.55)
    if 10 <= hh < 12:
        return (3.0, 0.40)
    if 12 <= hh < 13:
        return (5.5, 0.65)
    if 13 <= hh < 17:
        return (3.0, 0.40)
    if 17 <= hh < 19:
        return (7.0, 0.70)
    return (2.0, 0.35)

# --------- 시퀀스 생성 ---------
def generate_day_sequence() -> List[StopEvent]:
    """09~19시 구간에서 시간대별 포아송 과정으로 정차 이벤트 생성(층은 1..N_FLOORS 랜덤)"""
    events: List[StopEvent] = []
    cur = DAY_START
    while cur < DAY_END:
        hh = int(cur // 3600)
        rate_per_min, _ = pattern_per_hour(hh)

        # 이번 1분 동안 발생하는 정차 개수
        k = _poisson(rate_per_min)
        # 각 정차의 초 단위 오프셋을 균일분포로 배치
        for _ in range(k):
            sec_offset = random.uniform(0, 60)
            t = cur + sec_offset
            floor = _sample_floor_for_time(hh)
            events.append(StopEvent(t=t, floor=floor))
        cur += 60.0

    # 시간순 정렬
    events.sort(key=lambda e: e.t)
    return events

def _poisson(lam: float) -> int:
    # numpy 없이 포아송 표본
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= random.random()
        if p <= L:
            return k - 1

def _sample_floor_for_time(hh: int) -> int:
    # 시간대별로 낮은층/높은층 쏠림을 살짝 줘서 더 현실적으로
    if 9 <= hh < 10:     # 업피크: 중상층 선호
        w = [1] + [2]*3 + [3]*5 + [2]*4 + [1]*2  # 1..15 가중
    elif 17 <= hh < 19:  # 다운피크: 저층/로비 선호
        w = [4]*3 + [3]*5 + [2]*5 + [1]*2
    else:                # 일반 시간대: 균일에 가깝게
        w = [2]*15
    floors = list(range(1, N_FLOORS+1))
    return random.choices(floors, weights=w, k=1)[0]

# --------- 동일 시퀀스에 대해 정책별 총 소요시간 계산 ---------
def simulate_total_time(events: List[StopEvent], use_short_logic: bool, seed: int = 123) -> float:
    """
    events: 시간순 정렬된 정차 시퀀스(동일 시퀀스를 양 정책에 재사용)
    use_short_logic: True면 4/3/4 적용(조건 성사 시), False면 항상 4/7/4
    반환값: 총 소요시간(초)
    """
    rnd = random.Random(seed)  # 정책 간 동일한 '추가 난수' 사용을 위해 시드 고정

    total_sec = 0.0
    travel_sec = 0.0
    door_sec = 0.0

    cur_floor = 1
    for i, ev in enumerate(events):
        # 이동 시간
        travel = abs(ev.floor - cur_floor) * FLOOR_TIME
        travel_sec += travel
        total_sec += travel
        cur_floor = ev.floor

        # '다른 층 콜 대기' 조건: 다음 이벤트가 가까우면 True
        has_other_calls = False
        if i + 1 < len(events):
            dt_next = events[i+1].t - ev.t
            has_other_calls = (dt_next <= OTHER_CALL_WINDOW)

        # IN→OUT + OUT 3초 클리어 성사 여부(시간대별 확률)
        hh = int(ev.t // 3600)
        _, p_cond = pattern_per_hour(hh)
        cond_hit = rnd.random() < p_cond

        hold = HOLD_BASE
        if use_short_logic and has_other_calls and cond_hit:
            hold = HOLD_SHORT

        door_cycle = OPEN_T + hold + CLOSE_T
        door_sec += door_cycle
        total_sec += door_cycle

    # 마지막에 1층으로 귀환(테스트 일관용)
    travel_back = abs(cur_floor - 1) * FLOOR_TIME
    travel_sec += travel_back
    total_sec += travel_back

    return total_sec

# --------- 실행 ---------
def main():
    # 1) 동일 시퀀스 생성
    events = generate_day_sequence()
    print(f"[생성] 정차 이벤트 수: {len(events)} (09:00~19:00)")

    # 2) 정책별 총 소요시간(초)
    t_base = simulate_total_time(events, use_short_logic=False)  # 4/7/4
    t_fast = simulate_total_time(events, use_short_logic=True)   # 4/3/4 (조건 성사 시 단축)

    # 3) 결과 산출
    improved = t_base - t_fast
    improved_ratio = improved / t_base

    def pct(x): return round(x * 100.0, 2)

    print("\n===== 결과 (동일 시퀀스 적용전/후 비교) =====")
    # ▼▼▼ 여기만 '분' 단위 출력(방법 1) ▼▼▼
    print(f"- 기본(4/7/4)   총 소요시간: {t_base/60:.1f}분")
    print(f"- 단축(4/3/4)   총 소요시간: {t_fast/60:.1f}분")
    print(f"- 절감 시간:     {(improved)/60:.1f}분")

    print("\n[측정항목 : 시간 단축]")
    print(f"개선전 : {pct(1.0)}%")
    print(f"개선 후 : {pct(t_fast / t_base)}%")
    print(f"개선율 : {pct(improved_ratio)}%")

if __name__ == "__main__":
    main()
