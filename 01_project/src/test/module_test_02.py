# YOLO + DoorController (OPEN_HOLD: 사람이 감지되는 동안 무기한 유지, 미감지 3초 후 즉시 닫힘)
import cv2
import time
from ultralytics import YOLO

# =========================
# 사용자 설정값
# =========================
MODEL_CANDIDATES = ['yolo11n.pt', 'yolov8n.pt']  # 자동 시도 목록
CONFIDENCE = 0.5            # 최소 신뢰도
SHOW_FPS = True             # 좌상단 FPS 표기
CAM_INDEX = 0               # 웹캠 인덱스
TARGET_CLASS_IDS = [0, 16]  # COCO: person=0, dog=16 (표시용)
CLASS_NAME = {0: 'person', 16: 'dog'}

# 도어 타이밍(초)
OPENING_SEC = 4             # 문 열리는 시간
CLOSING_SEC = 4             # 문 닫히는 시간
NO_PERSON_GRACE = 3         # 열린 상태에서 '미감지' 지속 대기 시간(초) → 초과 시 닫힘

START_ON_BOOT = True        # 시작 즉시 사이클 시작(테스트 편의용)

# =========================
# 유틸: 모델 로드
# =========================
def load_model():
    """가중치 후보들을 순차 시도하여 YOLO 모델 로드"""
    last_err = None
    for w in MODEL_CANDIDATES:
        try:
            model = YOLO(w)
            print(f"[INFO] Loaded weights: {w}")
            return model
        except Exception as e:
            last_err = e
            print(f"[WARN] Failed to load {w}: {e}")
    raise RuntimeError(f"Failed to load any YOLO weights from {MODEL_CANDIDATES}\nLast error: {last_err}")

# =========================
# UI: 경고 배너
# =========================
def draw_caution_banner(frame, kinds):
    """화면 상단 중앙에 'CAUTION: 대상들' 배너 표시"""
    h, w = frame.shape[:2]
    label = " / ".join(sorted(kinds)) if kinds else "TARGET"
    text = f"CAUTION: {label.upper()}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.8
    thickness = 4
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    y = 60
    pad = 20
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + pad), (0, 0, 255), -1)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

# =========================
# UI: 상태/잔여시간 오버레이
# =========================
def draw_state_overlay(frame, state, remain_sec, open_hold_info):
    """좌상단 상태·잔여시간·OPEN_HOLD 정보 표시"""
    y = 50
    x = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thick = 2
    cv2.putText(frame, f"STATE: {state}", (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    cv2.putText(frame, f"REMAIN: {max(0, int(remain_sec))}s", (x, y + 25), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    cv2.putText(frame, open_hold_info, (x, y + 50), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

# =========================
# 비블로킹 도어 상태머신
# =========================
class DoorController:
    # 상태 상수
    IDLE = "IDLE"
    OPENING = "OPENING"
    OPEN_HOLD = "OPEN_HOLD"
    CLOSING = "CLOSING"

    def __init__(self):
        # 상태/시간
        self.state = self.IDLE
        self.phase_start = 0.0       # 현 단계 시작 시각
        self.last_print_sec = 0      # 초당 카운트 중복 방지

        # OPEN_HOLD용
        self.hold_elapsed_base = 0.0 # OPEN_HOLD 누적 시간 카운트 기준
        self.last_seen_time = None   # 마지막 사람 감지 시각
        self.open_hold_info = ""     # UI 표시용 문구

    def start_cycle(self, now: float):
        """사이클 시작: OPENING으로 진입"""
        self.state = self.OPENING
        self.phase_start = now
        self.last_print_sec = 0
        print("문이 열립니다")  # 즉시 단계 시작 문구 출력

    def update(self, now: float, person_present: bool):
        """
        매 프레임 호출.
        - OPENING: 4초 후 OPEN_HOLD
        - OPEN_HOLD: 사람이 감지되는 동안 무기한 유지, '미감지 3초' 경과 시 즉시 CLOSING
        - CLOSING: 4초. (원하면 닫힘 중 감지 시 재개방 로직을 추가 가능)
        """
        remain = 0.0
        self.open_hold_info = ""

        if self.state == self.IDLE:
            if START_ON_BOOT and not getattr(self, "_boot_started", False):
                self._boot_started = True
                self.start_cycle(now)

        elif self.state == self.OPENING:
            elapsed = now - self.phase_start
            remain = OPENING_SEC - elapsed

            # 초당 카운트(1..OPENING_SEC)
            sec = int(elapsed) + 1
            if 1 <= sec <= OPENING_SEC and sec != self.last_print_sec:
                print(f"{sec} sec")
                self.last_print_sec = sec

            if elapsed >= OPENING_SEC:
                # OPEN_HOLD 진입
                self.state = self.OPEN_HOLD
                self.phase_start = now
                self.last_print_sec = 0
                self.hold_elapsed_base = now         # OPEN_HOLD 총 경과 출력용 기준
                self.last_seen_time = now if person_present else now  # 직전 감지 시각
                print("문이 열린 상태 유지")

        elif self.state == self.OPEN_HOLD:
            # 총 유지 시간(디버그/콘솔 출력용)
            hold_total_elapsed = now - self.hold_elapsed_base
            sec = int(hold_total_elapsed) + 1
            if sec != self.last_print_sec:
                print(f"{sec} sec")
                self.last_print_sec = sec

            # 감지 상태 갱신
            if person_present:
                self.last_seen_time = now

            # 사람 미감지 경과 시간 계산
            no_person_elapsed = (now - self.last_seen_time) if self.last_seen_time is not None else float('inf')

            # UI 정보: 감지면 "HOLDING (person present)", 미감지면 "closing in Ns"
            if person_present:
                self.open_hold_info = "OPEN_HOLD: holding (person present)"
            else:
                remain_to_close = max(0.0, NO_PERSON_GRACE - no_person_elapsed)
                self.open_hold_info = f"OPEN_HOLD: no person → closing in {int(remain_to_close)}s"

            # 미감지 3초 초과 → 즉시 닫힘으로 전환
            if not person_present and no_person_elapsed >= NO_PERSON_GRACE:
                self.state = self.CLOSING
                self.phase_start = now
                self.last_print_sec = 0
                print("문이 닫힙니다")

        elif self.state == self.CLOSING:
            elapsed = now - self.phase_start
            remain = CLOSING_SEC - elapsed

            # (선택) 닫힘 중 사람 감지 시 즉시 재개방을 원하시면 아래 블록 주석 해제
            # if person_present:
            #     self.state = self.OPENING
            #     self.phase_start = now
            #     self.last_print_sec = 0
            #     print("문이 열립니다")
            #     return self.state, remain, self.open_hold_info

            # 초당 카운트(1..CLOSING_SEC)
            sec = int(elapsed) + 1
            if 1 <= sec <= CLOSING_SEC and sec != self.last_print_sec:
                print(f"{sec} sec")
                self.last_print_sec = sec

            if elapsed >= CLOSING_SEC:
                self.state = self.IDLE
                self.phase_start = now
                self.last_print_sec = 0
                print("문이 닫혔습니다")

        return self.state, max(0.0, remain), self.open_hold_info

# =========================
# 메인 루프
# =========================
def main():
    # 카메라 오픈
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    # 모델 로드
    model = load_model()

    # 컨트롤러
    door = DoorController()
    last_time = time.time()

    print("[INFO] Press 'q' to quit.")
    print("[INFO] Press 'o' to manually start a door cycle.")

    window_name = "YOLO + DoorController"
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame.")
            break

        # YOLO 추론
        results = model(frame, conf=CONFIDENCE, classes=TARGET_CLASS_IDS, verbose=False)[0]

        # 감지 상태
        person_present = False
        caution_kinds = set()

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = CLASS_NAME.get(cls_id, str(cls_id))

                # 박스/라벨
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

                # 문 제어는 person만 사용
                if cls_id == 0:
                    person_present = True
                if cls_id in CLASS_NAME:
                    caution_kinds.add(class_name)

        # 경고 배너
        if caution_kinds:
            draw_caution_banner(frame, caution_kinds)

        # 도어 업데이트(비블로킹) + 상태/잔여시간 오버레이
        now = time.time()
        state, remain, open_hold_info = door.update(now, person_present) or (door.IDLE, 0, "")
        draw_state_overlay(frame, state, remain, open_hold_info)

        # FPS
        if SHOW_FPS:
            fps = 1.0 / (now - last_time) if (now - last_time) > 0 else 0.0
            last_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 220, 30), 2, cv2.LINE_AA)

        # 출력
        cv2.imshow(window_name, frame)

        # 키 입력
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o'):  # 수동 사이클 시작(테스트용)
            door.start_cycle(time.time())

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
