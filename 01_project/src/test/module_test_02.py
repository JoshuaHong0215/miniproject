# 캠에서 YOLO + ROI로 사람과 개 감지, ROI 내부 감지 시 빨간 박스 + "CAUTION" 배너 표시

import cv2
import time
from ultralytics import YOLO

# =========================
# 사용자 설정값
# =========================
MODEL_CANDIDATES = ['yolo11n.pt', 'yolov8n.pt']  # 자동 시도 목록
CONFIDENCE = 0.5          # 최소 신뢰도
USE_GPU = True            # CUDA 사용 가능 시 내부적으로 사용
SHOW_FPS = True           # 좌상단 FPS 표기
CAM_INDEX = 0             # 웹캠 인덱스
TARGET_CLASS_IDS = [0, 16]  # COCO: person=0, dog=16
CLASS_NAME = {0: 'person', 16: 'dog'}
# =========================

def load_model():
    """
    # 목적: 가중치 파일 후보들을 순차로 시도하여 YOLO 모델 로드
    """
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

def draw_caution_banner(frame, kinds):
    """
    # 목적: 화면 상단 중앙에 'CAUTION: 대상들' 배너 표시
    # 입력: kinds = {'person', 'dog'} 등 경고를 유발한 클래스명들의 집합
    """
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

# def draw_roi(frame, roi):
#     """
#     # 목적: 설정된 ROI를 화면에 녹색 박스로 시각화
#     """
#     if roi is None:
#         return
#     x, y, w, h = roi
#     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
#     cv2.putText(frame, "ROI", (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 0), 2, cv2.LINE_AA)

# def intersects_roi(box_xyxy, roi):
#     """
#     # 목적: 바운딩박스 중심점이 ROI 내부인지 판정
#     # 입력: box_xyxy = (x1, y1, x2, y2), roi = (rx, ry, rw, rh) or None
#     # 반환: True/False
#     """
#     if roi is None:
#         return True
#     x1, y1, x2, y2 = box_xyxy
#     cx = (x1 + x2) // 2
#     cy = (y1 + y2) // 2

#     rx, ry, rw, rh = roi
#     return (rx <= cx <= rx + rw) and (ry <= cy <= ry + rh)

# def intersects_roi(box_xyxy, roi):
#     """
#     box_xyxy: (x1, y1, x2, y2)
#     roi: (rx, ry, rw, rh)
#     """
#     if roi is None:
#         return True  # ROI 미설정 시 무조건 True

#     x1, y1, x2, y2 = box_xyxy
#     rx, ry, rw, rh = roi
#     roi_x2, roi_y2 = rx + rw, ry + rh

#     # 겹치는 영역 크기 계산
#     overlap_x = max(0, min(x2, roi_x2) - max(x1, rx))
#     overlap_y = max(0, min(y2, roi_y2) - max(y1, ry))

#     return overlap_x > 0 and overlap_y > 0


def main():
    # 카메라 오픈
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    # 모델 로드
    model = load_model()

    # ROI 정보
    roi = None
    last_time = time.time()

    print("[INFO] Press 's' to select ROI, 'c' to clear ROI, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame.")
            break

        # 추론: 사람(0), 개(16)만
        results = model(frame, conf=CONFIDENCE, classes=TARGET_CLASS_IDS, verbose=False)[0]

        # ROI 시각화
        # draw_roi(frame, roi)

        # 경고 상태 및 원인 클래스들
        caution_triggered = True
        caution_kinds = set()

        # 디텍션 처리
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                # 정수 좌표로 변환
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                class_name = CLASS_NAME.get(cls_id, str(cls_id))

                # ROI 내부 여부 판정(박스 중심 기준)
                in_roi = intersects_roi((x1, y1, x2, y2), roi)

                # ROI 내부면 경고용 빨간 박스, 외부면 회색 박스
                color = (0, 0, 255) if in_roi else (180, 180, 180)
                if in_roi:
                    caution_triggered = True
                    caution_kinds.add(class_name)

                # 박스/라벨
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                # 콘솔 로그(디버깅)
                if in_roi:
                    print(f"[CAUTION] {class_name} in ROI: conf={conf:.2f}, box=({x1},{y1},{x2},{y2})")

        # 경고 배너
        if caution_triggered:
            draw_caution_banner(frame, caution_kinds)

        # FPS 표기
        if SHOW_FPS:
            now = time.time()
            fps = 1.0 / (now - last_time) if (now - last_time) > 0 else 0.0
            last_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 220, 30), 2, cv2.LINE_AA)

        # 출력
        cv2.imshow("YOLO Person & Dog with ROI (Press s/c/q)", frame)

        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):      # 종료
            break
        elif key == ord('s'):    # ROI 선택
            frozen = frame.copy()
            sel = cv2.selectROI("Select ROI (Enter/Space=OK, c=Cancel)", frozen, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select ROI (Enter/Space=OK, c=Cancel)")
            x, y, w, h = map(int, sel)
            if w > 0 and h > 0:
                roi = (x, y, w, h)
                print(f"[INFO] ROI set to (x={x}, y={y}, w={w}, h={h})")
            else:
                print("[INFO] ROI selection canceled.")
        elif key == ord('c'):    # ROI 해제
            roi = None
            print("[INFO] ROI cleared.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
