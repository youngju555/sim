# -*- coding: utf-8 -*-
"""
MyCobot 320 M5 (pymycobot)
[개선 버전 v5.11.1 - YOLO + 회전각 보정 + Z축 높이 보정]
(핸드셰이크 ACK 이벤트 추가판)
"""

import threading
import cv2
import time
import argparse
import numpy as np
from ultralytics import YOLO
import queue  # [!!! v5.9 추가 !!!]

# ---------------------------------------------------------------------------
# 0. 로봇 클래스 불러오기
# ---------------------------------------------------------------------------
try:
    from pymycobot.mycobot320 import MyCobot320 as CobotClass
except Exception:
    from pymycobot.mycobot import MyCobot as CobotClass

# ---------------------------------------------------------------------------
# 1. 전역 변수, Lock, Event [!!! v5.11 수정 !!!]
# ---------------------------------------------------------------------------
g_target_object = None  # [v5.11] YOLO가 계산한 로봇 좌표 + 클래스 ID + 각도
g_coord_lock = threading.Lock()  # 위 좌표를 안전하게 읽고 쓰기 위한 Lock
args = None  # argparse 결과

# [v5.9] 스레드 간 통신용 Event
e_robot_task_ready = threading.Event()  # YOLO -> Robot "물건 찾았다, 출발해"
e_robot_task_done = threading.Event()  # Robot -> YOLO "작업 끝났다, 다시 찾아도 돼"
e_robot_task_done.set()  # 초기 상태는 "작업 완료" (즉시 탐지 시작 가능)

# [v5.12 추가] 로봇이 YOLO의 ready 신호를 수신(ack)했음을 알리는 Event
e_robot_ack_received = threading.Event()

# [v5.9] 스레드 간 프레임 전달용 Queue
frame_queue = queue.Queue(maxsize=1)
# 디버그 및 GUI 표시용 (YOLO가 처리한 최종 프레임)
processed_frame_buffer = {"frame": None}

# ---------------------------------------------------------------------------
# 2. 로봇 기본 자세/캘리브레이션 값 [!!! v5.11 수정 !!!]
# ---------------------------------------------------------------------------
POSES = {
    "Home": [59.8, -215.9, 354.6, -175.33, 8.65, 86.68],  # 시작/대기 위치
    "Place": [105.8, -65.0, 483.4, -116.46, 4.87, -78.69],  # (사용자 정의 - 현재 로직에선 미사용)
    "Box1": [291.3, 210.0, 200, -172.57, -1.46, -87.15],  # 1. 파란색 놓는 곳
    "Box2": [333.4, 11.7, 200, -175.19, -0.08, -89.53],  # 2. 빨간색 놓는 곳
    "Box3": [319.9, -169.5, 200, -172.32, -2.86, -87.15],  # 3. 노란색 놓는 곳
    "Box1_up": [229.8, 132.6, 386.4, -147.34, 9.15, -74.66],  # Box1 접근(위)
    "Box2_up": [264.0, -1.3, 379.0, -153.71, 11.7, -106.33],  # Box2 접근(위)
    "Box3_up": [228.0, -203.0, 362.8, -146.13, 15.2, -149.53],  # Box3 접근(위)
}

DEFAULT_SPEED = 20
CAMERA_MATRIX = np.array([
    [539.13729067, 0.0, 329.02126026],
    [0.0, 542.34217387, 242.10995541],
    [0.0, 0.0, 1.0]
])
DIST_COEFFS = np.array([[0.20528603, -0.76664068, -0.00096614, 0.00111892, 0.97630004]])

# [!!! v5.11 추가 !!!] 각도 계산을 위한 HSV 색상 범위
# (class_id 0=blue, 1=red, 2=yellow 매핑)
COLOR_RANGES = {
    "red": ([0, 120, 70], [10, 255, 255]),
    "green": ([35, 80, 40], [85, 255, 255]),
    "blue": ([90, 80, 70], [130, 255, 255]),
    "yellow": ([20, 100, 100], [35, 255, 255]),
    "red2": ([170, 120, 70], [180, 255, 255])
}

# ---------------------------------------------------------------------------
# 3. 픽셀 좌표 → 로봇 좌표 변환 (v5.9 원본)
# ---------------------------------------------------------------------------
def pixel_to_robot(cx, cy, distance_cm, frame_w, frame_h):
    # (v5.9와 동일)
    pts = np.array([[[cx, cy]]], dtype=np.float32)
    undistorted_pts = cv2.undistortPoints(pts, CAMERA_MATRIX, DIST_COEFFS, P=None)
    norm_x, norm_y = undistorted_pts[0, 0]
    scale_z = distance_cm * 10.0
    x_cam = norm_x * scale_z
    y_cam = norm_y * scale_z
    
    TCP_BASE_OFFSET_X = 59.8
    TCP_BASE_OFFSET_Y = -215.9
    CAMERA_TO_TCP_OFFSET_X = 75.0
    CAMERA_TO_TCP_OFFSET_Y = 0.0
    
    robot_x = TCP_BASE_OFFSET_X + CAMERA_TO_TCP_OFFSET_X + y_cam
    robot_y = TCP_BASE_OFFSET_Y + CAMERA_TO_TCP_OFFSET_Y + x_cam
    
    TCP_BASE_OFFSET_Z = 354.6
    robot_z_ignored = TCP_BASE_OFFSET_Z - scale_z
    
    return {"x": round(robot_x, 2), "y": round(robot_y, 2), "z_debug": round(robot_z_ignored, 2)}

# ---------------------------------------------------------------------------
# 4. [신규 v5.9] 카메라 '읽기' 스레드 (초고속 영상 수급)
# ---------------------------------------------------------------------------
def camera_read_thread(stop_event, cap, frame_queue):
    print("📷 카메라 '읽기' 스레드 시작")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass
        
        time.sleep(0.01)
    print("📷 카메라 '읽기' 스레드 종료")

# ---------------------------------------------------------------------------
# 5. [!!! v5.11 수정 !!!] 'YOLO + 각도 계산' 처리 스레드 (느린 두뇌)
# ---------------------------------------------------------------------------
def yolo_process_thread(stop_event, frame_queue, model):
    """Queue에서 프레임을 꺼내서 YOLO 예측 및 각도 계산 수행"""
    global g_target_object, g_coord_lock, processed_frame_buffer
    
    print("🧠 YOLO '처리+각도계산' 스레드 시작")
    stable_frames = 0
    
    while not stop_event.is_set():
        # 1. 로봇이 작업 중이면 탐지 안 함
        if not e_robot_task_done.is_set():
            stable_frames = 0
            time.sleep(0.1)
            continue
            
        # 2. 로봇이 쉬고 있으면, Queue에서 최신 프레임 꺼내기
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # 3. YOLO 예측 (가장 느린 부분)
        results = model.predict(frame, imgsz=640, conf=0.6, verbose=False)
        
        # YOLO 결과 프레임 (시각화)
        frame_vis = results[0].plot()

        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        # 5. 물체 감지 및 좌표/각도 계산
        if len(boxes) > 0:
            stable_frames += 1
            if stable_frames >= 3:  # 3프레임 연속 감지 시 "확정"
                x1, y1, x2, y2 = map(int, boxes[0])
                class_id = int(classes[0])
                
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                distance_cm = 19.0  # 임시 고정거리
                
                # --- [!!! v5.11 추가: 각도 계산 로직 !!!] ---
                angle = 0.0  # 기본값

                # 1. class_id를 기반으로 색상 이름 매핑
                if class_id == 0: target_color_name = "blue"
                elif class_id == 1: target_color_name = "red"
                elif class_id == 2: target_color_name = "yellow"
                else: target_color_name = "blue" # 감지 안되면 파란색으로 간주

                # 2. YOLO가 감지한 영역(ROI)만 잘라냄
                roi = frame[y1:y2, x1:x2]
                
                if roi.size > 0: # ROI가 유효한 경우에만
                    # 3. 색상 기반 마스크 생성
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    
                    lower, upper = COLOR_RANGES[target_color_name]
                    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                    
                    # 'red'의 경우 범위가 2개일 수 있음
                    if target_color_name == "red":
                         lower2, upper2 = COLOR_RANGES["red2"]
                         mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
                         mask = cv2.bitwise_or(mask, mask2)

                    mask = cv2.erode(mask, None, iterations=2)
                    mask = cv2.dilate(mask, None, iterations=2)
                    
                    # 4. 컨투어를 찾아 최소 사각형으로 각도 계산
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        c = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(c) > 100: # ROI 내에서 최소 100픽셀 이상
                            rect = cv2.minAreaRect(c)
                            (w_box, h_box) = rect[1]
                            raw_angle = rect[2]

                            # OpenCV 각도 정규화 (0~-90도 범위)
                            if w_box < h_box:
                                angle = raw_angle + 90
                            else:
                                angle = raw_angle
                            
                            print(f"📐 각도 계산 성공: {angle:.2f} (Raw: {raw_angle:.2f})")
                            
                            # (디버그) 계산된 사각형을 원본 프레임에 다시 그리기
                            box = cv2.boxPoints(rect)
                            box = np.intp(box)
                            # box좌표는 ROI(x1, y1) 기준이므로 원본 프레임 기준으로 오프셋
                            cv2.drawContours(frame_vis, [box + np.array([x1, y1])], 0, (0, 255, 255), 2)
                # --- [각도 계산 로직 종료] ---

                print(f"🎯 YOLO 객체 중심: ({cx}, {cy}), 클래스 ID: {class_id}, 각도: {angle:.2f}")
                h, w, _ = frame.shape
                coord = pixel_to_robot(cx, cy, distance_cm, w, h)

                with g_coord_lock:
                    # [v5.11] 좌표, 클래스 ID, 각도를 함께 저장
                    g_target_object = {"coord": coord, "class_id": class_id, "angle": angle}
                
                # --- [v5.12 변경: ACK 핸드셰이크 추가] ---
                e_robot_task_ready.set()   # 로봇 스레드에게 "출발 신호"
                e_robot_task_done.clear()  # "탐지 임무 완료, 로봇 끝날 때까지 대기"

                # 로봇이 신호를 수신(ACK)할 때까지 잠깐 대기 (타임아웃으로 무한 대기 방지)
                if not e_robot_ack_received.wait(timeout=1.0):
                    print("⚠️ 로봇 응답 지연 - YOLO: ACK 타임아웃 (1.0s)")
                else:
                    print("🤝 로봇이 YOLO 신호 수신(ACK) 확인")
                e_robot_ack_received.clear()  # 다음 라운드를 위해 초기화
                # --- [ACK 핸드셰이크 종료] ---

                stable_frames = 0
        else:
            stable_frames = 0
        
        # 4. GUI 표시용 프레임 (시각화 최종본)
        processed_frame_buffer["frame"] = frame_vis
            
    print("🧠 YOLO '처리+각도계산' 스레드 종료")

# ---------------------------------------------------------------------------
# 6. [!!! v5.11 (Z축 수정) !!!] '로봇 제어' 스레드 (느린 팔다리)
# ---------------------------------------------------------------------------
def robot_control_thread(stop_event, mc, dry_run):
    """로봇의 모든 움직임(sleep 포함)을 전담"""
    global g_target_object, g_coord_lock
    
    print("🤖 로봇 '제어' 스레드 시작")
    
    # 1. (딱 한 번) 홈 위치로 이동
    if not dry_run and mc is not None:
        print("🤖 로봇을 홈 위치로 이동합니다...")
        mc.send_coords(POSES["Home"], DEFAULT_SPEED)
        time.sleep(3)
        print("🏠 홈 위치 도달. 탐지를 시작합니다.")
    else:
        print("🏠 [dry-run] 홈 위치 도달. 탐지를 시작합니다.")
        
    e_robot_task_done.set() # YOLO가 탐지를 시작하도록 허용

    # 2. 메인 루프 (신호 대기)
    while not stop_event.is_set():
        # e_robot_task_ready 신호가 올 때까지 무한정 대기 (Blocking)
        if not e_robot_task_ready.wait(timeout=0.5):
            continue # 0.5초마다 stop_event 체크

        # --- [v5.12 변경: YOLO에게 ACK 전송] ---
        # YOLO가 보낸 ready 신호를 수신했음을 알려줌 (즉시 ACK)
        e_robot_ack_received.set()
        # --- [ACK 전송 완료] ---

        # 신호가 오면, 좌표와 클래스 ID, 각도를 가져와서 전체 시퀀스 실행
        current_task = None
        with g_coord_lock:
            if g_target_object is not None:
                current_task = g_target_object.copy()
                g_target_object = None
        
        if current_task:
            current_coord = current_task["coord"]
            class_id = current_task["class_id"]
            angle = current_task.get("angle", 0.0)  # [!!! v5.11 추가 !!!]
            
            print(f"🤖 인식 성공 → 로봇 이동 시작: {current_coord}, 클래스 ID: {class_id}, 각도: {angle:.2f}")
            pick_x = current_coord["x"]
            pick_y = current_coord["y"]

            # [v5.10] 클래스 ID에 따라 목표 위치 결정
            if class_id == 0: # 1. Blue
                place_pose_name = "Box1"
                approach_pose_name = "Box1_up"
            elif class_id == 1: # 2. Red
                place_pose_name = "Box2"
                approach_pose_name = "Box2_up"
            elif class_id == 2: # 3. Yellow
                place_pose_name = "Box3"
                approach_pose_name = "Box3_up"
            else:
                print(f"⚠️ 알 수 없는 클래스 ID: {class_id}. 기본값 'Box2'로 이동합니다.")
                place_pose_name = "Box2"
                approach_pose_name = "Box2_up"
            
            # POSES 딕셔너리에서 실제 좌표 배열 가져오기
            place_pose = POSES[place_pose_name]
            approach_pose = POSES[approach_pose_name]
            print(f"  ↳ 목표 지점: {place_pose_name}")

            # [!!! v5.11 (Z축 수정) !!!] v8.0 코드의 고정 Z 값 사용
            Z_APPROACH = 300.0  # 1. 접근 높이 (v8.0 값)
            Z_GRASP = 300.0     # 2. 잡기 높이 (v8.0에서 260.0+40 = 300.0 이었음)
            Z_LIFT = 360.0      # 3. 들어올린 높이 (v8.0에서 260.0+100 = 360.0 이었음)
            
            PICK_RX, PICK_RY, PICK_RZ = -175.33, 8.65, 86.68 # 기본 RZ(Yaw) 자세
            
            # [!!! v5.11 수정 !!!] RZ(Yaw) 값에 계산된 각도 보정
            yaw_offset = angle * 0.35 
            corrected_rz = PICK_RZ + yaw_offset
            print(f"  ↳ RZ 보정: {corrected_rz:.2f} (기본: {PICK_RZ} + 오프셋: {yaw_offset:.2f} (각도: {angle:.2f}))")

            print(f"  ↳ 접근Z={Z_APPROACH:.1f}, 잡기Z={Z_GRASP:.1f}, 들어올리기Z={Z_LIFT:.1f}")

            if not dry_run and mc is not None:
                # --- v5.8 픽업 시퀀스 (v5.10이 로직 사용) ---
                mc.set_gripper_value(50, 80, 1) # 그리퍼 열기
                time.sleep(1)
                
                mc.send_coords([pick_x, pick_y, Z_APPROACH, PICK_RX, PICK_RY, corrected_rz], 25, 0) # mode=0 (각도)
                time.sleep(3) # v8.0 기준 3초
                
                mc.send_coords([pick_x, pick_y, Z_GRASP, PICK_RX, PICK_RY, corrected_rz], 15, 0) # mode=0 (각도)
                time.sleep(2) # v8.0 기준 2초
                
                mc.set_gripper_value(8, 20, 1) # 그리퍼 닫기 (v8.0 기준)
                time.sleep(1.5)
                
                mc.send_coords([pick_x, pick_y, Z_LIFT, PICK_RX, PICK_RY, corrected_rz], 15, 0) # mode=0 (각도)
                time.sleep(1.5) # v8.0 기준 1.5초
                
                mc.send_coords(approach_pose, DEFAULT_SPEED, 1) # 예: Box1_up (선형)
                time.sleep(3)
                mc.send_coords(place_pose, DEFAULT_SPEED, 1) # 예: Box1 (선형)
                time.sleep(3)
                
                mc.set_gripper_state(0, 80) # 그리퍼 열기
                time.sleep(1.5)
                mc.send_coords(POSES["Home"], DEFAULT_SPEED) # mode=0 (기본값)
                time.sleep(3)
                print("✅ 1회 피킹 완료")
            else:
                print("  [dry-run] 로봇 없이 동작 흐름만 실행")
                time.sleep(5) # 시뮬레이션 대기

            # 작업이 끝났음을 알림
            e_robot_task_ready.clear() # "출발 신호" 끄기
            e_robot_task_done.set()  # YOLO에게 "다시 탐지 시작" 신호
            
    print("🤖 로봇 '제어' 스레드 종료")

# ---------------------------------------------------------------------------
# 7. 메인 루프 (GUI 담당)
# ---------------------------------------------------------------------------
def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--speed", type=int, default=20)
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", type=str, default="best.pt")
    args = parser.parse_args()

    print(f"🧠 YOLOv8 모델('{args.model}') 로드 중...")
    try:
        model = YOLO(args.model, task="detect")
        print("✅ YOLO 모델 로드 성공")
    except Exception as e:
        print(f"❌ YOLO 모델 로드 실패: {e}")
        return
        
    stop_event = threading.Event()
    mc = None
    cap = None
    
    threads = []

    try:
        # 1) 로봇 초기화 (v5.9 원본)
        if not args.dry_run:
            try:
                mc = CobotClass(args.port, args.baud)
                time.sleep(0.5)
                mc.power_on()
                print("🔌 로봇 Power ON 완료")
                mc.set_gripper_value(50, 20, 1) # [!!! v5.11 (Z축 수정) !!!] v8.0 기준으로 열기
                time.sleep(1)
            except Exception as e:
                print(f"❌ 로봇 연결 실패: {e}")
                mc = None
                args.dry_run = True
        else:
            print("🟡 dry-run 모드로 시작")

        # 2) 카메라 초기화 (v5.9 원본)
        print(f"📷 메인: 카메라 {args.camera}번 열기 시도...")
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"⚠️ {args.camera}번 카메라 실패 → 0번으로 재시도")
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("camera open failed")
        print("✅ 메인: 카메라 연결 성공")

        # 3) [v5.9] 3개의 스레드 시작
        
        # Thread 1: 카메라 읽기
        t_cam = threading.Thread(
            target=camera_read_thread,
            args=(stop_event, cap, frame_queue),
            daemon=True
        )
        t_cam.start()
        threads.append(t_cam)

        # Thread 2: YOLO 처리 (v5.11 수정됨)
        t_yolo = threading.Thread(
            target=yolo_process_thread,
            args=(stop_event, frame_queue, model),
            daemon=True
        )
        t_yolo.start()
        threads.append(t_yolo)

        # Thread 3: 로봇 제어 (v5.11 수정됨)
        t_robot = threading.Thread(
            target=robot_control_thread,
            args=(stop_event, mc, args.dry_run),
            daemon=True
        )
        t_robot.start()
        threads.append(t_robot)

        print("✅ 메인 루프 시작 (GUI 표시 담당, q로 종료)")
        
        # 4) 메인 루프 (GUI만 담당)
        while not stop_event.is_set():
            frame = processed_frame_buffer.get("frame")
            
            if frame is None:
                try:
                    frame = frame_queue.get_nowait()
                except queue.Empty:
                    time.sleep(0.01)
                    continue

            cv2.imshow("Camera View", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
                
            time.sleep(0.01)

    except Exception as e:
        print(f"🚨 메인 루프에서 에러 발생: {e}")
    finally:
        # 7) 종료 처리
        print("🛑 종료 신호 감지... 모든 스레드 정리 중...")
        stop_event.set()
        
        for t in threads:
            t.join(timeout=1.0)
            
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        if mc:
            mc.power_off()
        print("🔒 프로그램 종료")


if __name__ == "__main__":
    main()
