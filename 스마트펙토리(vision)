#!/usr/bin/env python3  finalll!!!!!!!!
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger 
from my_robot_interfaces.srv import DetectItem 
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
import time 
import threading 
import math
from PIL import Image as PILImage
from torchvision import transforms
from ultralytics import YOLO

# ==============================================================================
# [설정] 경로 및 파라미터 
# ==============================================================================
YOLO_WEIGHTS_PATH = '/home/young/runs/obb/train3/weights/best.pt' 
WEIGHTS_PATH = '/home/young/final_ws/src/final/final/padim_weights/cube'
DEFECT_IMAGE_SAVE_DIR = '/home/young/final_ws/src/final/defect_images' 

FIXED_SIZE = 120 
NUM_RANDOM_CHANNELS = 300 
TOP_N_PATCHES = 10 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANOMALY_THRESHOLD = 80.0 
TARGET_DEVICE_ID = '/dev/v4l/by-id/usb-046d_C270_HD_WEBCAM_200901010001-video-index0'

# [복구됨] 고해상도(1280x960) 기준 ROI 좌표
ROI_X = 470
ROI_Y = 130
ROI_W = 300
ROI_H = 350

# [복구됨] 박스 상태 확인용 ROI (1280x960 기준)
BOX_ROI_X = 850
BOX_ROI_Y = 580
BOX_ROI_W = 400
BOX_ROI_H = 350

BOX_FULL_THRESHOLD = 3  

CUBE_REAL_SIZE_MM = 50.0 
CENTER_ROBOT_X = -42.0  
CENTER_ROBOT_Y = 225.0   
FRAME_TIMEOUT_SEC = 1.0 

# [천재 심영주 HSV 설정값] (툴 검증 완료)
HSV_LOWER_GREEN = np.array([51, 69, 69])
HSV_UPPER_GREEN = np.array([95, 255, 255])

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        
        self.TOP_N_PATCHES = TOP_N_PATCHES
        self.ANOMALY_THRESHOLD = ANOMALY_THRESHOLD
        self.DEVICE = DEVICE
        
        self.last_detected_box = None
        self.last_detected_quality = ""
        self.last_detected_score = 0.0
        self.last_detect_time = 0.0
        
        self.is_camera_open = False
        self.latest_frame = None
        self.last_frame_time = 0.0    
        self.frame_lock = threading.Lock() 
        self.ai_lock = threading.Lock()    
        self.running = True           

        self.image_pub = self.create_publisher(RosImage, '/vision/defect_img', 10)
        
        # 1. 픽킹 좌표 감지 서비스
        self.detect_srv = self.create_service(DetectItem, '/vision/detect_item', self.handle_detection_request)
        
        # 2. 박스 상태 확인 서비스
        self.box_check_srv = self.create_service(Trigger, '/vision/check_box_full', self.handle_box_check_request)
        
        if not os.path.exists(DEFECT_IMAGE_SAVE_DIR):
            os.makedirs(DEFECT_IMAGE_SAVE_DIR)
        
        self.bridge = CvBridge()
        self.get_logger().info(f"🚀 Vision Node Started. Using Device: {self.DEVICE}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), PILImage.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 모델 로드
        self.load_yolo_model()
        self.load_padim_model()
        self.setup_camera()
        
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        self.display_timer = self.create_timer(0.033, self._display_callback)

    def load_yolo_model(self):
        try:
            self.get_logger().info(f"🚀 Loading YOLO OBB: {YOLO_WEIGHTS_PATH}")
            self.yolo_model = YOLO(YOLO_WEIGHTS_PATH, task='obb')
        except Exception as e:
            self.get_logger().error(f"❌ YOLO Fail: {e}")

    def find_camera_index(self, device_path):
        if not os.path.exists(device_path): return None
        try:
            real_path = os.path.realpath(device_path)
            if 'video' in real_path and real_path.startswith('/dev/video'):
                return int(real_path.split('video')[-1])
            return None
        except: return None
            
    def setup_camera(self):
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            self.cap.release()
        camera_index = self.find_camera_index(TARGET_DEVICE_ID)
        if camera_index is None: camera_index = 4
        
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            # [설정] 고해상도(1280x960) & FPS 15 (대역폭 안정화)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            
            time.sleep(2)
            self.is_camera_open = True
            
            w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.get_logger().info(f"✅ Camera {camera_index} OK. Res: {int(w)}x{int(h)}")
        else:
            self.is_camera_open = False
    
    def _capture_loop(self):
        fail_count = 0
        while self.running and rclpy.ok():
            if not self.is_camera_open or not self.cap.isOpened():
                self.setup_camera(); time.sleep(1.0); continue
            ret, frame = self.cap.read()
            if ret:
                fail_count = 0 
                with self.frame_lock:
                    self.latest_frame = frame
                    self.last_frame_time = time.time()
            else:
                fail_count += 1
                if fail_count > 30:
                    self.is_camera_open = False; self.cap.release(); fail_count = 0; time.sleep(1.0)
            time.sleep(0.005)

    def _display_callback(self):
        if not self.running: return
        frame_to_show = None
        with self.frame_lock:
            if self.latest_frame is not None:
                frame_to_show = self.latest_frame.copy()
        
        if frame_to_show is not None:
            # 픽킹 ROI (Blue)
            cv2.rectangle(frame_to_show, (ROI_X, ROI_Y), (ROI_X+ROI_W, ROI_Y+ROI_H), (255, 0, 0), 2)
            cv2.putText(frame_to_show, "Pick ROI", (ROI_X, ROI_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # 박스 ROI (Yellow)
            cv2.rectangle(frame_to_show, (BOX_ROI_X, BOX_ROI_Y), (BOX_ROI_X+BOX_ROI_W, BOX_ROI_Y+BOX_ROI_H), (0, 255, 255), 2)
            cv2.putText(frame_to_show, "Box ROI", (BOX_ROI_X, BOX_ROI_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if self.last_detected_box is not None and (time.time() - self.last_detect_time < 5.0):
                color = (0, 0, 255) if self.last_detected_quality == "DEFECT" else (0, 255, 0)
                cv2.drawContours(frame_to_show, [self.last_detected_box], 0, color, 3)
                
                # 시각화 라벨에 각도 추가
                label = f"{self.last_detected_quality} ({self.last_detected_score:.1f})"
                cv2.putText(frame_to_show, label, (self.last_detected_box[1][0], self.last_detected_box[1][1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Vision View", frame_to_show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False; rclpy.shutdown()

    def load_padim_model(self):
        self.get_logger().info("🧠 Loading PaDiM...")
        try:
            mean_vec_np = np.load(os.path.join(WEIGHTS_PATH, 'mean_vector.npy'))
            inv_cov_np = np.load(os.path.join(WEIGHTS_PATH, 'inv_cov_matrix.npy'))
            self.random_channels = np.load(os.path.join(WEIGHTS_PATH, 'random_channels.npy'))
            self.mean_vector = torch.from_numpy(mean_vec_np).to(self.DEVICE).float()
            self.inv_cov_matrix = torch.from_numpy(inv_cov_np).to(self.DEVICE).float()
            
            self.model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
            self.feature_maps = {}
            def hook_fn(module, input, output, name):
                if name == 'layer3': output = F.interpolate(output, size=(28, 28), mode='bilinear', align_corners=False)
                elif name == 'layer1': output = F.avg_pool2d(output, kernel_size=2)
                self.feature_maps[name] = output
            self.model.layer1.register_forward_hook(lambda m, i, o: hook_fn(m, i, o, 'layer1'))
            self.model.layer2.register_forward_hook(lambda m, i, o: hook_fn(m, i, o, 'layer2'))
            self.model.layer3.register_forward_hook(lambda m, i, o: hook_fn(m, i, o, 'layer3'))
            self.model.fc = torch.nn.Identity(); self.model.eval(); self.model.to(self.DEVICE)
        except Exception as e: self.model = None

    def extract_features(self, img_bgr):
        roi_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(roi_img_rgb)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.DEVICE)
        with torch.no_grad(): _ = self.model(input_tensor)
        combined = torch.cat([self.feature_maps['layer1'], self.feature_maps['layer2'], self.feature_maps['layer3']], dim=1)
        return combined.permute(0, 2, 3, 1).flatten(1, 2).squeeze(0)

    def detect_anomaly(self, img_bgr):
        if self.model is None: return "ERROR", 0.0
        features_reduced = self.extract_features(img_bgr)[:, self.random_channels]
        delta = features_reduced - self.mean_vector
        dist = torch.sqrt(torch.abs(torch.sum(torch.matmul(delta, self.inv_cov_matrix) * delta, dim=1)))
        score = torch.mean(torch.topk(dist, k=min(self.TOP_N_PATCHES, len(dist)))[0]).item()
        quality = "DEFECT" if score > self.ANOMALY_THRESHOLD else "GOOD"
        return quality, score

    def crop_rotated_rect(self, image, center, rotation_rad, target_size):
        # 1. 각도 정규화 (-45 ~ 45도 유지)
        angle_deg = math.degrees(rotation_rad)
        
        # 큐브는 90도 대칭이므로 회전각을 최소화
        angle_deg = angle_deg % 90
        if angle_deg > 45:
            angle_deg -= 90
        
        # 2. 회전 변환 행렬 구하기
        # 이미지 전체를 돌리면 느리니까, 중심점 기준으로 필요한 만큼만 돌림
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        
        # 3. 이미지 회전 (전체 이미지를 돌림 - 가장 확실한 방법)
        # 속도가 걱정되겠지만 1280 해상도에서 이 정도는 15FPS 방어 가능
        h, w = image.shape[:2]
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        
        # 4. 회전된 이미지에서 중심점 기준으로 자르기
        # 주의: 회전하면 좌표계가 바뀌지만, 중심점 기준 회전이라 center 좌표는 유지됨
        cx, cy = int(center[0]), int(center[1])
        half = target_size // 2
        
        start_y = cy - half
        end_y = cy + half
        start_x = cx - half
        end_x = cx + half
        
        # 이미지 범위 벗어나는지 체크
        if start_x < 0 or start_y < 0 or end_x > w or end_y > h:
            return None
            
        cropped = rotated[start_y:end_y, start_x:end_x]
        return cropped

    def pixel_to_robot(self, px, py, mm_per_pixel):
        dx_px = px - (ROI_W / 2)
        dy_px = py - (ROI_H / 2)
        dx_mm = dx_px * mm_per_pixel
        dy_mm = dy_px * mm_per_pixel
        robot_x = (dy_mm * -1.1) + CENTER_ROBOT_X
        robot_y = (dx_mm * -1.2) + CENTER_ROBOT_Y
        return robot_x, robot_y

    def save_defect_image_to_file(self, image_bgr):
        try:
            filename = f"defect_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(os.path.join(DEFECT_IMAGE_SAVE_DIR, filename), image_bgr)
        except: pass

    # ==========================================================================
    # 박스 상태 확인 핸들러 (서비스 콜백)
    # ==========================================================================
    def handle_box_check_request(self, request, response):
        self.get_logger().info("📦 Vision: Checking Box (Hybrid Mode: YOLO + HSV)...")
        
        raw = None
        with self.frame_lock:
            if self.latest_frame is not None:
                raw = self.latest_frame.copy()
        
        if raw is None:
            response.success = False; response.message = "NO_FRAME"; return response

        try:
            # ROI 추출 (고해상도 기준)
            roi_img = raw[BOX_ROI_Y : BOX_ROI_Y+BOX_ROI_H, BOX_ROI_X : BOX_ROI_X+BOX_ROI_W]
            if roi_img.size == 0:
                response.success = False; response.message = "ROI_ERROR"; return response

            # A. YOLO Check
            yolo_count = 0
            try:
                results = self.yolo_model(roi_img, imgsz=640, conf=0.25, verbose=False)
                for r in results:
                    if r.obb is not None:
                        yolo_count += len(r.obb)
            except Exception as e:
                self.get_logger().warn(f"YOLO Check Fail: {e}")

            # B. HSV Check (수정된 HSV 값 적용)
            hsv_count = 0
            hsv_area_total = 0
            try:
                hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, HSV_LOWER_GREEN, HSV_UPPER_GREEN)
                
                kernel = np.ones((5,5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    # 고해상도이므로 노이즈 임계값도 원래대로(1000) 유지
                    if area > 1000: 
                        hsv_count += 1
                        hsv_area_total += area
            except Exception as e:
                self.get_logger().warn(f"HSV Check Fail: {e}")

            cond_yolo = (yolo_count >= BOX_FULL_THRESHOLD)
            cond_hsv_count = (hsv_count >= BOX_FULL_THRESHOLD)
            # 면적 임계값도 고해상도 기준(12000)으로 유지
            cond_hsv_area = (hsv_area_total > 20000) 

            is_full = cond_yolo or cond_hsv_count or cond_hsv_area
            log_msg = f"📦 Result - YOLO:{yolo_count}, HSV_Cnt:{hsv_count}, Area:{int(hsv_area_total)}"
            self.get_logger().info(log_msg)

            if is_full:
                response.success = True; response.message = f"FULL ({log_msg})"
            else:
                response.success = False; response.message = f"NOT_FULL ({log_msg})"

        except Exception as e:
            self.get_logger().error(f"❌ Box Check Logic Error: {e}")
            response.success = False; response.message = "ERROR"
            
        return response

    # ==========================================================================
    # [핵심] 픽킹 서비스 핸들러 (YOLO 위치 + OpenCV 각도)
    # ==========================================================================
    def handle_detection_request(self, request, response):
        self.get_logger().info("▶️ [Genius Logic] Service Call Received")
        
        if not self.ai_lock.acquire(blocking=False):
            response.success = False; response.message = "BUSY"; return response

        try:
            raw = None
            with self.frame_lock:
                if self.latest_frame is not None and (time.time() - self.last_frame_time) < FRAME_TIMEOUT_SEC:
                    raw = self.latest_frame.copy()
            
            if raw is None:
                response.success = False; response.message = "FRAME_STALE"; return response

            # =========================================================
            # [단계 1] YOLO로 대략적인 위치 찾기 (필터링 역할)
            # =========================================================
            results = self.yolo_model(raw, imgsz=640, conf=0.4, verbose=False)
            
            found_yolo = False
            yolo_box = None # [cx, cy, w, h]
            
            for r in results:
                if r.obb is None: continue
                obb_data = r.obb.xywhr.cpu().numpy()
                for obb in obb_data:
                    cx, cy, w, h, rot = obb
                    # ROI 안에 중심이 들어오는지 확인
                    if (ROI_X <= cx <= ROI_X + ROI_W) and (ROI_Y <= cy <= ROI_Y + ROI_H):
                        yolo_box = [cx, cy, w, h]
                        found_yolo = True
                        break
                if found_yolo: break

            if not found_yolo:
                # YOLO가 못 찾으면 바로 종료
                response.success = True; response.message = "NO_OBJECT"
                response.quality = "NO_OBJECT"; response.center = [0.0, 0.0]; response.angle = 0.0
                return response

            # =========================================================
            # [단계 2] YOLO 박스 주변을 정밀 스캔 (Hybrid Refinement)
            # =========================================================
            cx, cy, w, h = yolo_box[0], yolo_box[1], yolo_box[2], yolo_box[3]
            
            # YOLO가 알려준 중심점 기준으로 넉넉하게(여유 20px) 자른다
            crop_size = int(max(w, h)) + 40 
            
            x1 = max(0, int(cx - crop_size//2))
            y1 = max(0, int(cy - crop_size//2))
            x2 = min(raw.shape[1], int(cx + crop_size//2))
            y2 = min(raw.shape[0], int(cy + crop_size//2))
            
            mini_crop = raw[y1:y2, x1:x2] # YOLO가 찍어준 곳만 뜯어냄
            
            # 여기서 HSV 적용 (전체 화면 아님! 딱 요만큼만!)
            hsv = cv2.cvtColor(mini_crop, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, HSV_LOWER_GREEN, HSV_UPPER_GREEN)
            
            # 노이즈 제거
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            final_center_x = cx
            final_center_y = cy
            final_angle = 0.0
            
            valid_contour = False
            
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 500: # 너무 작은 노이즈 제외
                    rect = cv2.minAreaRect(c)
                    valid_contour = True
                    
                    # 로컬 좌표(mini_crop 기준) -> 글로벌 좌표 변환
                    local_cx, local_cy = rect[0]
                    final_center_x = x1 + local_cx
                    final_center_y = y1 + local_cy
                    
                    # 각도 정규화
                    raw_angle = rect[2]
                    if raw_angle > 45: final_angle = raw_angle - 90
                    elif raw_angle < -45: final_angle = raw_angle + 90
                    else: final_angle = raw_angle
                    
                    # 보정값 적용 (캘리브레이션)
                    OFFSET_ANGLE = -10.0 
                    final_angle += OFFSET_ANGLE
                    
                    self.get_logger().info(f"📐 Refined Angle: {final_angle:.1f}")

            if not valid_contour:
                # HSV가 실패하면 YOLO 좌표를 믿음 (비상시)
                self.get_logger().warn("⚠️ HSV refinement failed, using raw YOLO coords")
                final_angle = math.degrees(yolo_box[4] if len(yolo_box)>4 else 0)

            # =========================================================
            # [단계 3] PaDiM 추론 및 좌표 변환
            # =========================================================
            
            # 1. 좌표 변환 (픽셀 -> 로봇)
            # 큐브 크기를 기준으로 mm_per_pixel 역산 (높이나 줌에 따라 변하므로 동적 계산이 좋음)
            # 하지만 YOLO w,h는 불안정하므로 고정값을 쓰거나 HSV rect 크기를 쓰는게 나음
            # 여기서는 HSV rect가 유효하면 그 크기를 사용
            if valid_contour:
                rect_w, rect_h = rect[1]
                obj_px = max(rect_w, rect_h)
            else:
                obj_px = max(w, h)
                
            mm_per_pixel = CUBE_REAL_SIZE_MM / obj_px if obj_px > 0 else 1.0
            
            roi_rel_x = final_center_x - ROI_X
            roi_rel_y = final_center_y - ROI_Y
            robot_x, robot_y = self.pixel_to_robot(roi_rel_x, roi_rel_y, mm_per_pixel)

            # 2. PaDiM용 이미지 자르기 (보정된 중심점과 각도 사용!)
            # ★★★ 여기가 제일 중요함 ★★★
            rotation_rad = math.radians(final_angle)
            center_tuple = (final_center_x, final_center_y)
            
            ai_input = self.crop_rotated_rect(raw, center_tuple, rotation_rad, FIXED_SIZE)
            
            quality = "ERROR"
            score = 0.0
            
            if ai_input is not None and ai_input.size > 0:
                quality, score = self.detect_anomaly(ai_input)
            
            # 3. 결과 시각화
            # 화면에 보여줄 박스 그리기
            if valid_contour:
                # minAreaRect 복원해서 그리기
                # (주의: rect는 로컬 좌표였으므로 글로벌로 다시 만들어야 함)
                global_rect = ((final_center_x, final_center_y), rect[1], rect[2])
                box = cv2.boxPoints(global_rect).astype(int)
                self.last_detected_box = box
            else:
                # YOLO 박스 그리기
                self.last_detected_box = None # YOLO OBB 그리기 복잡하니 생략 혹은 기존 유지
                
            self.last_detected_quality = quality
            self.last_detected_score = score
            self.last_detect_time = time.time()
            
            if quality == "DEFECT":
                self.save_defect_image_to_file(raw)
                self.get_logger().warn(f"🔴 DEFECT ({score:.1f})")
            else:
                self.get_logger().info(f"🟢 GOOD ({score:.1f})")

            response.success = True
            response.quality = quality
            response.message = f"{quality} (Score: {score:.1f})"
            response.center = [float(robot_x), float(robot_y)]
            response.angle = float(final_angle)

        except Exception as e:
            self.get_logger().error(f"❌ Logic Error: {e}")
            import traceback
            traceback.print_exc()
            response.success = False; response.quality = "ERROR"
        finally:
            self.ai_lock.release()
            
        return response

    def __del__(self):
        self.running = False 
        if hasattr(self, 'capture_thread'): self.capture_thread.join(timeout=1.0) 
        if hasattr(self, 'cap'): self.cap.release()
        cv2.destroyAllWindows() 

def main(args=None):
    rclpy.init(args=args); node = VisionNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
