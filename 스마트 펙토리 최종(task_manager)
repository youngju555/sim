#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from my_robot_interfaces.srv import ArmCommand, DetectItem 
from std_srvs.srv import Trigger, SetBool
from std_msgs.msg import Int32
import threading
import sys
import time

class TaskManagerNode(Node):
    def __init__(self):
        super().__init__('task_manager_node')
        
        self.cb_group = ReentrantCallbackGroup()

        # [Servers]
        self.start_srv = self.create_service(
            Trigger,
            '/system/start_work',
            self.handle_system_start,
            callback_group=self.cb_group
        )
        self.trigger_srv = self.create_service(
            Trigger, 
            '/robot_arm/detect',
            self.handle_controller_trigger,
            callback_group=self.cb_group
        )

        # [Clients]
        self.vision_client = self.create_client(
            DetectItem, 
            '/vision/detect_item',
            callback_group=self.cb_group
        )
        self.vision_box_client = self.create_client(
            Trigger, 
            '/vision/check_box_full', 
            callback_group=self.cb_group
        )
        self.arm_client = self.create_client(
            ArmCommand, 
            '/arm/execute_cmd', 
            callback_group=self.cb_group
        )
        self.agv_client = self.create_client(
            SetBool, 
            '/agv/request_dispatch', 
            callback_group=self.cb_group
        )
        
        # [NEW] 안전 해제용 클라이언트
        self.safety_reset_client = self.create_client(
            Trigger, 
            '/arm/safety_reset', 
            callback_group=self.cb_group
        )

        self.count_pub = self.create_publisher(Int32, '/robot/work_cnt', 10)
        
        self.is_arm_moving = False
        self.is_system_active = False 
        self.is_waiting_agv = False 
        self.total_count = 0 
        
        self.get_logger().info('✅ Task Manager Ready. (Mode: Vision Box Check)')
        
        self.input_thread = threading.Thread(target=self._user_input_loop, daemon=True)
        self.input_thread.start()

    def _user_input_loop(self):
        print("\n" + "="*40)
        print(" [TEST MODE COMMANDS]")
        print("  - 's' + 엔터   : 시스템 시작 (Start & Check Box)")
        print("  - '1234' + 엔터: 🔓 안전 펜스 잠금 해제 (일시정지 풀기)")
        print("  - 그냥 엔터    : 작업 트리거 (PLC 신호 시뮬레이션)")
        print("  - 'q' + 엔터   : 종료")
        print("="*40 + "\n")

        while rclpy.ok():
            try:
                cmd = input()
                if cmd == 's':
                    self.get_logger().info("⌨️ User Input: SYSTEM START")
                    self.start_system_logic()
                elif cmd == 'q':
                    self.get_logger().info("👋 Shutting down...")
                    rclpy.shutdown()
                    sys.exit(0)
                
                # [NEW] 비밀번호 입력 처리
                elif cmd == '1234':
                    self.get_logger().info("⌨️ User Input: SAFETY UNLOCK CODE '1234'")
                    self.call_safety_reset()

                else:
                    self.get_logger().info("⌨️ User Input: TRIGGER RECEIVED")
                    self._execute_task_logic(source="KEYBOARD")
            except Exception as e:
                print(f"Input Error: {e}")

    # [NEW] 안전 해제 서비스 호출 함수
    def call_safety_reset(self):
        if not self.safety_reset_client.wait_for_service(1.0):
            self.get_logger().error("❌ Safety Reset Service Unavailable")
            return
        
        req = Trigger.Request()
        future = self.safety_reset_client.call_async(req)
        future.add_done_callback(self.safety_reset_done_callback)

    def safety_reset_done_callback(self, future):
        try:
            res = future.result()
            if res.success:
                self.get_logger().info(f"🔓 Safety Reset SUCCESS: {res.message}")
            else:
                self.get_logger().error(f"🚫 Safety Reset FAILED: {res.message}")
        except Exception as e:
            self.get_logger().error(f"❌ Safety Reset Error: {e}")

    def handle_system_start(self, request, response):
        self.get_logger().info("📢 Cmd from Ros Controller: SYSTEM START")
        self.start_system_logic()
        
        response.success = True
        msg = "Ready." if not self.is_waiting_agv else "Started but Waiting for AGV."
        response.message = msg
        return response

    def handle_controller_trigger(self, request, response):
        self.get_logger().info("📢 Cmd from Ros Controller: EXECUTE TASK")
        
        if self.is_arm_moving or self.is_waiting_agv:
            self.get_logger().warn(f"⚠️ Ignored: Robot is BUSY (Moving: {self.is_arm_moving}, AGV: {self.is_waiting_agv})")
            response.success = False
            response.message = "BUSY"
            return response
        
        result_msg = self._execute_task_logic(source="SERVICE")
        
        if result_msg in ["NOT_ACTIVE", "WAIT_VISION", "VISION_FAIL", "PAUSED_FOR_AGV", "NO_OBJECT"]:
            response.success = False
        else:
            response.success = True
            
        response.message = result_msg
        self.get_logger().info(f"📤 Sending Response: Success={response.success}, Msg={response.message}")
        return response

    def start_system_logic(self):
        # 1. 홈 이동
        self.send_arm_command("home", [0.0, 0.0, 0.0])
        self.is_system_active = True
        
        # 2. 시작하자마자 박스 상태 확인
        self.get_logger().info("🔍 System Start: Checking Box Status...")
        self.check_box_and_act()

    def _execute_task_logic(self, source):
        if not self.is_system_active:
            self.get_logger().warn(f"⚠️ [{source}] Ignored: System is NOT ACTIVE.")
            return "NOT_ACTIVE"

        if self.is_waiting_agv:
            self.get_logger().warn(f"⏳ [{source}] Ignored: Waiting for AGV... (Box is Full)")
            return "PAUSED_FOR_AGV"

        # Vision Service 호출 (Blocking)
        vision_resp = self.call_vision_service()

        if vision_resp is None:
            self.get_logger().error(f"❌ [{source}] Vision Service Failed.")
            return "VISION_FAIL"
        
        if vision_resp.quality == "NO_OBJECT":
            self.get_logger().info(f"⚪ [{source}] No Object Detected.")
            return "NO_OBJECT"

        # 작업 수행
        quality = vision_resp.quality
        
        if quality == "GOOD":
            self.get_logger().info(f"🟢 [{source}] Action: Pick Item (Good)")
            self.send_arm_command("pick_good", [vision_resp.center[0], vision_resp.center[1], vision_resp.angle])
        else:
            self.get_logger().info(f"🔴 [{source}] Action: Discard Item (Bad)")
            self.send_arm_command("discard_bad", [0.0, 0.0, 0.0])
            
        return quality

    def call_vision_service(self):
        # 1. 서비스 서버가 켜져 있는지 확인
        if not self.vision_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("❌ Vision service is not available.")
            return None

        req = DetectItem.Request()
        
        # 2. [수정] spin을 삭제하고 동기 호출(call) 사용
        # MultiThreadedExecutor 덕분에 여기서 기다려도 다른 통신이 막히지 않습니다.
        try:
            response = self.vision_client.call(req) 
            return response
        except Exception as e:
            self.get_logger().error(f"❌ Vision Call Failed: {e}")
            return None

    def send_arm_command(self, cmd, coord):
        if not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("❌ Arm service not available!")
            return

        self.is_arm_moving = True

        req = ArmCommand.Request()
        req.command = cmd
        req.target_coord = coord
        
        future = self.arm_client.call_async(req)
        future.add_done_callback(self.arm_done_callback)

    def arm_done_callback(self, future):
        try:
            result = future.result()
            self.get_logger().info(f"🤖 Arm Status: {result.message}")
            
            self.is_arm_moving = False

            self.total_count += 1
            msg = Int32()
            msg.data = self.total_count
            self.count_pub.publish(msg)

            self.check_box_and_act()
            
        except Exception as e:
            self.get_logger().error(f"❌ Callback Error: {e}")
            self.is_arm_moving = False 


    def check_box_and_act(self):
        if not self.vision_box_client.wait_for_service(1.0):
            self.get_logger().error("❌ Vision Box Service Unavailable")
            return

        req = Trigger.Request()
        future = self.vision_box_client.call_async(req)
        future.add_done_callback(self.box_check_done_callback)

    def box_check_done_callback(self, future):
        try:
            res = future.result()
            
            if res.success: 
                self.get_logger().warn(f"🛑 Vision Confirmed: BOX IS FULL! ({res.message}) Calling AGV...")
                self.is_waiting_agv = True
                self.control_agv(enable=True)
            else:
                self.get_logger().info(f"✅ Box Not Full ({res.message}). System Ready.")
                
        except Exception as e:
            self.get_logger().error(f"❌ Box Check Error: {e}")

    def control_agv(self, enable: bool):
        if not self.agv_client.wait_for_service(1.0):
            self.get_logger().error("❌ AGV Service Unavailable")
            return

        req = SetBool.Request()
        req.data = enable
        
        action_str = "CALL" if enable else "CANCEL"
        self.get_logger().info(f"🚚 Sending AGV Command: {action_str}...")
        
        future = self.agv_client.call_async(req)
        future.add_done_callback(lambda f: self.agv_done_callback(f, action_str))

    def agv_done_callback(self, future, action_str):
        try:
            res = future.result()
            
            if res.success and action_str == "CALL":
                self.get_logger().info(f"✅ AGV Process Complete (Box Replaced). Resuming...")
                self.is_waiting_agv = False 
                
                self.send_arm_command("home", [0.0, 0.0, 0.0])

            elif not res.success:
                self.get_logger().error(f"⚠️ AGV Failed: {res.message}. Robot still paused.")
                
        except Exception as e:
            self.get_logger().error(f"❌ AGV Service Call Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = TaskManagerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
