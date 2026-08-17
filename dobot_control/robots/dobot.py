from typing import Dict
import numpy as np
import time
from dobot_function.manipulate_utils import load_ini_data_gripper
from dobot_function.function_utils import log_write
from threading import Event, Lock, Thread
from dobot_control.gripper.dobot_gripper import DobotGripper
from dobot_control.robots import dobot_api


class DobotRobot:
    """A class representing a UR robot."""

    def __init__(self, robot_ip: str = "192.168.5.1", no_gripper: bool = False):
        [print("in dobot robot") for _ in range(4)]
        self.robot_control = dobot_api.DobotApiMove(robot_ip, 30003)
        self.robot_inter = dobot_api.DobotApiDashboard(robot_ip, 29999)

        self.robot_ip = robot_ip
        self.robot_inter.EnableRobot()
        self.robot_inter.SpeedFactor(20)
        self.robot_inter.AccJ(20)
        self.robot_inter.SpeedJ(20)
        self.robot_inter.SetTool(1, 0, 0, 197, 0, 0, 0)
        self.robot_inter.Tool(1)  # 设置当前工具坐标系，使得Getpose()获取坐标时为此工具下

        self.com_list = {"192.168.5.1": "GRIPPER_LEFT", "192.168.5.2": "GRIPPER_RIGHT"}  # left, right
        _, gripper_dict = load_ini_data_gripper()
        self.gripper_list = gripper_dict[self.com_list[robot_ip]].pos
        self.gripper_id_name = gripper_dict[self.com_list[robot_ip]].id_name

        if not no_gripper:
            self.gripper = DobotGripper(port=gripper_dict[self.com_list[robot_ip]].port,
                                        id_name=self.gripper_id_name,
                                        servo_pos=self.gripper_list)
            print("gripper connected")
        self.robot_is_err = False
        self.robot_inter.StopDrag()
        self._use_gripper = not no_gripper
        self.robot_status = dobot_api.DobotApiStatus(robot_ip, 30004)
        self._stop_thread = Event()
        self._start_reading_thread()
        self._lock = Lock()


    def init_gripper(self):
        self.gripper.move(0, 100, 1)
        time.sleep(0.3)
        self.gripper.move(255, 100, 1)

    def _start_reading_thread(self):
        self._reading_thread = Thread(target=self.get_robot_err)
        self._reading_thread.daemon = True
        self._reading_thread.start()

    def get_robot_err(self):
        while not self._stop_thread.is_set():
            time.sleep(0.001)
            with self._lock:
                if self.robot_status.get_error():
                    self.robot_is_err = False

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        # print(self.robot_is_err)
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        robot_joints_angle = list(map(float, self.robot_inter.GetAngle().split("{")[1].split("}")[0].split(",")))
        robot_joints = [np.deg2rad(robot_joint) for robot_joint in robot_joints_angle]
        if self._use_gripper:
            gripper_pos = [1.0]
            pos = np.append(robot_joints, gripper_pos)
        else:
            gripper_pos = [1.0]
            pos = np.append(robot_joints, gripper_pos)
        return pos

    def get_XYZrxryrz_state(self) -> np.ndarray:
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        pos = np.array(list(map(float, self.robot_inter.GetPose().split("{")[1].split("}")[0].split(","))))  # 单位：度数
        return pos

    def command_joint_state(self, joint_state: np.ndarray):
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        robot_joints_angle = joint_state[:6]
        robot_joints = [np.rad2deg(robot_joint) for robot_joint in robot_joints_angle]
        self.robot_control.ServoJ(robot_joints[0],
                          robot_joints[1],
                          robot_joints[2],
                          robot_joints[3],
                          robot_joints[4],
                          robot_joints[5],
                          0.03)
        if self._use_gripper:
            gripper_pos = int(joint_state[-1] * 255)
            self.gripper.move(gripper_pos, 100, 1)

    def get_observations(self) -> Dict[str, np.ndarray]:
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        joints = self.get_joint_state()
        pos_quat = np.zeros(7)
        gripper_pos = np.array([joints[-1]])
        return {
            "joint_positions": joints,
            "joint_velocities": joints,
            "ee_pos_quat": pos_quat,
            "gripper_position": gripper_pos,
        }

    def set_do_status(self, which_do):
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        self.robot_inter.DO(which_do[0], which_do[1])
        return 1

    def disconnect(self):
        self.robot_inter.DisableRobot()
        self._stop_thread.set()
        self._reading_thread.join()
        self.robot_control.close()
        self.robot_inter.close()
        self.robot_status.close()
        self.gripper.disconnect()



def main():
    dobot1 = DobotRobot("192.168.5.1", no_gripper=False)
    # dobot2 = DobotRobot("192.168.5.2", no_gripper=False)
    # dobot1.init_gripper()
    # dobot2.init_gripper()
    # dobot2.set_light("red", 0)
    # while 1:
    #     aaa = dobot1.get_joint_state()
    #     print(2)
    #     bbb = dobot2.get_joint_state()
    #     # continue
    #     print(bbb)
    # dobot = DobotRobot("192.168.5.2", no_gripper=False)
    # set_light(dobot, "red", 0)


if __name__ == "__main__":
    main()
