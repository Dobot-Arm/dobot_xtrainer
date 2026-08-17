import numpy as np
import time
from dobot_function.function_utils import log_write


class BimanualRobot:
    def __init__(self, robot_l, robot_r):
        self._robot_l = robot_l
        self._robot_r = robot_r

        self.err_detect()
        self.is_connect = True

    def err_detect(self):
        assert not self._robot_l.robot_is_err, "left robot error!"
        assert not self._robot_r.robot_is_err, "right robot error!"

    def init_gripper(self):
        self.err_detect()
        self._robot_l.init_gripper()
        self._robot_r.init_gripper()

    def get_joint_state(self):
        self.err_detect()
        return np.concatenate(
            (self._robot_l.get_joint_state(), self._robot_r.get_joint_state())
        )

    def command_joint_state(self, joint_state: np.ndarray, flag_in):
        self.err_detect()
        if flag_in[0]:
            self._robot_l.command_joint_state(joint_state[: 7])
        if flag_in[1]:
            self._robot_r.command_joint_state(joint_state[7 :])
        return 1

    def get_observations(self):
        self.err_detect()
        l_obs = self._robot_l.get_observations()
        r_obs = self._robot_r.get_observations()
        assert l_obs.keys() == r_obs.keys()
        return_obs = {}
        for k in l_obs.keys():
            try:
                return_obs[k] = np.concatenate((l_obs[k], r_obs[k]))
            except Exception as e:
                print(e)
                print(k)
                print(l_obs[k])
                print(r_obs[k])
                raise RuntimeError()

        return return_obs

    def set_light(self, light_color, light_status):
        print("set light")
        self.err_detect()
        for i in range(1, 4):
            self._robot_l.set_do_status([i, 0])
        if light_color == "red":
            self._robot_l.set_do_status([1, light_status])
        elif light_color == "yellow":
            self._robot_l.set_do_status([2, light_status])
        elif light_color == "green":
            self._robot_l.set_do_status([3, light_status])

    def get_XYZrxryrz_state(self) -> np.ndarray:
        self.err_detect()
        return np.concatenate(
            (self._robot_l.get_XYZrxryrz_state(), self._robot_r.get_XYZrxryrz_state())
        )

    def gripper_control(self, gripper_status):
        if gripper_status == "open":
            self._robot_l.gripper.move(255, 100, 1)
            self._robot_r.gripper.move(255, 100, 1)
        elif gripper_status == "close":
            self._robot_l.gripper.move(0, 100, 1)
            self._robot_r.gripper.move(0, 100, 1)

    def disconnect(self):
        try:
            self._robot_l.disconnect()
        except Exception as e:
            print(e)
        try:
            self._robot_r.disconnect()
        except Exception as e:
            print(e)
        self.is_connect = False

    def get_error(self):
        if self._robot_l.robot_is_err or self._robot_r.robot_is_err:
            return True
        else:
            return False

def main():
    pass


if __name__ == "__main__":
    main()
