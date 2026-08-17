import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from dobot_control.agents.agent import Agent
from dobot_control.robots.dynamixel import DynamixelRobot
import time
import configparser
import os
from dobot_control.agents.agent import BimanualAgent


@dataclass
class DobotRobotConfig:
    joint_ids: Sequence[int]
    append_id: int
    port: str
    joint_offsets: Sequence[float]
    joint_signs: Sequence[int]
    gripper_config: Tuple[int, int, int]
    start_joints: Sequence[float]
    baud_rate: int

    def __post_init__(self):
        assert len(self.joint_ids) == len(self.joint_offsets)
        assert len(self.joint_ids) == len(self.joint_signs)

    def make_robot(self, start_joints: Optional[np.ndarray] = None, using_sensor: bool = False) -> DynamixelRobot:
        return DynamixelRobot(
            joint_ids=self.joint_ids,
            append_id=self.append_id,
            joint_offsets=list(self.joint_offsets),
            real=True,
            joint_signs=list(self.joint_signs),
            port=self.port,
            gripper_config=self.gripper_config,
            start_joints=start_joints,
            using_sensor=using_sensor,
            baudrate=self.baud_rate
        )

class DobotAgent(Agent):
    def __init__(
        self,
        using_sensor: bool,
        which_hand: str,
        dobot_config: Optional[DobotRobotConfig] = None,
        start_joints: Optional[np.ndarray] = None,
    ):
        self.which_hand = which_hand
        self.using_sensor = using_sensor
        self.torque_enable = True
        assert dobot_config
        self._robot = dobot_config.make_robot(start_joints=start_joints, using_sensor=self.using_sensor)

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        return self._robot.get_joint_state()

    def get_gripper_pos(self):
        return self._robot.get_gripper_pos()

    def set_torque(self, _flag = False):
        self._robot.set_torque_mode(_flag)
        self.torque_enable = _flag

    def get_keys(self):
        return self._robot.get_key_status()

    def get_err(self):
        return self._robot.get_err()

    def close(self):
        self._robot.close()


def main() -> None:
    pass


if __name__ == "__main__":
    from dobot_function.manipulate_utils import load_ini_data_hands
    _, hands_dict = load_ini_data_hands()
    print(hands_dict)
    left_agent = DobotAgent(which_hand="LEFT", dobot_config=hands_dict["HAND_LEFT"])
    right_agent = DobotAgent(which_hand="RIGHT", dobot_config=hands_dict["HAND_RIGHT"])
    print(222)
    # bi_agent = BimanualAgent(left_agent, right_agent)

    while 1:
        tic = time.time()
        print(left_agent.get_keys())
        # print(bi_agent.get_keys(), bi_agent.act({}))
        toc = time.time()
        print(toc-tic)