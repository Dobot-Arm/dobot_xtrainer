import numpy as np
import time
import configparser
from dobot_control.agents.dobot_agent import DobotRobotConfig
import os
from pathlib import Path
from dataclasses import dataclass
from dobot_function.function_utils import wait_period, log_write
from dobot_control.dynamixel.driver import DynamixelDriver
from dobot_function.function_utils import (scan_port, log_write, mk_dir, save_frame, wait_period, time_print,
                                           get_firmware_version_satisfied, get_robot_type)

def load_ini_data_camera():
    camera_dict = {"top": None, "left": None, "right": None}
    ini_file_path = str(Path(__file__).parent.parent) + "/settings/dobot_config/dobot_settings.ini"
    ini_file = configparser.ConfigParser()
    ini_file.read(ini_file_path)
    for _cam in camera_dict.keys():
        camera_dict[_cam] = ini_file.get("CAMERA", _cam)
    return ini_file, camera_dict


def load_ini_data_robot():
    robot_dict = {"ROBOT_LEFT": None, "ROBOT_RIGHT": None}
    ini_file_path = str(Path(__file__).parent.parent) + "/settings/dobot_config/dobot_settings.ini"
    ini_file = configparser.ConfigParser()
    ini_file.read(ini_file_path)
    for _cam in robot_dict.keys():
        robot_dict[_cam] = ini_file.get(_cam, "ip")
    return ini_file, robot_dict


def load_ini_data_hands():
    ini_file_path = str(Path(__file__).parent.parent) + "/settings/dobot_config/dobot_settings.ini"
    ini_file = configparser.ConfigParser()
    ini_file.read(ini_file_path)

    hands_dict = {"HAND_LEFT": None, "HAND_RIGHT": None}
    for _hand in hands_dict.keys():
        hands_dict[_hand] = DobotRobotConfig(
            joint_ids=[int(i) for i in ini_file.get(_hand, "joint_ids").split(",")],
            append_id=int(ini_file.get(_hand, "append_id")),
            port=ini_file.get(_hand, "port"),
            joint_offsets=[float(i) for i in ini_file.get(_hand, "joint_offsets").split(",")],
            joint_signs=[int(i) for i in ini_file.get(_hand, "joint_signs").split(",")],
            gripper_config=[int(i) for i in ini_file.get(_hand, "gripper_config").split(",")],
            start_joints=[float(i) for i in ini_file.get(_hand, "start_joints").split(",")],
            baud_rate= int(ini_file.get(_hand, "baud_rate"))
        )
    return ini_file, hands_dict


@dataclass
class GripperConfig:
    id_name: int
    pos: tuple
    port: str


def load_ini_data_gripper():
    ini_file_path = str(Path(__file__).parent.parent) + "/settings/dobot_config/dobot_settings.ini"
    ini_file = configparser.ConfigParser()
    ini_file.read(ini_file_path)

    gripper_dict = {"GRIPPER_LEFT": None, "GRIPPER_RIGHT": None}
    for _gripper in gripper_dict.keys():
        gripper_dict[_gripper] = GripperConfig(id_name=int(ini_file.get(_gripper, "id")),
                                               pos=list([int(i) for i in ini_file.get(_gripper, "pos").split(",")]),
                                               port=ini_file.get(_gripper, "port"))
    return ini_file, gripper_dict


# robot init move
def robot_pose_init(env):
    # go to the first point
    reset_joints_left = np.deg2rad([-90, 30, -110, 20, 90, 90, 1])  #
    reset_joints_right = np.deg2rad([90, -30, 110, -20, -90, -90, 1])
    reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
    curr_joints = env.get_obs()["joint_positions"]
    max_delta = (np.abs(curr_joints - reset_joints)).max()
    steps = min(int(max_delta / 0.01), 100)
    for jnt in np.linspace(curr_joints, reset_joints, steps):
        env.step(jnt, [1, 1])

    # go to the second point
    reset_joints_left = np.deg2rad([-90, 0, -90, 0, 90, 90, 1])  #
    reset_joints_right = np.deg2rad([90, 0, 90, 0, -90, -90, 1])
    reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
    curr_joints = env.get_obs()["joint_positions"]
    max_delta = (np.abs(curr_joints - reset_joints)).max()
    steps = min(int(max_delta / 0.01), 100)
    for jnt in np.linspace(curr_joints, reset_joints, steps):
        env.step(jnt, [1, 1])


# main hand pose dev check
def obs_action_check(env, agent):
    obs = env.get_obs()
    joints = obs["joint_positions"]
    action = agent.act(obs)
    if (action - joints > 0.6).any():
        print("Action is too big")
        # print which joints are too big
        joint_index = np.where(action - joints > 0.5)
        for j in joint_index:
            print(
                f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
            )
        return 0, 0
    else:
        return 1, action


# nova2 dev joint check
def servo_action_check(action, last_action, flag_in, step_len=0.9):
    ind_list = []
    if flag_in[0] and not flag_in[1]:
        ind_list = [i for i in range(6)]
    elif not flag_in[0] and flag_in[1]:
        ind_list = [i+7 for i in range(6)]
    elif flag_in[0] and flag_in[1]:
        ind_list = [i for i in range(14)]
    assert len(ind_list), "err in servo_action_check"
    if (np.abs(action - last_action) > step_len).any():
        joint_index = np.where(np.abs(action - last_action) > step_len)
        print("action: ", action)
        print("last_action: ", last_action)
        for j in joint_index[0]:
            if j != 6 and j != 13 and (j in ind_list):
                pi_2_cal = (action[j] - last_action[j])/np.pi
                if abs(pi_2_cal) > 1.85 and abs(pi_2_cal) < 2.15:
                    action[j] = action[j] - 2 * np.pi * (pi_2_cal / abs(pi_2_cal))
                else:
                    print("Servo action dev is too big")
                    print(
                        f"Joint [{j}], leader: {action[j]}, follower: {last_action[j]}, "
                        f"diff: {(action[j] - last_action[j])/np.pi}")
                    return 0, action
    return 1, action


# pose check between main hand and the follower
def pose_check(env, agent, flag_in):
    start_pos = agent.act({})
    joints = env.get_joint_state()
    err_pose_check, action_return = servo_action_check(start_pos, joints, flag_in, 0.6)
    if err_pose_check:
        return 1, action_return
    else:
        return 0, action_return


def dynamic_approach(env, agent, flag_in):
    err1, action1 = pose_check(env, agent, flag_in)
    assert err1 != 0, env.set_light("red", 1)
    joints = env.get_joint_state()
    # log_write(__file__, "joints: " + str(joints))
    # log_write(__file__, "action1: " + str(action1))
    joints[6] = action1[6]
    joints[13] = action1[13]
    if flag_in[0] and not flag_in[1]:
        abs_deltas = max(np.abs(action1[:6] - joints[:6]))
    elif not flag_in[0] and flag_in[1]:
        abs_deltas = max(np.abs(action1[7:13] - joints[7:13]))
    else:
        abs_deltas = max(np.abs(action1 - joints))
    # log_write(__file__,  "abs_deltas: " + str(abs_deltas))
    steps = int(abs_deltas / 0.01)
    # log_write(__file__, "steps: " + str(steps))

    for jnt in np.linspace(joints, action1, steps):
        # print(jnt)
        env.command_joint_state(jnt, flag_in)
        tic = time.time()
        wait_period(50, tic)

        # log_write(__file__, "flag_in: " + str(flag_in))
        # log_write(__file__, "jnt: " + str(jnt))
    # time.sleep(0.05)
    return action1


def dh_transformation_matrix(theta, d, a, alpha):
    """
    Create the DH transformation matrix
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    return np.array([
        [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
        [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
        [0, sin_alpha, cos_alpha, d],
        [0, 0, 0, 1]
    ])

def claw_width(coef):
    """
    Calculate the claw width
    """
    claw_servo = 2.3818 - coef * 1.5401
    cos_claw_servo = np.cos(claw_servo)
    claw_wid = 0.03 * cos_claw_servo + 0.5 * np.sqrt(0.0036 * cos_claw_servo ** 2 + 0.0028)
    return claw_wid

def forward_kinematics(q0, q1, q2, q3, q4, q5, y, which_robot_type):
    """
    Compute the forward kinematics
    """
    if which_robot_type == "Nova 2":
        dh_params = [
            (q0, 0.2234, 0, np.pi / 2),
            (q1 - np.pi / 2, 0, -0.280, 0),
            (q2, 0, -0.225, 0),
            (q3 - np.pi / 2, 0.1175, 0, np.pi / 2),
            (q4, 0.120, 0, -np.pi / 2),
            (q5, 0.088, 0, 0)
        ]
    if which_robot_type == "Nova 5":
        dh_params = [
            (q0, 0.240, 0, np.pi / 2),
            (q1 - np.pi / 2, 0, -0.400, 0),
            (q2, 0, -0.330, 0),
            (q3 - np.pi / 2, 0.135, 0, np.pi / 2),
            (q4, 0.120, 0, -np.pi / 2),
            (q5, 0.088, 0, 0)
        ]
    t = np.eye(4)
    for params in dh_params:
        t = np.dot(t, dh_transformation_matrix(*params))
    t_tool = np.eye(4)
    t_tool[:3, 3] = np.array([0, y, 0.2])
    t_final = np.dot(t, t_tool)
    pos = t_final[:3, 3]
    return pos


def calculate_vel_pos(action, last_action, total_time, which_robot_t):
    """
    Calculate the velocity for forward kinematics
    """
    claw_left = claw_width(action[6])
    claw_right = claw_width(action[13])

    positions = {}
    vel = {}

    for side in ['left', 'right']:
        for paw in ['left', 'right']:
            coef = 1 if paw == 'left' else -1
            claw = claw_left if side == 'left' else claw_right
            claw *= coef

            current_fk = forward_kinematics(*action[0:6] if side == 'left' else action[7:13], claw, which_robot_t)
            last_fk = forward_kinematics(*last_action[0:6] if side == 'left' else last_action[7:13],claw, which_robot_t)

            positions[f'{side}_{paw}'] = current_fk
            vel[f'{side}_{paw}'] = (current_fk - last_fk) / total_time

    return positions, vel

# Check that the positions is within a safe zone
def is_within_safe_position(position, x_range, y_range, z_min):
    return x_range[0] <= position[0] <= x_range[1] and \
           y_range[0] <= position[1] <= y_range[1] and \
           position[2] > z_min


def check_pose_protection(positions, vel, what_to_do, x_range_left, x_range_right, y_range, z_range_left, z_range_right):
    protect_err = False
    warnings = []

    delta_left_left = vel['left_left']
    delta_left_right = vel['left_right']
    delta_right_left = vel['right_left']
    delta_right_right = vel['right_right']

    positions_mm = {key: value * 1000 for key, value in positions.items()}
    # Define a safe zone
    # left arm (jaw tip position) limit:  290>x>-450  -750<Y<-160  z>44;
    # right arm (jaw tip position) limit:  450>x>-290  -750<Y<-160  z>42;
    # x_range_left = (-450, 290)
    # x_range_right = (-290, 450)
    # y_range = (-750, -160)
    # z_range_left = 44
    # z_range_right = 42

    if what_to_do[0, 1]:  # The left hand is in sync
        # Z direction speed limit -1 m/s
        if delta_left_left[2] < -1 or delta_left_right[2] < -1:
            warnings.append("[Warn]:The left robot speed of the TCP is moving too fast!")
            warnings.append(f"delta_left_left: {delta_left_left[2]}")
            protect_err = True
        # Left arm working space limitation
        positions_to_check = ['left_left', 'left_right']
        x_ranges = [x_range_left, x_range_left]
        z_ranges = [z_range_left, z_range_left]
        if not all(is_within_safe_position(positions_mm[pos], x_range, y_range, z_range)
                   for pos, x_range, z_range in zip(positions_to_check, x_ranges, z_ranges)):
            warnings.append("[Warn]:The left arm is out of the safe zone!")
            protect_err = True

    if what_to_do[1, 1]:  # The right hand is in sync
        # Z direction speed limit -1 m/s
        if delta_right_left[2] < -1 or delta_right_right[2] < -1:
            warnings.append("[Warn]:The right robot speed of the TCP is moving too fast!")
            warnings.append(f"delta_right_left: {delta_right_left[2]}")
            protect_err = True
        # Right arm working space limitation
        positions_to_check = ['right_left', 'right_right']
        x_ranges = [x_range_right, x_range_right]
        z_ranges = [z_range_right, z_range_right]
        if not all(is_within_safe_position(positions_mm[pos], x_range, y_range, z_range)
                   for pos, x_range, z_range in zip(positions_to_check, x_ranges, z_ranges)):
            warnings.append("[Warn]:The right arm is out of the safe zone!")
            protect_err = True

    for warning in warnings:
        print(warning)

    return protect_err

def check_joint_safety(action):
    protect_err = False
    if not (action[2] < 0):
        print("[Warn]:The J3 joints of the robotic arm are out of the safe position! ")
        print(action)
        protect_err = True
    if not (action[9] > 0):
        print("[Warn]:The J3 joints of the robotic arm are out of the safe position! ")
        print(action)
        protect_err = True
    return protect_err


def get_config(which_hand, which_hand_config, pos_joint, grip_pos):
    gripper_ids = {"HAND_LEFT": [8], "HAND_RIGHT": [18]}
    # driver.set_torque_mode(False)
    print("--------------------", which_hand, "-------------------")
    curr_joints = pos_joint[:6]
    print("curr_joints: ", curr_joints)
    print("robot_joints: ", which_hand_config.start_joints)
    print("offsets", which_hand_config.joint_offsets)
    print([
        float("%.2f" % (curr_joints[i] * which_hand_config.joint_signs[i]+
                        which_hand_config.joint_offsets[i]))
        for i in range(6)])

    dev_pos = [
        float("%.2f" % (curr_joints[i] * which_hand_config.joint_signs[i] -
                        which_hand_config.start_joints[i] * which_hand_config.joint_signs[i] +
                        which_hand_config.joint_offsets[i]))
        for i in range(6)]
    print("dev(write): ", dev_pos)
    print("dev(*pi/2): ", [(i / np.pi) * 2 for i in dev_pos])
    print("dev(angle): ", [np.rad2deg(i) for i in dev_pos])
    print("----------------------------------------------")
    print(grip_pos)
    gripper_on = int(np.rad2deg(grip_pos) - 0.2)
    gripper_close = int(np.rad2deg(grip_pos) + 30)
    print(
        "gripper open (degrees)       ",
        gripper_on,
    )
    print(
        "gripper close (degrees)      ",
        gripper_close,
    )
    return dev_pos, [gripper_ids[which_hand][0], gripper_close, gripper_on]


if __name__ == "__main__":
    a, b = load_ini_data_camera()
    print(a, b)
    # action = [-1.44164974,  0.13345643, -2.07741816,  0.59677646, 1.60714534,  1.91935946,
    #            0.99817288,  0.95174802,  1.03044725,  1.90838818, -0.36576736, -1.41200051, -2.05291206,  1.]
    #
    # print(action[7:13])
    # print(np.rad2deg(0.6))
