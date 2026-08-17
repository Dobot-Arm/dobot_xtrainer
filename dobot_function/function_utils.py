import os.path
import datetime
import time
from http.client import responses

import cv2
import numpy as np
import serial.tools.list_ports as serial_stl
import subprocess
from pathlib import Path
import configparser
import pickle
from typing import Dict
import glob
import h5py
import requests


def deal_data(pos_list, top_list, left_list, right_list):
    if len(pos_list) < len(top_list):
        for i in range(len(top_list)):
            file_name = top_list[i].split("/")[-1].split(".")[0] + ".pkl"
            if not os.path.exists(os.path.dirname(pos_list[0])+f"/{file_name}"):
                print(top_list[i])
                os.remove(top_list[i])
                os.remove(left_list[i])
                os.remove(right_list[i])
                top_list.remove(top_list[i])
                left_list.remove(left_list[i])
                right_list.remove(right_list[i])
    elif len(pos_list) > len(top_list):
        for i in range(len(pos_list)):
            # file_name = pos_list[i].split("/")[-1].split(".")[0] + ".npy"
            file_name = pos_list[i].split("/")[-1].split(".")[0] + ".jpg"
            if not os.path.exists(os.path.dirname(pos_list[0])+f"/{file_name}"):
                print(pos_list[i])
                os.remove(pos_list[i])
                pos_list.remove(pos_list[i])
    return pos_list, top_list, left_list, right_list


def data_read_from_collect(one_dataset_dir):
    camera_names = ['top', 'left_wrist', 'right_wrist']
    print(camera_names)

    data_pose_list = glob.glob(one_dataset_dir + 'observation/*.pkl')
    print(data_pose_list)
    # images_top_list = glob.glob(one_dataset_dir + 'topImg/*.npy')
    # images_left_list = glob.glob(one_dataset_dir + 'leftImg/*.npy')
    # images_right_list = glob.glob(one_dataset_dir + 'rightImg/*.npy')
    images_top_list = glob.glob(one_dataset_dir + 'topImg/*.jpg')
    images_left_list = glob.glob(one_dataset_dir + 'leftImg/*.jpg')
    images_right_list = glob.glob(one_dataset_dir + 'rightImg/*.jpg')
    data_pose_list.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    images_top_list.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    images_left_list.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    images_right_list.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    print(images_right_list)

    data_pose_list, images_top_list, images_left_list, images_right_list = (
        deal_data(data_pose_list, images_top_list, images_left_list, images_right_list))

    is_sim = False
    qpos = []
    qvel = []
    action = []
    base_action = None
    image_dict = dict()
    image_li = [[], [], []]
    for cam_name in camera_names:
        image_dict[f'{cam_name}'] = []
    for i in range(len(data_pose_list)):
        with open(data_pose_list[i], "rb") as f:
            data_single = pickle.load(f)
            qpos.append(data_single['joint_positions'])
            qvel.append(data_single['joint_velocities'])
            action.append(data_single['control'])
            # image_top = cv2.imdecode(np.asarray(np.load(images_top_list[i]), dtype="uint8"), cv2.IMREAD_COLOR)
            # image_left = cv2.imdecode(np.asarray(np.load(images_left_list[i]), dtype="uint8"), cv2.IMREAD_COLOR)
            # image_right = cv2.imdecode(np.asarray(np.load(images_right_list[i]), dtype="uint8"), cv2.IMREAD_COLOR)
            image_top = cv2.imread(images_top_list[i])
            image_left = cv2.imread(images_left_list[i])
            image_right = cv2.imread(images_right_list[i])
            # cv2.imshow("0", image_right)
            # cv2.waitKey(1)
            image_li[0].append(image_top[:, :, ::-1])
            image_li[1].append(image_left[:, :, ::-1])
            image_li[2].append(image_right[:, :, ::-1])
    image_dict['top'] = np.array(image_li[0])
    image_dict['left_wrist'] = np.array(image_li[1])
    image_dict['right_wrist'] = np.array(image_li[2])
    return np.array(qpos), np.array(qvel), np.array(action), base_action, image_dict, is_sim

def save_frame(
    folder: str,
    timestamp: int,
    obs: Dict[str, np.ndarray],
    action: np.ndarray,
) -> None:
    obs["control"] = action  # add action to obs

    # make folder if it doesn't exist
    # folder.mkdir(exist_ok=True, parents=True)
    recorded_file = folder + str(timestamp) + ".pkl"
    print(recorded_file)

    with open(recorded_file, "wb") as f:
        pickle.dump(obs, f)


def time_print(str_):
    current_time = time.strftime("%Y-%m-%d %H-%M-%S:", time.localtime())
    print(str(current_time) + str(datetime.datetime.now().strftime("%f")[:-3]), str_)


def free_limit_and_set_one(file_name):
    ini_file_path = str(Path(__file__).parent.parent) + "/settings/dobot_config/dobot_settings.ini"
    ini_file = configparser.ConfigParser()
    ini_file.read(ini_file_path)
    computer_passwd = ini_file.get("COMPUTER", "passcode")
    comd = f"echo {computer_passwd} | sudo -S chmod 777 {file_name}"
    subprocess.run(comd, shell=True)
    with open(file_name, "w+") as f:
        f.write(str(1))


# scan_port, return: 0 or 1
def scan_port():
    rt_flag = 0
    ini_file_path = str(Path(__file__).parent.parent) + "/settings/dobot_config/dobot_settings.ini"
    ini_file = configparser.ConfigParser()
    ini_file.read(ini_file_path)
    computer_passwd = "000"

    com_list = []
    ports = list(serial_stl.comports())
    for i in ports:
        if "USB" in i.device:
            com_list.append(i.device)
        if "ACM" in i.device:
            com_list.append(i.device)
    for _port in com_list:
        computer_passwd = ini_file.get("COMPUTER", "passcode")
        comd = f"echo {computer_passwd} | sudo -S chmod 777 {_port}"
        rt_code = subprocess.run(comd, shell=True)
        if rt_code.returncode:
            rt_flag = 1
            break
    return rt_flag, com_list


# make new dir
def mk_dir(path_dir):
    if not os.path.isdir(path_dir):
        os.makedirs(path_dir, exist_ok=True)
        return True
    else:
        return False


# log maker
def log_write(file_name, data):
    log_path = "logs/"
    print("log_path: ", log_path)
    mk_dir(log_path)
    current_time = time.strftime("%Y-%m-%d %H-%M-%S:", time.localtime())
    with open(log_path+"log.txt", 'a') as f:
        f.writelines(str(current_time)+str(datetime.datetime.now().strftime("%f")[:-3])
                     + " [" + file_name.split("/")[-1] + "] "
                     + str(data))
        f.writelines("\n")
    f.close()


def wait_period(delay_time, start_t) -> None:
    delta_time_ = delay_time/1000
    start, end = 0, 0  # 声明变量
    start = time.time()  # servoJ发送结束时间; 精度延时开始计时时间
    # print("sss: ", start - start_t)
    if (start - start_t) < delta_time_:
        t = (delta_time_ - (start-start_t))   # 将输入t的单位转换为秒，-3是时间补偿
        # print(t)
        while end - start < t:  # 循环至时间差值大于或等于设定值时
            end = time.time()  # 记录结束时间


def save_videos(video, dt, video_path=None):
    if isinstance(video, list):
        print("you")
        cam_names = list(video[0].keys())
        cam_names = sorted(cam_names)
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1/dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [0, 1, 2]] # swap B and R channel
                # image = image[:, :, [2, 1, 0]] # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f'Saved video to: {video_path}')
    elif isinstance(video, dict):
        print("me")
        cam_names = list(video.keys())
        cam_names = sorted(cam_names)
        print(cam_names)
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [0, 1, 2]]  # swap B and R channel
            out.write(image)
        out.release()
        print(f'Saved video to: {video_path}')


def get_firmware_version_satisfied(robot_ip):
    try:
        response = requests.post("http://"+robot_ip+":22000/settings/version")
        if response.status_code == 200:
            rt_version = response.text.split("{")[1].split("\n\t")[3].split(":")[1].split("\"")[1].split("-")[0].split(".")
            rt_num = "".join(rt_version)
            return 1, int(rt_num)
        else:
            return 0, 0
    except Exception as e:
        print(e)
        return 0, 0


def get_robot_type(robot_ip):
    robot_type_list = ["Nova 2", "Nova 5"]
    try:
        response = requests.post("http://"+robot_ip+":22000/properties/controllerType")
        if response.status_code == 200:
            rt_version = response.text.split(":")[-1].split("\"")[1]
            if rt_version in robot_type_list:
                return 1, rt_version
            else:
                return 0, 0
        else:
            return 0, 0
    except Exception as e:
        print(e)
        return 0, 0


if __name__ == "__main__":
    left_version = get_firmware_version_satisfied("192.168.5.1")
    right_version = get_firmware_version_satisfied("192.168.5.2")
    if left_version[1]<=3585:
        print("sss")
    if right_version[1]<=3580:
        print("bbb")
    print(get_robot_type("192.168.5.1")[1], get_robot_type("192.168.5.2")[1])