import os.path
import cv2

from MainWindow import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qt import QRegExp, QRegExpValidator
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QLabel
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton
import sys
import datetime
from PyQt5.QtCore import Qt
import time
from button_switch import SwitchButton
from dobot_function.function_utils import (scan_port, log_write, mk_dir, save_frame, wait_period, time_print,
                                           get_firmware_version_satisfied, get_robot_type)
from dobot_function.manipulate_utils import (load_ini_data_camera, load_ini_data_gripper,
                                             load_ini_data_hands, dynamic_approach, servo_action_check,
                                             calculate_vel_pos, check_pose_protection, check_joint_safety,
                                             get_config, load_ini_data_robot)
from dobot_control.robots.dobot import DobotRobot
from dobot_control.cameras.realsense_camera import RealSenseCamera, get_device_ids
import numpy as np
import configparser
from dobot_control.robots.robot import BimanualRobot
from dobot_control.agents.dobot_agent import DobotAgent
from dobot_control.agents.agent import BimanualAgent
from dobot_control.dynamixel.driver import DynamixelDriver
from pathlib import Path
from dobot_control.gripper.dobot_gripper import DobotGripper
from sencondThread import WorkerThread
from threading import Event, Lock, Thread
import argparse


# detected_robot_type = detect_robot_type(robot_ip="192.168.5.1")   # return "nova2" or "nova5

class MainWinEntry(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.setFixedSize(883, 875)
        # com
        self.bi_robot = None
        self.flag_in = None
        self.last_action = None
        self.left_dir = None
        self.right_dir = None
        self.top_dir = None
        self.start_servo = None
        self.obs = None
        self.safe_limit = 0
        self.is_falling = False
        self.action = None

        # init switch button
        self.switch_robot = SwitchButton()
        self.switch_robot.setParent(self.PB_main_enableRobot)
        self.switch_camera = SwitchButton()
        self.switch_camera.setParent(self.PB_main_enableCamera)
        self.switch_mainHand = SwitchButton()
        self.switch_mainHand.setParent(self.PB_main_enableMainhand)

        self.init_status_bus()
        self.init_buttons()
        self.text_print("Welcome!")
        self.text_print("This product is supported on Ubuntu 20.04 LTS only.")

        self.x_range_left = (-450, 290)
        self.x_range_right = (-290, 450)
        self.y_range = (-750, -160)
        self.z_range_left = 44
        self.z_range_right = 42

        self.cam_left = None
        self.cam_right = None
        self.cam_top = None
        self.img_left = None
        self.img_right = None
        self.img_top = None
        self.bi_agent = None
        self.last_keys_status = np.array(([0, 0], [0, 0]))
        self.start_press_status = np.array(([0, 0], [0, 0]))  # start press
        self.keys_press_count = np.array(([0, 0, 0], [0, 0, 0]))
        self.what_to_do = np.array(([0, 0, 0], [0, 0, 0]))
        self.dt_time = "000"
        self.tic_timer_keys = 0
        self.toc_timer_keys = 0
        self.curr_light = "black"

        self.falling_protection_on = np.array([0])
        self.sensor_protection_on = np.array([0])
        self.img_show = np.array([0])
        self.robot_servo_on = np.array([0])
        self.last_status = np.array(([0, 0, 0], [0, 0, 0]))
        self.ini_file_path = str(Path(__file__).parent) + "/settings/dobot_config/dobot_settings.ini"
        self.save_dir = str(Path(__file__).parent.parent)+"/datasets/"+self.LE_task_name.text()+"/collect_data"
        print(self.save_dir)
        self.img_save_idx = 0
        self.is_sensor_version = 0
        # read setting file
        self.read_config_file()

        self.thread_camera_show_flag = np.array([0])
        self.thread_camera_show = Thread(target=self.cam_read)
        self.thread_camera_show.daemon = True
        self.thread_camera_show.start()

        self.thread_read_main_hand_joints_flag = np.array([0])
        self.thread_read_main_hand_joints = Thread(target=self.read_main_hand_keys)
        self.thread_read_main_hand_joints.daemon = True
        self.thread_read_main_hand_joints.start()

        self.thread_run_control_flag = np.array([0])
        self.thread_run_control = Thread(target=self.run_control)
        self.thread_run_control.daemon = True
        self.thread_run_control.start()

        self.find_port()  # update port list
        self.threads = None
        self.need_firmware_version = 3581
        self.which_robot_type = "Nova 2"
        # ^-?(180|1?[0-7]?\d(\.\d{2})?)$
        # aaa = QRegExpValidator(QRegExp('[0-9\.]+$'))
        # self.LE_settings_safeA_left_xmax.setValidator(aaa)
        log_write(__file__, "111")


    def __del__(self):
        self.thread_read_main_hand_joints_flag[0] = 0
        self.thread_camera_show_flag[0] = 0
        self.thread_run_control_flag[0] = 0

    def closeEvent(self, a0):
        self.thread_read_main_hand_joints_flag[0] = 0
        self.thread_camera_show_flag[0] = 0
        self.thread_run_control_flag[0] = 0
        try:
            if self.bi_robot.is_connect:
                self.bi_robot.set_light("red", 0)
            self.bi_robot.disconnect()
            self.bi_agent.disconnect()
        except Exception as e:
            print(e)

    def init_buttons(self):
        self.PB_main_start.setEnabled(False)
        self.PB_main_stop.setEnabled(False)

    def slot_main_cb_safety_protection(self, rt_fall):
        if rt_fall == 2:
            self.falling_protection_on[0] = 1
            self.text_print(f"Falling protection: on")
        elif rt_fall == 0:
            self.falling_protection_on[0] = 0
            self.text_print(f"Falling protection: off")
        else:
            self.text_print(f"Error: slot_main_cb_robot_falling return {rt_fall}", "r")

    def slot_main_cb_sensor_protection(self, rt_sensor):
        if rt_sensor == 2:
            self.sensor_protection_on[0] = 1
            self.text_print(f"Sensor protection: on")
        elif rt_sensor == 0:
            self.sensor_protection_on[0] = 0
            self.text_print(f"Sensor protection: off")
        else:
            self.text_print(f"Error: slot_main_cb_sensor_pro return {rt_sensor}", "r")

    def slot_main_cb_show_img(self, rt_img):
        if rt_img == 2:
            self.img_show[0] = 1
            self.text_print(f"Img show: on")
        elif rt_img == 0:
            self.img_show[0] = 0
            self.text_print(f"Img show: off")
            self.label_img_left.clear()
            self.label_img_right.clear()
            self.label_img_top.clear()
        else:
            self.text_print(f"Error: slot_main_cb_show_img return {rt_img}", "r")

    def slot_dataset_PB_open_dir(self):
        # print("in 11111111111")
        dir_choose = QFileDialog.getExistingDirectory(self, "Choose dataset dir", "../datasets/", options=QFileDialog.ShowDirsOnly)
        # print("open dir name: ", dir_choose)
        if os.path.isdir(dir_choose + "/collect_data"):
            self.LE_dataset_dir_name.setText(dir_choose)
        else:
            self.text_print("No data found!", "r")

    def slot_dataset_PB_transform_confirm(self):
        if self.LE_dataset_dir_name.text():
            inp_para = {"run_mode": int(1),
                        "dataset_dir": str(self.LE_dataset_dir_name.text()),
                        "is_make_videos": self.CB_dataset_make_video.currentText()}
            self.threads = WorkerThread(inp_para)
            self.threads.signal_trigger_data.connect(self.rt_thread_data)
            self.threads.start()
            self.text_print("Processing datasets: start")
        else:
            self.text_print("Please choose the correct dir first!", "r")

    def rt_thread_data(self, rt_thread):
        if rt_thread[0] == 1:
            self.text_print(str(rt_thread[1]), "b")
        elif rt_thread[0] == 2:
            self.text_print(str(rt_thread[1]), "b")
        elif rt_thread[0] == 11:
            self.text_print(str(rt_thread[1]), "b")
        elif rt_thread[0] == 21:
            self.text_print(str(rt_thread[1]), "3")

    def read_config_file(self):
        is_file_exist = os.path.exists(self.ini_file_path)
        if not is_file_exist:
            self.setEnabled(False)
            self.text_print("'dobot_settings' file missing! Please find it and restart the app", "r")
        else:
            ini_file = configparser.ConfigParser()
            ini_file.read(self.ini_file_path)

            self.LE_task_name.setText(ini_file.get("TASK_NAME", "task_name"))

            self.LE_settings_comPassW.setText(ini_file.get("COMPUTER", "passcode"))
            self.LE_settings_leftIP.setText(ini_file.get("ROBOT_LEFT", "ip"))
            self.LE_settings_safeA_left_xmin.setText(ini_file.get("ROBOT_LEFT", "safe_area").split(",")[0].strip())
            self.LE_settings_safeA_left_xmax.setText(ini_file.get("ROBOT_LEFT", "safe_area").split(",")[1].strip())
            self.LE_settings_safeA_left_ymin.setText(ini_file.get("ROBOT_LEFT", "safe_area").split(",")[2].strip())
            self.LE_settings_safeA_left_ymax.setText(ini_file.get("ROBOT_LEFT", "safe_area").split(",")[3].strip())
            self.LE_settings_safeA_left_zmin.setText(ini_file.get("ROBOT_LEFT", "safe_area").split(",")[4].strip())
            self.LE_settings_safeA_left_zmax.setText(ini_file.get("ROBOT_LEFT", "safe_area").split(",")[5].strip())

            self.LE_settings_rightIP.setText(ini_file.get("ROBOT_RIGHT", "ip"))
            self.LE_settings_safeA_right_xmin.setText(ini_file.get("ROBOT_RIGHT", "safe_area").split(",")[0].strip())
            self.LE_settings_safeA_right_xmax.setText(ini_file.get("ROBOT_RIGHT", "safe_area").split(",")[1].strip())
            self.LE_settings_safeA_right_ymin.setText(ini_file.get("ROBOT_RIGHT", "safe_area").split(",")[2].strip())
            self.LE_settings_safeA_right_ymax.setText(ini_file.get("ROBOT_RIGHT", "safe_area").split(",")[3].strip())
            self.LE_settings_safeA_right_zmin.setText(ini_file.get("ROBOT_RIGHT", "safe_area").split(",")[4].strip())
            self.LE_settings_safeA_right_zmax.setText(ini_file.get("ROBOT_RIGHT", "safe_area").split(",")[5].strip())

            self.LE_settings_gripper_left_pmin.setText(str(int(ini_file.get("GRIPPER_LEFT", "pos").split(",")[0])).strip())
            self.LE_settings_gripper_left_pmax.setText(str(int(ini_file.get("GRIPPER_LEFT", "pos").split(",")[1])).strip())
            self.LE_settings_gripper_right_pmin.setText(str(int(ini_file.get("GRIPPER_RIGHT", "pos").split(",")[0])).strip())
            self.LE_settings_gripper_right_pmax.setText(str(int(ini_file.get("GRIPPER_RIGHT", "pos").split(",")[1])).strip())

            self.x_range_left = (float(self.LE_settings_safeA_left_xmin.text()), float(self.LE_settings_safeA_left_xmax.text()))
            self.x_range_right = (float(self.LE_settings_safeA_right_xmin.text()), float(self.LE_settings_safeA_right_xmax.text()))
            self.y_range = (float(self.LE_settings_safeA_left_ymin.text()), float(self.LE_settings_safeA_left_ymax.text()))
            self.z_range_left = float(self.LE_settings_safeA_left_zmin.text())
            self.z_range_right = float(self.LE_settings_safeA_right_zmin.text())

            self.is_sensor_version = int(ini_file.get("IS_SENSOR_VERSION", "flag"))

    # timer: camera read
    def cam_read(self):
        while True:
            if self.thread_camera_show_flag.copy()[0]:
                try:
                    # log_write(__file__, "camera: 1")
                    self.img_left, _ = self.cam_left.read()
                    self.img_right, _ = self.cam_right.read()
                    self.img_top, _ = self.cam_top.read()
                    # log_write(__file__, "camera: 1.5")
                    if self.img_show:
                        self.label_img_left.setPixmap(self.img_to_pix(self.img_left, self.label_img_left))
                        self.label_img_right.setPixmap(self.img_to_pix(self.img_right, self.label_img_right))
                        self.label_img_top.setPixmap(self.img_to_pix(self.img_top, self.label_img_top))
                    else:
                        if self.label_img_left.pixmap():
                            self.label_img_left.clear()
                            self.label_img_right.clear()
                            self.label_img_top.clear()
                    if not self.switch_camera.state:
                        self.label_img_left.clear()
                        self.label_img_right.clear()
                        self.label_img_top.clear()
                    # log_write(__file__, "camera: 2")
                except:
                    self.thread_camera_show_flag[0] = 0
                    self.text_print("Error: read img", "r")
                    self.slot_main_PB_enableCamera(0)
                    break
            else:
                time.sleep(0.020)
                # log_write(__file__, "camera")

    def img_to_pix(self, img, label_img):

        height, width, _ = img.shape
        img = QImage(img.data.tobytes(), width, height, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        pixmap = QPixmap(pixmap).scaled(label_img.width(), label_img.height(),
                                        aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio)
        return pixmap

    def init_status_bus(self):
        qstBar = self.statusBar()
        qlabel_left = QLabel("Supporter: Dobot-547")
        qlabel_right = QLabel("Software Version: 202412051438")
        qstBar.addWidget(qlabel_left)
        qstBar.addPermanentWidget(qlabel_right)

    # log print
    def text_print(self, str_print, str_color="b"):
        current_time = time.strftime("%Y-%m-%d %H-%M-%S:", time.localtime())
        ms = datetime.datetime.now().strftime("%f")[:-3]
        self.textEditLog.moveCursor(QTextCursor.End, QTextCursor.KeepAnchor)
        if str_color == "b":
            self.textEditLog.append(current_time + ms + ": " + str_print)
        else:
            self.textEditLog.append(current_time + ms + ": "+"<font color=\"#FF0000\">"+str_print+"</font> ")

    def slot_main_PB_poseinit_confirm(self):
        try:
            reset_joints_left = np.deg2rad([-90, 30, -110, 20, 90, 90, 0])
            reset_joints_right = np.deg2rad([90, -30, 110, -20, -90, -90, 0])
            reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
            curr_joints = self.bi_robot.get_joint_state()
            max_delta = (np.abs(curr_joints - reset_joints)).max()
            steps = min(int(max_delta / 0.01), 80)
            print("sssssssssss", steps)
            for jnt in np.linspace(curr_joints, reset_joints, steps):
                self.bi_robot.command_joint_state(jnt, [1, 1])
                time.sleep(0.010)

            # go to the second point
            reset_joints_left = np.deg2rad([-90, 0, -90, 0, 90, 90, 0])  #
            reset_joints_right = np.deg2rad([90, 0, 90, 0, -90, -90, 0])
            reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
            curr_joints = self.bi_robot.get_joint_state()
            max_delta = (np.abs(curr_joints - reset_joints)).max()
            steps = min(int(max_delta / 0.01), 80)
            for jnt in np.linspace(curr_joints, reset_joints, steps):
                self.bi_robot.command_joint_state(jnt, [1, 1])
                time.sleep(0.010)
        except Exception as e:
            self.text_print(f"Init Error: {e}", "r")

    def slot_main_PB_Start(self):
        try:
            self.slot_main_PB_poseinit_confirm()
            self.save_dir = str(Path(__file__).parent.parent)+"/datasets/"+self.LE_task_name.text()+"/collect_data"
            self.robot_servo_on[0] = 1
            self.init_control_paras()
            self.bi_robot.set_light("red", 0)
            self.thread_run_control_flag[0] = 1
            self.PB_main_start.setEnabled(False)
            self.PB_main_stop.setEnabled(True)

            ini_file, _ = load_ini_data_camera()
            ini_file.set(section="TASK_NAME", option="task_name", value=self.LE_task_name.text())
            with open(self.ini_file_path, "w+") as _file:
                ini_file.write(_file)
            _file.close()
            self.groupBox_select.setEnabled(False)
        except Exception as e:
            self.text_print(f"Init Error: {e}", "r")
            self.slot_main_PB_enableRobot(0)

    def slot_main_PB_Stop(self):
        self.PB_main_start.setEnabled(True)
        self.PB_main_stop.setEnabled(False)
        try:
            # self.thread_run_control_flag[0] = 0
            self.slot_main_PB_enableMainhand(0)
            time.sleep(0.050)
            self.groupBox_select.setEnabled(True)
        except Exception as e:
            self.text_print(f"Init Error: {e}", "r")

    def find_port(self):
        self.tab_setting.setEnabled(True)
        self.tab_debug.setEnabled(True)
        self.tab_control.setEnabled(True)
        self.tab_dataset.setEnabled(True)
        self.widget_enable.setEnabled(True)
        rt_code, port_list = scan_port()
        baud_rate_list = [2000000, 1000000]
        if rt_code:
            self.text_print("Code error: Please check the computer passcode!", "r")
            self.widget_enable.setEnabled(False)
            self.tab_setting.setEnabled(True)
            self.tab_debug.setEnabled(False)
            self.tab_control.setEnabled(False)
            self.tab_dataset.setEnabled(False)
            self.widget_enable.setEnabled(False)
            self.tabWidget_main.setCurrentWidget(self.tab_setting)
        else:
            if len(port_list) == 4:
                ini_file, hands_dict = load_ini_data_hands()
                # find hand port
                for which_hand in hands_dict.keys():
                    for _port in port_list:
                        for _baud_rate in baud_rate_list:
                            try:
                                driver = DynamixelDriver(ids=hands_dict[which_hand].joint_ids,
                                                         append_id=hands_dict[which_hand].append_id,
                                                         port=_port,
                                                         baudrate=_baud_rate,
                                                         using_sensor=True if self.is_sensor_version==1 else False)
                                port_list.remove(_port)
                                print("Success(hand): ", which_hand, _port)
                                ini_file.set(section=which_hand, option="port", value=_port)
                                print(_baud_rate)
                                ini_file.set(section=which_hand, option="baud_rate", value=str(_baud_rate))
                                with open(self.ini_file_path, "w+") as _file:
                                    ini_file.write(_file)
                                _file.close()
                                driver.close()
                                break
                            except Exception as e:
                                print(e)
                                continue

                # find gripper port
                ini_file, gripper_dict = load_ini_data_gripper()
                for which_gripper in gripper_dict.keys():
                    for _port in port_list:
                        try:
                            gripper = DobotGripper(port=_port,
                                                   servo_pos=gripper_dict[which_gripper].pos,
                                                   id_name=gripper_dict[which_gripper].id_name)
                            ini_file.set(section=which_gripper, option="port", value=_port)
                            port_list.remove(_port)
                            print("Success(gripper): ", which_gripper, _port)

                            with open(self.ini_file_path, "w+") as _file:
                                ini_file.write(_file)
                            _file.close()
                            gripper.disconnect()
                            break
                        except Exception as e:
                            warnings = e
                            print("***WARNING***: ", warnings)
                            continue
            else:
                self.text_print(f"Find port error: There should be 4 ports, but {len(port_list)} detected actually! Please check and restart", "r")
                self.tabWidget_main.setEnabled(False)
                self.widget_enable.setEnabled(False)
            if len(port_list) == 0:
                self.text_print(f"Port init successfully", "b")
            else:
                self.tabWidget_main.setEnabled(False)
                self.widget_enable.setEnabled(False)
                self.text_print(f"Find port error: There should be no port left, {len(port_list)} left actually! Please check and restart", "r")

    def slot_main_PB_enableRobot(self, rt_bool):
        err_rt = 0
        _, robot_dict = load_ini_data_robot()
        if rt_bool:
            try:
                _robot_left = DobotRobot(robot_ip=robot_dict["ROBOT_LEFT"])
                _robot_right = DobotRobot(robot_ip=robot_dict["ROBOT_RIGHT"])
                self.bi_robot = BimanualRobot(_robot_left, _robot_right)
                self.bi_robot.get_joint_state()
                self.bi_robot.init_gripper()
                self.bi_robot.set_light("red", 0)
                self.text_print("Robot: enable")
                if self.switch_mainHand.state and self.switch_camera.state:
                    self.PB_main_start.setEnabled(True)
                    self.PB_main_stop.setEnabled(False)
                # get version
                _, version_left =  get_firmware_version_satisfied(robot_dict["ROBOT_LEFT"])
                _, version_right = get_firmware_version_satisfied(robot_dict["ROBOT_RIGHT"])
                if version_left < self.need_firmware_version:
                    self.text_print(f"Left hand error [192.168.5.1]: firmware version must >= {self.need_firmware_version} (found: {version_left}), please check and update.", "r")
                    err_rt = 1
                else:
                    self.text_print("Left hand version [192.168.5.1]: " + str(version_left))
                if version_right < self.need_firmware_version:
                    self.text_print(f"Right hand error [192.168.5.2]: firmware version must >= {self.need_firmware_version} (found: {version_right}), please check and update.", "r")
                    err_rt = 1
                else:
                    self.text_print("Right hand version [192.168.5.2]: " + str(version_right))
                # get robot type
                rt1, type_left = get_robot_type(robot_dict["ROBOT_LEFT"])
                rt2, type_right = get_robot_type(robot_dict["ROBOT_RIGHT"])
                if rt1==0:
                    self.text_print(f"Left hand error [192.168.5.1]: get robot type failed.", "r")
                    err_rt = 1
                else:
                    self.text_print("Left hand type [192.168.5.1]: " + type_left)
                if rt2==0:
                    self.text_print(f"Right hand error [192.168.5.2]: get robot type failed.", "r")
                    err_rt = 1
                else:
                    self.text_print("Right hand type [192.168.5.2]: " + type_right)
                if type_left != type_right:
                    self.text_print(f"detect different type of robot.", "r")
                    err_rt = 1
                else:
                    self.which_robot_type = type_left
            except Exception as e:
                self.text_print(str(e), "r")
                err_rt = 1
        else:
            try:
                self.bi_robot.set_light("red", 0)
            except Exception as e:
                print(e)
            try:
                self.bi_robot.disconnect()
            except Exception as e:
                print(e)
            self.PB_main_start.setEnabled(False)
            self.PB_main_stop.setEnabled(False)

            self.switch_robot.state = False
            self.switch_robot.update()
            self.PB_main_enableRobot.setChecked(False)

        # error print
        if err_rt:
            self.text_print("Robot error: please check the robot (include the gripper)", "r")
            self.switch_robot.state = False
            self.switch_robot.update()
            self.PB_main_enableRobot.setChecked(False)
            self.PB_main_start.setEnabled(False)
            self.PB_main_stop.setEnabled(False)
            if self.bi_robot is not None:
                try:
                    self.bi_robot.disconnect()
                except Exception as e:
                    print(e)

    def slot_main_PB_enableCamera(self, rt_bool_ec):
        self.CB_debug_cam_ids.clear()
        self.CB_debug_cam_which.clear()
        err_id_ec = [0, 0]  # num err and id error, init error
        err_id_same = 0
        if rt_bool_ec:
            # id, num check
            _, came_list_id = load_ini_data_camera()
            cam_list = [came_list_id["left"], came_list_id["right"], came_list_id["top"]]
            for item_ in cam_list:
                if cam_list.count(item_) > 1:
                    err_id_same = 1
                    self.slot_main_PB_enableCamera(0)
                    self.text_print("Error: same camera ID!", "r")
            print(cam_list)
            if not err_id_same:
                device_ids_list = get_device_ids()
                for name_key in came_list_id.keys():
                    if came_list_id[name_key] not in device_ids_list:
                        self.text_print(f"Setting file Error: {came_list_id[name_key]} is not in list ({device_ids_list})", "r")
                        err_id_ec[0] = 1
                if not err_id_ec[0]:
                    try:
                        self.cam_left = RealSenseCamera(flip=False, device_id=came_list_id["left"])
                        self.text_print(f"Camera {came_list_id['left']} enable")
                    except Exception as e:
                        self.text_print(f"Left camera ID error: {came_list_id['left']} {e}")
                        err_id_ec[1] = 1
                    try:
                        self.cam_right = RealSenseCamera(flip=True, device_id=came_list_id["right"])
                        self.text_print(f"Camera {came_list_id['right']} enable")
                    except Exception as e:
                        self.text_print(f"Right camera ID error: {came_list_id['right']} {e}")
                        err_id_ec[1] = 1
                    try:
                        self.cam_top = RealSenseCamera(flip=True, device_id=came_list_id["top"])
                        self.text_print(f"Camera {came_list_id['top']} enable")
                    except Exception as e:
                        self.text_print(f"Top camera ID error: {came_list_id['top']} {e}")
                        err_id_ec[1] = 1

                    self.thread_camera_show_flag[0] = 1
                    self.text_print("Camera: enable")
                    if self.switch_mainHand.state and self.switch_robot.state:
                        self.PB_main_start.setEnabled(True)
                        self.PB_main_stop.setEnabled(False)
        else:
            self.switch_camera.state = False
            self.switch_camera.update()
            self.PB_main_enableCamera.setChecked(False)

            self.thread_camera_show_flag[0] = 0
            time.sleep(0.030)
            self.text_print("Info: camera disabled.")
            self.label_img_left.clear()
            self.label_img_right.clear()
            self.label_img_top.clear()
            self.PB_main_start.setEnabled(False)
            self.PB_main_stop.setEnabled(False)
            try:
                self.cam_left.close()
            except:
                print("warning: left camera not open")
            try:
                self.cam_right.close()
            except:
                print("warning: right camera not open")
            try:
                self.cam_top.close()
            except:
                print("warning: top camera not open")
        if 1 in err_id_ec or err_id_same:
            self.switch_camera.state = False
            self.switch_camera.update()
            self.PB_main_enableCamera.setChecked(False)
            self.PB_main_start.setEnabled(False)
            self.PB_main_stop.setEnabled(False)

    def read_main_hand_keys(self):
        while 1:
            # print(1111111)
            if self.thread_read_main_hand_joints_flag.copy()[0]:
                try:
                    tic = time.time()
                    # log_write(__file__, "read  1")
                    now_keys = self.bi_agent.get_keys()
                    self.action = self.bi_agent.act({})
                    # print(now_keys)
                    # action_print = [np.rad2deg(i) for i in action[0:6]]
                    # log_write(__file__, action_print)
                    toc = time.time()
                    # print("read time: ", toc - tic)
                    err_id = self.bi_agent.get_err()

                    tic1 = time.time()
                    # print("bbbbbbbbbb", tic1-tic)
                    if err_id[0]:
                        self.text_print("Left main hand error!", "r")
                        self.slot_main_PB_enableMainhand(0)
                    if err_id[1]:
                        self.text_print("Right main hand error!", "r")
                        self.slot_main_PB_enableMainhand(0)
                    dev_keys = now_keys - self.last_keys_status
                    # log_write(__file__, "read  2")
                    # print(now_keys)
                    # button a
                    for i in range(2):
                        if dev_keys[i, 0] == -1:  # button a: start
                            self.tic_timer_keys = time.time()
                            self.start_press_status[i, 0] = 1
                        if dev_keys[i, 0] == 1 and self.start_press_status[i, 0]:  # button a: end
                            self.start_press_status[i, 0] = 0
                            self.toc_timer_keys = time.time()
                            if self.toc_timer_keys - self.tic_timer_keys < 0.5:
                                self.keys_press_count[i, 0] += 1
                                # print(i, keys_press_count[i, 0], "short press", toc-tic)
                                if self.keys_press_count[i, 0] % 2 == 1:
                                    self.what_to_do[i, 0] = 1
                                    # log_write(__file__, "ButtonA: ["+str(i)+"] unlock")
                                    print("ButtonA: [" + str(i) + "] unlock", self.what_to_do)
                                else:
                                    self.what_to_do[i, 0] = 0
                                    # log_write(__file__, "ButtonA: [" + str(i) + "] lock")
                                    print("ButtonA: [" + str(i) + "] lock", self.what_to_do)

                            elif self.toc_timer_keys - self.tic_timer_keys > 1:
                                self.keys_press_count[i, 1] += 1
                                # print(i, keys_press_count[i, 1], "long press", toc-tic)
                                if self.keys_press_count[i, 1] % 2 == 1:
                                    self.what_to_do[i, 1] = 1
                                    # log_write(__file__, "ButtonA: [" + str(i) + "] servo")
                                    print("ButtonA: [" + str(i) + "] servo")
                                else:
                                    self.what_to_do[i, 1] = 0
                                    # log_write(__file__, "ButtonA: [" + str(i) + "] stop servo")
                                    print("ButtonA: [" + str(i) + "] stop servo")

                    # button B
                    # more than one start servo
                    for i in range(2):
                        if dev_keys[i, 1] == -1:  # B button pressed
                            self.start_press_status[i, 1] = 1
                        if dev_keys[i, 1] == 1:
                            self.start_press_status[i, 1] = 0
                            if self.keys_press_count[0, 2] % 2 == 1:
                                if self.keys_press_count[0, 1] % 2 == 1 or self.keys_press_count[1, 1] % 2 == 1:
                                    self.what_to_do[0, 2] = 1
                                    # log_write(__file__, "ButtonB: [" + str(i) + "] recording")
                                    # new recording
                                    now_time = datetime.datetime.now()
                                    self.dt_time = int(now_time.strftime("%Y%m%d%H%M%S"))
                                    self.keys_press_count[0, 2] += 1
                            else:
                                self.what_to_do[0, 2] = 0
                                self.keys_press_count[0, 2] += 1
                                # log_write(__file__, "ButtonB: [" + str(i) + "] stop recording")

                    # status fall
                    if self.sensor_protection_on[0]:
                        for i in range(2):
                            if now_keys[i, 2] and self.what_to_do[i, 0]:  # button a: lock
                                self.text_print("Error: sensor detection!", "r")
                                try:
                                    self.bi_agent.set_torque(2, 1)
                                except Exception as e:
                                    print(e)
                                self.slot_main_PB_enableMainhand(0)
                                try:
                                    self.bi_robot.set_light("red", 1)
                                except Exception as e:
                                    print(e)
                                self.is_falling = 1
                                # break
                                # que: error and stop

                    self.last_keys_status = now_keys
                    # log_write(__file__, "read time 3")
                    toc = time.time()
                    # print("sssssssssss", toc-tic)
                except Exception as e:
                    self.text_print(f"Read main hand keys error: {e}")
                    self.slot_main_PB_enableMainhand(0)
            else:
                time.sleep(0.020)
                # log_write(__file__, "keys")

    def init_control_paras(self):
        self.flag_in = None
        self.last_action = None
        self.left_dir = None
        self.right_dir = None
        self.top_dir = None
        self.start_servo = None
        self.obs = None
        self.safe_limit = 0
        self.is_falling = False
        self.action = None

        self.last_keys_status = np.array(([0, 0, 0], [0, 0, 0]))
        self.start_press_status = np.array(([0, 0], [0, 0]))  # start press
        self.keys_press_count = np.array(([0, 0, 0], [0, 0, 0]))
        self.what_to_do = np.array(([0, 0, 0], [0, 0, 0]))

    def run_control(self):
        while 1:
            if self.thread_run_control_flag.copy()[0]:
                total_time = 0.04
                tic = time.time()
                # log_write(__file__, "time 1")

                # action = self.bi_agent.act({})
                # action_print = [np.rad2deg(i) for i in action[0:6]]
                # log_write(__file__, action_print)
                action = self.action
                # if action is None:
                    # time_print(action)
                # print(self.action)

                what_to_do_tmp = self.what_to_do.copy()
                dev_what_to_do = what_to_do_tmp - self.last_status
                self.last_status = what_to_do_tmp
                # button A: short press event. lock and unlock
                for i in range(2):
                    if dev_what_to_do[i, 0] != 0:
                        try:
                            self.bi_agent.set_torque(i, not what_to_do_tmp[i, 0])
                        except Exception as e:
                            print(e)

                if self.robot_servo_on:
                    # button A: long press event. servo or not
                    if dev_what_to_do[0, 1] == 1 or dev_what_to_do[1, 1] == 1:
                        # pose check between main hand and the follower
                        print("dynamic approach")
                        for i in range(2):
                            if what_to_do_tmp[i, 1]:
                                self.bi_agent.set_torque(i, True)
                        self.flag_in = np.array([what_to_do_tmp[0, 1], what_to_do_tmp[1, 1]])
                        self.start_servo = True
                        try:
                            self.last_action = dynamic_approach(self.bi_robot, self.bi_agent, self.flag_in)
                        except Exception as e:
                            self.text_print("Error: wrong position to approach!", "r")
                            self.bi_robot.set_light("red", 1)
                            self.slot_main_PB_Stop()
                            self.start_servo = False
                            time.sleep(1)
                        for i in range(2):
                            if what_to_do_tmp[i, 0]:
                                if what_to_do_tmp[i, 1]:
                                    self.bi_agent.set_torque(i, False)
                        try:
                            self.obs = self.bi_robot.get_observations()
                        except Exception as e:
                            print("line 682:", e)
                            self.slot_main_PB_enableRobot(0)
                            self.start_servo = False
                        if self.curr_light != "green" and self.start_servo:
                            self.curr_light = self.bi_robot.set_light("yellow", 1)

                    if dev_what_to_do[0, 1] == -1 or dev_what_to_do[1, 1] == -1:
                        self.flag_in = np.array([what_to_do_tmp[0, 1], what_to_do_tmp[1, 1]])
                        if what_to_do_tmp[0, 1] == 0 and what_to_do_tmp[1, 1] == 0:
                            self.bi_robot.set_light("green", 0)

                    if (what_to_do_tmp[0, 1] or what_to_do_tmp[1, 1]) and self.start_servo:
                        action = self.action
                        if action is not None:
                            # log_write(__file__, "time 2")
                            # print(action[0:6], action[7:13])
                            err3, action = servo_action_check(action, self.last_action, self.flag_in)
                            # log_write(__file__, "time 3")
                            assert err3 != 0, self.bi_robot.set_light("red", 1)

                            # ×××××××××××××××××××××××××××××Security protection×××××××××××××××××××××××××××××××××××××××××××
                            # [Note]: Modify the protection parameters in this section carefully !
                            if self.falling_protection_on:
                                protect_err = [False, False]
                                if (self.safe_limit < 1):
                                    self.safe_limit = self.safe_limit + 1
                                else:
                                    print("***************protection********", self.z_range_right, self.which_robot_type)
                                    positions, vel = calculate_vel_pos(action, self.last_action, total_time, self.which_robot_type)
                                    protect_err[0] = check_pose_protection(positions, vel, what_to_do_tmp,
                                                                           self.x_range_left,
                                                                           self.x_range_right,
                                                                           self.y_range,
                                                                           self.z_range_left,
                                                                           self.z_range_right)
                                    protect_err[1] = check_joint_safety(action)
                                if any(protect_err):
                                    if protect_err[0]:
                                        self.text_print("Error: out of safe area!", "r")
                                    elif protect_err[1]:
                                        self.text_print("Error: falling detection!", "r")
                                    self.bi_robot.set_light("red", 1)
                                    self.slot_main_PB_Stop()
                                    time.sleep(1)
                            # ×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
                            # log_write(__file__, "time 4")
                            # button B: recording or not
                            if dev_what_to_do[0, 2] == 1:
                                self.curr_light = self.bi_robot.set_light("green", 1)
                            elif dev_what_to_do[0, 2] == -1:
                                self.curr_light = self.bi_robot.set_light("yellow", 1)
                            if what_to_do_tmp[0, 2] == 1:
                                self.img_save_idx += 1
                                self.left_dir = self.save_dir + f"/{self.dt_time}/leftImg/"
                                self.right_dir = self.save_dir + f"/{self.dt_time}/rightImg/"
                                self.top_dir = self.save_dir + f"/{self.dt_time}/topImg/"
                                mk_dir(self.right_dir)
                                mk_dir(self.top_dir)
                                if mk_dir(self.left_dir):
                                    self.img_save_idx = 0
                                cv2.imwrite(self.top_dir + f"{self.img_save_idx}.jpg", self.img_top[:, :, ::-1])
                                cv2.imwrite(self.left_dir + f"{self.img_save_idx}.jpg", self.img_left[:, :, ::-1])
                                cv2.imwrite(self.right_dir + f"{self.img_save_idx}.jpg", self.img_right[:, :, ::-1])

                                obs_dir = self.save_dir + f"/{self.dt_time}/observation/"
                                if mk_dir(obs_dir):
                                    self.img_save_idx = 0
                                save_frame(obs_dir, self.img_save_idx, self.obs, action)
                                # log_write(__file__, "save time 3")
                            # log_write(__file__, "time 5")
                            try:
                                self.bi_robot.command_joint_state(action, self.flag_in)
                            except Exception as e:
                                print(e)
                                self.text_print(f"802: {e}", "r")
                                self.slot_main_PB_Stop()
                                self.slot_main_PB_enableRobot(0)
                            # log_write(__file__, "time 6")
                            try:
                                self.obs = self.bi_robot.get_observations()
                            except Exception as e:
                                print(e)
                                self.text_print(f"810: {e}", "r")
                                self.slot_main_PB_Stop()
                                self.slot_main_PB_enableRobot(0)
                            self.obs["joint_positions"][6] = action[6]
                            self.obs["joint_positions"][13] = action[13]
                            self.last_action = action
                        else:
                            self.start_servo = False
                            self.safe_limit = 0
                    # log_write(__file__, "time 7")
                toc = time.time()
                if toc-tic > 0.050:
                    print("TOTAL TIME (S): ", toc-tic)
                # else:
                #     print("TOTAL TIME: ", toc - tic)
                # log_write(__file__, f"total time {toc-tic}")
                # log_write(__file__, "time 8")
            else:
                time.sleep(0.020)
                # log_write(__file__, "control")

    def slot_main_PB_enableMainhand(self, rt_bool_em):
        if not self.thread_read_main_hand_joints.is_alive():
            log_write(__file__, "833: dead main hand")
            self.thread_read_main_hand_joints.start()
        if not self.thread_run_control.is_alive():
            log_write(__file__, "833: dead run control")
            self.thread_run_control.start()
        print("Thread alive: ", self.thread_read_main_hand_joints.is_alive(), self.thread_run_control.is_alive())
        self.init_control_paras()
        err_em = 0
        _, hands_dict = load_ini_data_hands()
        # try:
        #     self.bi_robot.set_light("red", 0)
        # except Exception as e:
        #     print("eeer", e)
        if rt_bool_em:
            self.find_port()
            try:
                left_agent = DobotAgent(which_hand="LEFT", dobot_config=hands_dict["HAND_LEFT"],
                                        using_sensor=True if self.is_sensor_version==1 else False)
            except Exception as e:
                self.text_print(f"Left main hand connect error: {e}", "r")
            try:
                right_agent = DobotAgent(which_hand="RIGHT", dobot_config=hands_dict["HAND_RIGHT"],
                                         using_sensor=True if self.is_sensor_version==1 else False)
            except Exception as e:
                self.text_print(f"Right main hand connect error:{e}", "r")
            try:
                self.bi_agent = BimanualAgent(left_agent, right_agent)
            except:
                self.find_port()
            self.robot_servo_on[0] = 0
            self.thread_read_main_hand_joints_flag[0] = 1
            self.thread_run_control_flag[0] = 1
            self.text_print("Main hand: enable")
            if self.switch_camera.state and self.switch_robot.state:
                self.PB_main_start.setEnabled(True)
                self.PB_main_stop.setEnabled(False)
        else:
            if self.bi_agent is not None:
                self.bi_agent.disconnect()
            self.thread_run_control_flag[0] = 0
            self.thread_read_main_hand_joints_flag[0] = 0
            self.PB_main_start.setEnabled(False)
            self.PB_main_stop.setEnabled(False)

            self.switch_mainHand.state = False
            self.switch_mainHand.update()
            self.PB_main_enableMainhand.setChecked(False)

        if err_em:
            self.switch_mainHand.state = False
            self.switch_mainHand.update()
            self.PB_main_enableMainhand.setChecked(False)

    def slot_debug_rob_getangle(self):
        try:
            rt_angle = self.bi_robot.get_joint_state()
            rt_angle = [np.rad2deg(i) for i in rt_angle]
            self.text_print("Left robot pose: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f" %
                            (rt_angle[0], rt_angle[1], rt_angle[2], rt_angle[3], rt_angle[4], rt_angle[5]), "b")
            self.text_print("Right robot pose: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f" %
                            (rt_angle[7], rt_angle[8], rt_angle[9], rt_angle[10], rt_angle[11], rt_angle[12]), "b")
        except Exception as e:
            self.text_print(f"Get angle Error: {e}", "r")

    def slot_debug_rob_light(self):
        try:
            self.bi_robot.set_light(self.CB_debug_robot_light.currentText(), 1)
        except Exception as e:
            self.text_print(f"Set light Error: {e}", "r")

    def slot_debug_gripper_open(self, rt_bool):
        if rt_bool:
            self.PB_debug_gripper_control.setText("Close")
            try:
                self.bi_robot.gripper_control("open")
            except Exception as e:
                self.text_print(f"Set light Error: {e}", "r")
        else:
            self.PB_debug_gripper_control.setText("Open")
            try:
                self.bi_robot.gripper_control("close")
            except Exception as e:
                self.text_print(f"Set light Error: {e}", "r")

    def slot_debug_cam_searchID(self):
        if self.switch_camera.state:
            self.text_print("Search failed: please disable the camera first!", "r")
        else:
            self.CB_debug_cam_ids.clear()
            self.CB_debug_cam_which.clear()
            device_ids_list = get_device_ids()
            if len(device_ids_list) == 3:
                ini_file, cam_dict = load_ini_data_camera()
                for idx in range(len(device_ids_list)):
                    self.CB_debug_cam_ids.addItem(device_ids_list[idx])
                    self.CB_debug_cam_which.addItem(list(cam_dict.keys())[idx])
                    ini_file.set(section="CAMERA", option=list(cam_dict.keys())[idx], value=device_ids_list[idx])
                    with open(self.ini_file_path, "w+") as _file:
                        ini_file.write(_file)
                    _file.close()
                self.text_print(f"Found {len(device_ids_list)} Realsense cameras")
                self.text_print(f"Please finish the camera ID setting")
                self.tab_setting.setEnabled(False)
                self.tab_debug.setEnabled(True)
                self.tab_control.setEnabled(False)
                self.tab_dataset.setEnabled(False)
                self.widget_enable.setEnabled(False)
            else:
                for idx in range(len(device_ids_list)):
                    self.text_print(f"Found {device_ids_list[idx]}", "r")
                self.text_print(f"Found {len(device_ids_list)} Realsense cameras (3 needed), please check and re-search", "r")

    def slot_debug_cam_comfirmID(self):
        if self.CB_debug_cam_ids.count():
            which_cam_id = self.CB_debug_cam_ids.currentText()
            which_cam_name = self.CB_debug_cam_which.currentText()
            ini_file, _ = load_ini_data_camera()
            ini_file.set(section="CAMERA", option=which_cam_name, value=which_cam_id)
            self.text_print(f"Set {which_cam_id} to {which_cam_name}: ok")
            with open(self.ini_file_path, "w+") as _file:
                ini_file.write(_file)
            _file.close()
            self.CB_debug_cam_which.removeItem(self.CB_debug_cam_which.currentIndex())
            self.CB_debug_cam_ids.removeItem(self.CB_debug_cam_ids.currentIndex())
            if self.CB_debug_cam_ids.count() == 0:
                self.tab_setting.setEnabled(True)
                self.tab_debug.setEnabled(True)
                self.tab_control.setEnabled(True)
                self.tab_dataset.setEnabled(True)
                self.widget_enable.setEnabled(True)
        else:
            self.text_print("Warning: please 'Search'  first!", "r")

    def slot_settings_para_apply(self):
        if self.switch_camera.state or self.switch_robot.state or self.switch_mainHand.state:
            self.text_print("Warning: please disable the robot, camera and main hand first!", "r")
        elif self.LE_settings_leftIP.text() == self.LE_settings_rightIP.text():
            self.text_print("Apply Error: the IP of two robot must be different!", "r")
        else:
            ini_file, cam_dict = load_ini_data_camera()
            # safe_area update
            ini_file.set(section="ROBOT_LEFT", option="safe_area", value=
                        str([float(self.LE_settings_safeA_left_xmin.text()), float(self.LE_settings_safeA_left_xmax.text()),
                             float(self.LE_settings_safeA_left_ymin.text()), float(self.LE_settings_safeA_left_ymax.text()),
                             float(self.LE_settings_safeA_left_zmin.text()), float(self.LE_settings_safeA_left_zmax.text())]).replace("[", '').replace("]", ''))
            ini_file.set(section="ROBOT_RIGHT", option="safe_area", value=
                        str([float(self.LE_settings_safeA_right_xmin.text()), float(self.LE_settings_safeA_right_xmax.text()),
                             float(self.LE_settings_safeA_right_ymin.text()), float(self.LE_settings_safeA_right_ymax.text()),
                             float(self.LE_settings_safeA_right_zmin.text()), float(self.LE_settings_safeA_right_zmax.text())]).replace("[", '').replace("]", ''))
            # ip update
            ini_file.set(section="ROBOT_LEFT", option="ip", value=self.LE_settings_leftIP.text())
            ini_file.set(section="ROBOT_RIGHT", option="ip", value=self.LE_settings_rightIP.text())

            # gripper update
            # ip update
            ini_file.set(section="GRIPPER_LEFT", option="pos", value=
                    str([int(self.LE_settings_gripper_left_pmin.text()), int(self.LE_settings_gripper_left_pmax.text())]).replace("[", '').replace("]", ''))
            ini_file.set(section="GRIPPER_RIGHT", option="pos", value=
                    str([int(self.LE_settings_gripper_right_pmin.text()), int(self.LE_settings_gripper_right_pmax.text())]).replace("[", '').replace("]", ''))
            print(str([int(self.LE_settings_gripper_left_pmin.text()), int(self.LE_settings_gripper_left_pmax.text())]).replace("[", '').replace("]", ''))
            # computer password update
            ini_file.set(section="COMPUTER", option="passcode", value=self.LE_settings_comPassW.text())
            with open(self.ini_file_path, "w+") as _file:
                ini_file.write(_file)
            _file.close()

            self.x_range_left = (float(self.LE_settings_safeA_left_xmin.text()), float(self.LE_settings_safeA_left_xmax.text()))
            self.x_range_right = (float(self.LE_settings_safeA_right_xmin.text()), float(self.LE_settings_safeA_right_xmax.text()))
            self.y_range = (float(self.LE_settings_safeA_left_ymin.text()), float(self.LE_settings_safeA_left_ymax.text()))
            self.z_range_left = float(self.LE_settings_safeA_left_zmin.text())
            self.z_range_right = float(self.LE_settings_safeA_right_zmin.text())

            self.find_port()
            self.text_print("New parameters are saved in config file")

    def slot_debug_calibration_confirm(self):
        if self.switch_mainHand.state:
            ini_file, hands_dict = load_ini_data_hands()
            pose_two = {"HAND_LEFT": self.bi_agent.act({})[:7], "HAND_RIGHT": self.bi_agent.act({})[7:]}
            gripper_two = {"HAND_LEFT": self.bi_agent.get_gripper_pos()[0], "HAND_RIGHT": self.bi_agent.get_gripper_pos()[-1]}
            for _hand in hands_dict.keys():
                offsets, pos_gripper = get_config(_hand, hands_dict[_hand], pose_two[_hand], gripper_two[_hand])
                ini_file.set(section=_hand, option="joint_offsets",
                             value=str(offsets).replace("[", '').replace("]", ''))
                ini_file.set(section=_hand, option="gripper_config",
                             value=str(pos_gripper).replace("[", '').replace("]", ''))
                with open(self.ini_file_path, "w+") as _file:
                    ini_file.write(_file)
                _file.close()
            self.slot_main_PB_enableMainhand(0)
            self.text_print("Calibration success: ok")
        else:
            self.text_print("Calibration error: you should enable main hand first!", "r")

    def error_disable_all(self, err_string):
        self.centralwidget.setEnabled(False)
        self.text_print(err_string, "r")
        self.text_print("Please check, and restart it!", "r")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # 适应高DPI设备
    app.setAttribute(Qt.AA_EnableHighDpiScaling)
    # 解决图片在不同分辨率显示模糊问题
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    window = MainWinEntry()
    window.setWindowTitle("X-Trainer")

    window.show()
    sys.exit(app.exec_())
