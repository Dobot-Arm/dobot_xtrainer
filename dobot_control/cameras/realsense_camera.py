import time
from typing import List, Optional, Tuple
import numpy as np
from dobot_control.cameras.camera import CameraDriver
import cv2
import pyrealsense2 as rs
from dobot_function.manipulate_utils import load_ini_data_camera, load_ini_data_gripper, load_ini_data_hands

def get_device_ids() -> List[str]:

    ctx = rs.context()
    devices = ctx.query_devices()
    device_ids = []
    for dev in devices:
        dev.hardware_reset()
        device_ids.append(dev.get_info(rs.camera_info.serial_number))
    time.sleep(2)
    return device_ids


class RealSenseCamera(CameraDriver):
    def __repr__(self) -> str:
        return f"RealSenseCamera(device_id={self._device_id})"

    def __init__(self, device_id: Optional[str] = None, flip: bool = False):
        print("init", device_id)
        self._device_id = device_id
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(device_id)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 90)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 90)
        self._pipeline.start(config)
        self._flip = flip
        # print(device_id)
        for _ in range(50):
            self.read()

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,  # farthest: float = 0.12
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read a frame from the camera.

        Args:
            img_size: The size of the image to return. If None, the original size is returned.
            farthest: The farthest distance to map to 255.

        Returns:
            np.ndarray: The color image, shape=(H, W, 3)
            np.ndarray: The depth image, shape=(H, W, 1)
        """

        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        # depth_image = cv2.convertScaleAbs(depth_image, alpha=0.03)
        if img_size is None:
            image = color_image[:, :, ::-1]
            depth = depth_image
        else:
            image = cv2.resize(color_image, img_size)[:, :, ::-1]
            depth = cv2.resize(depth_image, img_size)

        # rotate 180 degree's because everything is upside down in order to center the camera
        if self._flip:
            image = cv2.rotate(image, cv2.ROTATE_180)
            depth = cv2.rotate(depth, cv2.ROTATE_180)[:, :, None]
        else:
            depth = depth[:, :, None]

        return image, depth


    def close(self):
        self._pipeline.stop()


if __name__ == "__main__":
    device_ids = get_device_ids()
    print(f"Found {len(device_ids)} devices")
    print(device_ids)
    came_list_id = load_ini_data_camera()
    print(came_list_id)
    for name_key in came_list_id.keys():
        print(came_list_id[name_key])
        if came_list_id[name_key] not in device_ids:
            print("error", came_list_id[name_key])
    # rs2 = RealSenseCamera(flip=True, device_id=device_ids[1])
    # rs3 = RealSenseCamera(flip=True, device_id=device_ids[2])
    # img_list = [None, None, None]
    #
    # while True:
    #     tic = time.time()
    #     img_list[0], _ = rs1.read()
    #     img_list[1], _ = rs2.read()
    #     img_list[2], _ = rs3.read()
    #     cv2.waitKey(1)
    #     toc = time.time()
    #     print(toc-tic)