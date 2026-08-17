import sys
import time
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import argparse
from pathlib import Path
import os
from dobot_function.function_utils import mk_dir, save_videos, data_read_from_collect
import cv2
import h5py
from dobot_function.function_utils import log_write


class WorkerThread(QThread):
    signal_trigger_data = pyqtSignal(tuple)

    def __init__(self, args):
        super().__init__()
        self.args = args
        print(self.args)

    def run(self):
        if self.args["run_mode"] == 1:
            dataset_dir = self.args["dataset_dir"] + "/collect_data/"
            print(dataset_dir)
            output_video_dir = self.args["dataset_dir"] + "/output_videos/"
            mk_dir(output_video_dir)
            output_train_data = self.args["dataset_dir"] + "/train_data/"
            mk_dir(output_train_data)

            all_data_dir = os.listdir(dataset_dir)
            # log_write(__file__, all_data_dir)
            all_data_dir.sort(key=lambda x: int(x))
            # log_write(__file__, all_data_dir)
            MIRROR_STATE_MULTIPLY = np.array((1, 1, 1, 1, 1, 1, 1))

            for idx in range(len(all_data_dir)):
                print("dealing with : ", idx)
                one_data_dir = dataset_dir + all_data_dir[idx] + "/"
                print(one_data_dir)
                qpos, qvel, action, base_action, image_dict, is_sim = data_read_from_collect(one_data_dir)
                qpos = np.concatenate([qpos[:, :7] * MIRROR_STATE_MULTIPLY, qpos[:, 7:] * MIRROR_STATE_MULTIPLY], axis=1)
                qvel = np.concatenate([qvel[:, :7] * MIRROR_STATE_MULTIPLY, qvel[:, 7:] * MIRROR_STATE_MULTIPLY], axis=1)
                action = np.concatenate([action[:, :7] * MIRROR_STATE_MULTIPLY, action[:, 7:] * MIRROR_STATE_MULTIPLY],
                                        axis=1)
                if base_action is not None:
                    base_action = base_action * np.array((1, 1))

                if 'left_wrist' in image_dict.keys():
                    image_dict['left_wrist'], image_dict['right_wrist'] = \
                        image_dict['left_wrist'], image_dict['right_wrist']
                elif 'cam_left_wrist' in image_dict.keys():
                    image_dict['cam_left_wrist'], image_dict['cam_right_wrist'] = \
                        image_dict['cam_left_wrist'][:, :, ::-1], image_dict['cam_right_wrist'][:, :, ::-1]
                else:
                    raise Exception('No left_wrist or cam_left_wrist in image_dict')

                if 'top' in image_dict.keys():
                    image_dict['top'] = image_dict['top']
                elif 'cam_high' in image_dict.keys():
                    image_dict['cam_high'] = image_dict['cam_high'][:, :, ::-1]
                else:
                    raise Exception('No top or cam_high in image_dict')

                # saving
                data_dict = {
                    '/observations/qpos': qpos,
                    '/observations/qvel': qvel,
                    '/action': action,
                    '/base_action': base_action,
                } if base_action is not None else {
                    '/observations/qpos': qpos,
                    '/observations/qvel': qvel,
                    '/action': action,
                }
                for cam_name in image_dict.keys():
                    data_dict[f'/observations/images/{cam_name}'] = image_dict[cam_name]
                max_timesteps = len(qpos)

                COMPRESS = True

                if COMPRESS:
                    # JPEG compression
                    t0 = time.time()
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # tried as low as 20, seems fine
                    compressed_len = []
                    for cam_name in image_dict.keys():
                        image_list = data_dict[f'/observations/images/{cam_name}']
                        compressed_list = []
                        compressed_len.append([])
                        for image in image_list:
                            result, encoded_image = cv2.imencode('.jpg', image,
                                                                 encode_param)  # 0.02 sec # cv2.imdecode(encoded_image, 1)
                            compressed_list.append(encoded_image[:, 0])
                            compressed_len[-1].append(len(encoded_image))
                        data_dict[f'/observations/images/{cam_name}'] = compressed_list
                    print(f'compression: {time.time() - t0:.2f}s')

                    # pad so it has same length
                    t0 = time.time()
                    compressed_len = np.array(compressed_len)
                    padded_size = compressed_len.max()
                    for cam_name in image_dict.keys():
                        compressed_image_list = data_dict[f'/observations/images/{cam_name}']
                        padded_compressed_image_list = []
                        for compressed_image in compressed_image_list:
                            padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                            image_len = len(compressed_image)
                            padded_compressed_image[:image_len] = compressed_image
                            padded_compressed_image_list.append(padded_compressed_image)
                        data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list
                    print(f'padding: {time.time() - t0:.2f}s')

                # HDF5
                t0 = time.time()
                dataset_path = os.path.join(output_train_data, f'episode_init_{idx}')
                with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                    root.attrs['sim'] = is_sim
                    root.attrs['compress'] = COMPRESS
                    obs = root.create_group('observations')
                    image = obs.create_group('images')
                    for cam_name in image_dict.keys():
                        if COMPRESS:
                            _ = image.create_dataset(cam_name, (max_timesteps, padded_size), dtype='uint8',
                                                     chunks=(1, padded_size), )
                        else:
                            _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                                     chunks=(1, 480, 640, 3), )
                    qpos = obs.create_dataset('qpos', (max_timesteps, 14))
                    qvel = obs.create_dataset('qvel', (max_timesteps, 14))
                    action = root.create_dataset('action', (max_timesteps, 14))
                    if base_action is not None:
                        base_action = root.create_dataset('base_action', (max_timesteps, 2))

                    for name, array in data_dict.items():
                        root[name][...] = array

                    if COMPRESS:
                        _ = root.create_dataset('compress_len', (len(image_dict.keys()), max_timesteps))
                        root['/compress_len'][...] = compressed_len

                self.signal_trigger_data.emit((1, str(f'Processing datasets ({idx+1}/{len(all_data_dir)}) [{all_data_dir[idx]}] : done ')))

                if self.args["is_make_videos"] == "yes":
                    save_videos(image_dict, 0.02, video_path=os.path.join(output_video_dir + f'{all_data_dir[idx]}_video.mp4'))
            self.signal_trigger_data.emit((11, str(f'Processing datasets: finished')))