import torch
import carla
import math
import numpy as np
# import torchvision.transforms as transforms
from typing import *
import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import random
import time
import numpy as np
import cv2

IM_WIDTH = 640
IM_HEIGHT = 480


"""
使用示例：
    ...
    >>> rgb_camera = ...
    >>> patch = torch.load('[PATCH_FILE_PATH]', map_location='cpu')
    >>> # 下面的参数都应调整合理，patch位置尽量设置在车附近方便看效果，patch_rotation三个值分别对应pitch, yaw, roll的旋转角度
    >>> # 经过旋转之前patch的初始状态是平躺在地上的，所以角度也需要设置一个合理的；pixel_length应该需要尝试找一个合适的大小
    >>> carla_patch = CarlaParch(patch, patch_location=(10, 12.0, 2.3), patch_rotation=(10, 20, 30), pixel_length=0.1)
    >>> camera_callback = CameraCallback(rgb_camera, carla_patch, '[PATH_TO_SAVE_IMGS]')
    >>> # 启用调用rgb_camera的listen方法，传入camera_callback, 如果没问题的话每一帧画面都会被保存下来
    >>> # 或者你觉得怎么处理得到的每帧画面更好你可以简单修改一下最后一行代码
    >>> rgb_camera.listen(camera_callback)
    ...
"""


class CarlaPatch:
    """
    init parameters:
        flat_patch: torch.Tensor or ndarray, with shape (C, H, W)
        patch_location: tuple or carla.Location, (x, y, z), e.g. (10, 10.1, 0.6)
        patch_rotation: tuple or carla.Rotation, (pitch, yaw, roll), e.g. (30 degrees, 20, 10)
        pixel_length: float [对应的是patch的一个像素在仿真世界里对应的长度，这个数值大概多少我还没有概念， 需要尝试一下]
    """
    def __init__(
            self,
            flat_patch: torch.Tensor or np.ndarray,
            patch_location: Tuple or carla.Location,
            patch_rotation: Tuple or carla.Rotation,
            pixel_length: float = 0.1,
            patch_transform: Callable = None

    ):
        self.flat_patch = flat_patch.cpu() if isinstance(flat_patch, torch.Tensor) else torch.tensor(flat_patch)

        self.patch_location = patch_location
        self.patch_rotation = patch_rotation
        self.pixel_length = pixel_length

        self.patch_pos_3d = self._trans_all_dots2pos_3d()

    @staticmethod
    def _default_patch_trans(patch: torch.Tensor):
        pass

    def _trans_all_dots2pos_3d(self) -> torch.Tensor:
        """
        通过给定的location，rotation和输入的patch，返回每个像素点的坐标，形状是（3， H， W），3对应x, y, z坐标
        """
        patch_height, patch_width = self.flat_patch.shape[1], self.flat_patch.shape[2]

        x_init_coords = torch.arange(0, self.pixel_length * patch_height, self.pixel_length)
        y_init_coords = torch.arange(0, self.pixel_length * patch_width, self.pixel_length)
        z_init_coords = torch.zeros((patch_height, patch_width))
        patch_init_coords = torch.stack([
            x_init_coords.unsqueeze(-1).broadcast_to(-1, patch_width),
            y_init_coords.unsqueeze(0).broadcast_to(patch_height, -1),
            z_init_coords,
        ], dim=0)

        if isinstance(self.patch_location, carla.Location):
            x, y, z = self.patch_location.x, self.patch_location.y, self.patch_location.z
        else:
            x, y, z = self.patch_location[0], self.patch_location[1], self.patch_location[2]

        location = torch.tensor([x, y, z]).unflatten(0, (3, 1, 1)).broadcast_to((3, patch_height, patch_width))

        if isinstance(self.patch_rotation, carla.Rotation):
            pitch, yaw, roll = self.patch_rotation.pitch, self.patch_rotation.yaw, self.patch_rotation.roll
        else:
            pitch, yaw, roll = self.patch_rotation[0], self.patch_rotation[1], self.patch_rotation[2]

        pitch = pitch / 180.0 * math.pi
        yaw = pitch / 180.0 * math.pi
        roll = pitch / 180.0 * math.pi

        rx = torch.tensor([
            [1, 0, 0], [0, math.cos(roll), - math.sin(roll)], [0, math.sin(roll), math.cos(roll)],
        ], dtype=torch.float)
        ry = torch.tensor([
            [math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [- math.sin(pitch), 0, math.cos(pitch)]
        ], dtype=torch.float)
        rz = torch.tensor([
            [math.cos(roll), - math.sin(roll), 0], [math.sin(roll),  math.cos(roll), 0], [0, 0, 1]
        ], dtype=torch.float)

        coords_rotated = (rz @ ry @ rx @ patch_init_coords.reshape(3, -1))

        coords = coords_rotated.reshape(3, patch_height, patch_width) + location

        return torch.concat([coords, torch.ones((1, patch_height, patch_width))], dim=0)

    def get_patch(self):
        """
        返回patch本身以及patch每个像素点的3d坐标
        """
        return self.flat_patch, self.patch_pos_3d

    def __call__(self, *args, **kwargs):
        return self.get_patch()


class CameraCallback:
    """
    init_parameters:
        rgb_camera: Sensor in carla
        carla_patch: instance of CarlaPatch
        output_folder_path: path of folder to save camera outputs(image format)
    """
    def __init__(self, rgb_camera: carla.Sensor, carla_patch: CarlaPatch, output_folder_path=None, depth_camera=None):
        self.camera_transform = rgb_camera
        self.carla_patch = carla_patch
        self.patch_data, self.patch_global_coords = self.carla_patch()
        self.folder_path = output_folder_path

    def _get_matrices(self, camera_data):
        WINDOW_WIDTH_HALF = camera_data.height / 2
        WINDOW_HEIGHT_HALF = camera_data.width / 2
        in_mat = np.identity(3)
        in_mat[0, 2] = WINDOW_WIDTH_HALF
        in_mat[1, 2] = WINDOW_HEIGHT_HALF
        f = WINDOW_WIDTH_HALF / (math.tan(90.0 * math.pi / 360.0))
        in_mat[0, 0] = in_mat[1, 1] = f

        ex_mat = self.camera_transform.get_transform().get_inverse_matrix()

        return torch.tensor(in_mat, dtype=torch.float32), torch.tensor(ex_mat)

    @staticmethod
    def _convert_raw_data(camera_data):
        camera_data.convert(carla.ColorConverter.Raw)
        img_tensor = torch.tensor(camera_data.raw_data, dtype=torch.uint8)
        # convert to (c, h, w)
        return img_tensor.reshape((camera_data.height, camera_data.width, 4))[:, :, :3].permute((2, 1, 0))

    def __call__(self, camera_data: carla.Image):

        in_mat, ex_mat = self._get_matrices(camera_data)

        # axis_rota_mat = torch.tensor([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
        # patch_camera_coords = axis_rota_mat @ torch.inverse(ex_mat) @ patch_global_coords.reshape(3, -1)
        patch_camera_coords = ex_mat @ self.patch_global_coords.reshape(4, -1)
        cords_xyz = patch_camera_coords[:3, :]
        camera_coords_minus_zx = torch.stack([cords_xyz[1, :], -cords_xyz[2, :], -cords_xyz[0, :]])
        patch_pos2d = in_mat @ camera_coords_minus_zx
        patch_pos2d = torch.tensor([
            (patch_pos2d[0] / patch_pos2d[2]).numpy(),
            (patch_pos2d[1] / patch_pos2d[2]).numpy(),
            (patch_pos2d[2]).numpy()]
        )
        # patch_pos2d[0] /= patch_pos2d[2]
        # patch_pos2d[1] /= patch_pos2d[2]

        patch_pos2d = patch_pos2d.reshape(self.patch_data.shape)
        # 暂时简单使用ReLU当filter过滤掉非法的负值索引, 后续结合深度相机进一步考虑遮挡、视野外的情况
        indices = patch_pos2d[0: 2].to(torch.long)
        indices = torch.max(torch.tensor([0, 0]).unflatten(0, (2, 1, 1)), indices)
        indices = torch.min(torch.tensor([camera_data.height - 1, camera_data.width - 1]).unflatten(0, (2, 1, 1)), indices)
        img_tensor = CameraCallback._convert_raw_data(camera_data)
        # TODO: add mask when using depth camera
        # replace pixels in camera output with patch pixels
        img_tensor[:, indices[1], indices[0]] = (self.patch_data * 255).to(torch.uint8)

        # save camera output to disk
        # transforms.ToPILImage()(img_tensor).save(self.folder_path + f'{camera_data.frame}.jpg')
        # i = np.array(image.raw_data)
        # i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        # i3 = i2[:, :, :3]
        # cv2.imshow("", img_tensor.permute(2, 1, 0).numpy().reshape(IM_HEIGHT, IM_WIDTH, 3))
        # cv2.waitKey(1)
        cv2.imwrite(self.folder_path + f'{camera_data.frame}.jpg', img_tensor.permute(2, 1, 0).numpy())
        # return img_tensor / 255.0


def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("", i3)
    cv2.waitKey(1)
    return i3/255.0


if __name__ == '__main__':
    actor_list = []
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(500.0)

        world = client.get_world()

        blueprint_library = world.get_blueprint_library()

        bp = blueprint_library.filter('model3')[0]
        print(bp)

        spawn_point = carla.Transform(
            carla.Location(x=10, y=14, z=1.0),
            carla.Rotation(pitch=0, yaw=180, roll=0))

        vehicle = world.spawn_actor(bp, spawn_point)
        vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
        # vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.

        actor_list.append(vehicle)

        # https://carla.readthedocs.io/en/latest/cameras_and_sensors
        # get the blueprint for this sensor
        blueprint = blueprint_library.find('sensor.camera.rgb')
        # change the dimensions of the image
        blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
        blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
        blueprint.set_attribute('fov', '110')

        # Adjust sensor relative to vehicle
        spawn_point = carla.Transform(carla.Location(x=-2.5, y=0, z=3.7))

        # spawn the sensor and attach to vehicle.
        sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)

        # add sensor to list of actors
        actor_list.append(sensor)

        # do something with this sensor
        patch = torch.load('C:/Users/Kai/Desktop/patch.pt', map_location='cpu')

        # x z y
        carla_patch = CarlaPatch(patch, patch_location=(100, 100, -100.3), patch_rotation=(70, 60, 40),
                                 pixel_length=0.5)

        camera_callback = CameraCallback(spawn_point, carla_patch, 'C:/Users/kai/Desktop/frames/')
        sensor.listen(lambda data: camera_callback(data))

        time.sleep(5)

    finally:
        print('destroying actors')
        for actor in actor_list:
            actor.destroy()
        print('done.')