# Project GLS
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from graphics_utils import getWorld2View2, focal2fov, fov2focal
import pyrender
import numpy as np
import trimesh
import os
from pyrender.shader_program import ShaderProgramCache
import numpy as np
from plyfile import PlyData, PlyElement
import cv2
import open3d as o3d
from os import makedirs

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

path = ''
ply_path = os.path.join(path, "sparse/0", "points.ply")
bin_path = os.path.join(path, "sparse/0/points3D.bin")
txt_path = os.path.join(path, "sparse/0/points3D.txt")
npy_path = os.path.join(path, "sparse/0/points3D.npy")
try:
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
except:
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
    cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

try:
    xyz, rgb, _ = read_points3D_binary(bin_path)
except:
    xyz, rgb, _ = read_points3D_text(txt_path)

storePly(ply_path, xyz, rgb)
pcd = o3d.t.io.read_point_cloud(ply_path)
print(pcd)
depthp_path = os.path.join(path, "depth_p")
makedirs(depthp_path, exist_ok=True)
for idx, key in enumerate(cam_extrinsics):
    extr = cam_extrinsics[key]
    intr = cam_intrinsics[extr.camera_id]
    print(extr.name)
    height = intr.height
    width = intr.width

    R = np.transpose(qvec2rotmat(extr.qvec))
    T = np.array(extr.tvec)

    if intr.model=="SIMPLE_PINHOLE":
        focal_length_x = intr.params[0]
        focal_length_y = intr.params[0]
    else:
        focal_length_x = intr.params[0]
        focal_length_y = intr.params[1]

    # 设置相机内参
    cx = 0.5 * width
    cy = 0.5 * height

    pose = np.identity(4)
    pose[:3,:3] = R.transpose(-1,-2)
    pose[:3, 3] = T

    intrinsic = np.array([[focal_length_x, 0, cx],
                        [0, focal_length_y, cy],
                        [0, 0, 1]])


    depth_reproj = pcd.project_to_depth_image(width, height, intrinsic, pose, depth_scale=1000.0, depth_max=5.0)
    depth_reproj = np.asarray(depth_reproj).astype(np.uint16)
    depthp_file = os.path.join(depthp_path, extr.name[:-4]+'.png')
    cv2.imwrite(depthp_file, depth_reproj)