#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import cv2
import open3d as o3d
import yaml
from scipy.spatial.transform import Rotation as R
from glob import glob

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fx: float
    fy: float
    fid: np.array
    normal: np.array
    depth: np.array
    objects: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def load_poses(pose_path, num):
    poses = []
    with open(pose_path, "r") as f:
        lines = f.readlines()
    for i in range(num):
        line = lines[i]
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        c2w[:3,3] = c2w[:3,3] * 10.0
        w2c = np.linalg.inv(c2w)
        w2c = w2c
        poses.append(w2c)
    poses = np.stack(poses, axis=0)
    return poses

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, objects_folder, load_normal=True, load_depth=False):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            # assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        ti = np.array((float(idx)/ len(cam_extrinsics))).astype(np.float32) 

        image = Image.open(image_path)
        object_path = os.path.join(objects_folder, image_name + '.png')
        objects = np.array(Image.open(object_path)) if os.path.exists(object_path) else None

        if load_normal:
            normal_path = image_path.replace('/images/', '/normals/')[:-4]+".png"
            # normal1 = np.load(normal_path).astype(np.float32)
            normal1 = cv2.imread(normal_path).astype(np.float32)
            normal1 = (normal1 / 255.0) * 2.0 - 1.0 
            normal = np.zeros_like(normal1)
            normal[:,:,0] = normal1[:,:,2]*-1
            normal[:,:,1] = normal1[:,:,1]*-1
            normal[:,:,2] = normal1[:,:,0]*-1
            normal = normal[:,:,::-1]
        else:
            normal = None

        if load_depth:
            depth_path = image_path.replace('/images/', '/depth_p/')[:-4] + '.png'
            depth = np.array(cv2.imread(depth_path,-1))*0.001 if os.path.exists(depth_path) else None
            depth[depth < 0.01] = 0.0
        else:
            depth = None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX,
                              image_path=image_path, image_name=image_name, 
                              width=width, height=height, objects=objects, fx=focal_length_x, fy=focal_length_y, fid=ti, normal=normal, depth=depth)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

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

def readColmapSceneInfo(path, images, eval, llffhold=8, object_path=None):
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
    
    reading_dir = "images" if images == None else images
    object_dir = 'object_mask' if object_path == None else object_path
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), objects_folder=os.path.join(path, object_dir))
    # cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : int(x.image_name.split('_')[-1]))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    js_file = f"{path}/split.json"
    train_list = None
    test_list = None
    if os.path.exists(js_file):
        with open(js_file) as file:
            meta = json.load(file)
            train_list = meta["train"]
            test_list = meta["test"]
            print(f"train_list {len(train_list)}, test_list {len(test_list)}")

    if train_list is not None:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in train_list]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_list]
        print(f"train_cam_infos {len(train_cam_infos)}, test_cam_infos {len(test_cam_infos)}")
    elif eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    ply_path0 = os.path.join(path, "iphone_pointcloud.ply")
    if not os.path.exists(ply_path) or True:

        pcd_load = o3d.io.read_point_cloud(ply_path0)
        xyz = np.asarray(pcd_load.points)
        rgb = np.asarray(pcd_load.colors)

        storePly(ply_path, xyz, rgb)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, split, extension=".jpg"):
    cam_infos = []

    cam_path2 = os.path.join(path, 'reconworld_colmap_allframe_pose.txt')
    cams_colmap, idxs_colmap = load_tum_pose(cam_path2)

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        frames = contents["frames"]

        for idx, frame in enumerate(frames):

            focal_length_x,focal_length_y = float(frame['fl_x']), float(frame['fl_y'])
            height,width = float(frame['h']), float(frame['w'])
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)

            cam_name = os.path.join(path, split, frame["file_path"].split('/')[-1])

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGB"))/ 255.0
            arr = im_data
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            idx2 = int(frame["file_path"].split('/')[-1].split('.')[0].split('_')[-1])
            ti = np.array(float(idx2)).astype(np.float32)

            object_path = image_path.replace(split, 'object_mask')[:-4] + '.png'
            objects = np.array(Image.open(object_path)) if os.path.exists(object_path) else None

            # depth_path = image_path.replace(split, 'depth_p')[:-4] + '.png'
            # depth = np.array(cv2.imread(depth_path,-1))*0.001 if os.path.exists(depth_path) else None
            # depth[depth<0.01] = 0.0

            # normal_path = image_path.replace(split, 'normals')[:-4] + "_normal.npy"
            # normal1 = np.load(normal_path).astype(np.float32) if os.path.exists(normal_path) else None
            normal_path = image_path.replace('/images/', '/normals/')[:-4]+".png"
            # normal1 = np.load(normal_path).astype(np.float32)
            normal1 = cv2.imread(normal_path).astype(np.float32 )if os.path.exists(normal_path) else None
            normal1 = (normal1 / 255.0) * 2.0 - 1.0 
            if normal1 is not None:
                normal = np.zeros_like(normal1)
                normal[:,:,0] = normal1[:,:,2]*-1
                normal[:,:,1] = normal1[:,:,1]*-1
                normal[:,:,2] = normal1[:,:,0]*-1
                normal1 = normal[:,:,::-1]

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], objects=objects, fx=focal_length_x, fy=focal_length_y, fid=ti, normal=normal1, depth=depth))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms.json", white_background, 'images', extension)
    
    if not eval:
        test_cam_infos = []
        train_cam_infos.extend(test_cam_infos)
    else:
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms(path, "transforms.json", white_background, 'images', extension)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    ply_path0 = os.path.join(path, "point_cloud.ply")

    if not os.path.exists(ply_path) or True:
        pcd_load = o3d.io.read_point_cloud(ply_path0)
        xyz = np.asarray(pcd_load.points)
        rgb = np.asarray(pcd_load.colors)*255.0

        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}