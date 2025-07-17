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

import torch
from scene import Scene, DeformModel
import os
import json
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
import open3d as o3d
from scene.app_model import AppModel
import copy
from collections import deque
import colorsys
from PIL import Image
from sklearn.decomposition import PCA
import torch.nn.functional as F
# def post_process_mesh(mesh, min_len=1000):
#     """
#     Post-process a mesh to filter out floaters and disconnected parts
#     """
#     import copy
#     cluster_to_keep=1
#     print("post processing the mesh to have {} clusters cluster_to_kep".format(cluster_to_keep))
#     mesh_0 = copy.deepcopy(mesh)
#     with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
#             triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

#     triangle_clusters = np.asarray(triangle_clusters)
#     cluster_n_triangles = np.asarray(cluster_n_triangles)
#     cluster_area = np.asarray(cluster_area)
#     n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
#     n_cluster = max(n_cluster, min_len) # filter meshes smaller than 50
#     triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
#     mesh_0.remove_triangles_by_mask(triangles_to_remove)
#     mesh_0.remove_unreferenced_vertices()
#     mesh_0.remove_degenerate_triangles()
#     print("num vertices raw {}".format(len(mesh.vertices)))
#     print("num vertices post {}".format(len(mesh_0.vertices)))
#     return mesh_0

def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def feature_to_rgb(features):
    # Input features shape: (16, H, W)
    
    # Reshape features for PCA
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T

    # Apply PCA and get the first 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.cpu().numpy())

    # Reshape back to (H, W, 3)
    pca_result = pca_result.reshape(H, W, 3)

    # Normalize to [0, 255]
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min() + 1e-5)

    rgb_array = pca_normalized.astype('uint8')

    return rgb_array

def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)           # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5       # Alternate between 0.5 and 1.0
    l = 0.5

    
    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3, ), dtype=np.uint8)
    if id==0:   #invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb

def visualize_obj(objects):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask

def clean_mesh(mesh, min_len=1000):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_len
    mesh_0 = copy.deepcopy(mesh)
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    return mesh_0

def get_depth3(view, normgt, depth1):
    try:
        c2w = (view.world_view_transform.T).inverse()

        W, H = view.image_width, view.image_height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W) / 2],
            [0, H / 2, 0, (H) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        projection_matrix = c2w.T @ view.full_proj_transform
        intrins = (projection_matrix @ ndc2pix)[:3,:3].T
        
        grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
        points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
        rayd = (points @ intrins.inverse().T).reshape(H,W,3).permute(2,0,1)
        rayd = rayd / (rayd.norm(dim=0, keepdim=True) + 1e-5)
        costhe = (rayd*(-normgt)).sum(0)
        depth3 = depth1 * costhe * costhe
        return depth3
    except:
        return torch.zeros_like(depth1)

def render_set(model_path, name, iteration, views, scene, gaussians, pipeline, background, deform, 
               app_model=None, max_depth=5.0, volume=None, use_depth_filter=False):
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    print(gts_path)
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    render_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_depth")
    render_depth_path_opt = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_depth_opt")
    render_depth_path_m = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_depth_metric")
    render_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_normal")
    gt_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_normal")

    makedirs(gts_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(render_depth_path, exist_ok=True)
    makedirs(render_normal_path, exist_ok=True)
    makedirs(gt_normal_path, exist_ok=True)
    makedirs(render_depth_path_opt, exist_ok=True)
    makedirs(render_depth_path_m, exist_ok=True)

    fuse_path = os.path.join(model_path, name, "fuse_source")
    fuse_path_o = os.path.join(fuse_path, "objects_pred")
    fuse_path_gt = os.path.join(fuse_path, "gt_objects_color")
    fuse_path_f = os.path.join(fuse_path, "objects_feature16")
    fuse_path_all = os.path.join(fuse_path, "concat")

    makedirs(fuse_path_o, exist_ok=True)
    makedirs(fuse_path_gt , exist_ok=True)
    makedirs(fuse_path_f, exist_ok=True)
    makedirs(fuse_path_all, exist_ok=True)

    depths_tsdf_fusion = []
    # fourcc = cv2.VideoWriter.fourcc(*'DIVX') 
    # gt, _ = views[0].get_image()
    # size = (gt.shape[-1]*5,gt.shape[-2])
    # fps = float(5) if 'train' in render_path else float(1)
    # writer = cv2.VideoWriter(os.path.join(fuse_path,'result_ours.avi'), fourcc, fps, size)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt, _ = view.get_image()

        out = render(view, gaussians, pipeline, background)
        rendering_obj = out["render_object"].detach()
        fid = view.fid
        time_input = fid.unsqueeze(0).to('cuda')
        logits = deform.step(rendering_obj, time_input.detach()) 

        rendering = out["render"].clamp(0.0, 1.0)
        _, H, W = rendering.shape

        depth = out["plane_depth"].squeeze()
        
        gt_objects = view.objects.detach()
        depth_tsdf = depth.clone()

        depth = depth.detach().cpu().numpy()
        cv2.imwrite(os.path.join(render_depth_path_m, view.image_name + ".png"), (depth*1000).astype(np.uint16))
        depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
        depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)

        normal = out["depth_normal"].permute(1,2,0)
        # normal = normal/(normal.norm(dim=-1, keepdim=True)+1.0e-8)
        normal = normal.detach().cpu().numpy()
        normal = ((normal+1) * 127.5).astype(np.uint8).clip(0, 255)
        normalgt = ((view.normal.permute(1,2,0).cpu().numpy()+1) * 127.5).astype(np.uint8).clip(0, 255)

        # if name == 'test':
        torchvision.utils.save_image(gt.clamp(0.0, 1.0), os.path.join(gts_path, view.image_name + ".jpg"))
        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".jpg"))
        # else:
        #     rendering_np = (rendering.permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
        #     cv2.imwrite(os.path.join(render_path, view.image_name + ".jpg"), rendering_np)
        cv2.imwrite(os.path.join(render_depth_path, view.image_name + ".jpg"), depth_color)

        cv2.imwrite(os.path.join(render_normal_path, view.image_name + ".jpg"), normal)
        cv2.imwrite(os.path.join(render_normal_path, view.image_name + ".jpg"), normal)
        cv2.imwrite(os.path.join(gt_normal_path, view.image_name + ".jpg"), normalgt)

        if use_depth_filter:
            view_dir = torch.nn.functional.normalize(view.get_rays(), p=2, dim=-1)
            depth_normal = out["depth_normal"].permute(1,2,0)
            depth_normal = torch.nn.functional.normalize(depth_normal, p=2, dim=-1)
            dot = torch.sum(view_dir*depth_normal, dim=-1).abs()
            angle = torch.acos(dot)
            mask = angle > (80.0 / 180 * 3.14159)
            depth_tsdf[mask] = 0
        depths_tsdf_fusion.append(depth_tsdf.squeeze())

        depth_normal = out["depth_normal"]
        normal = out["rendered_normal"]
        wein = out["rendered_alpha"].detach()
        normal_error = (1 - (depth_normal * view.normal).sum(dim=0))[None]

        depth2 = out['plane_depth'].unsqueeze(0)
        temp_y = torch.zeros_like(view.normal)
        temp_y[1,:,:] = -torch.ones_like(view.normal[1,:,:]) #得到-y向量：[0,-1,0]

        temp_yn0 = (temp_y * view.normal).sum(0)
        temp_yn1 = (temp_y * normal).sum(0)
        mask0 = (temp_yn1 > temp_yn0).float().detach() 
        mask1 = (temp_yn0 <= 0.0).float().detach() 
        weid3 = out["rendered_alpha"].detach()
        depth3 = get_depth3(view, view.normal, depth2)

        mask4 = (normal_error > 0.1).detach()
        depth4 = depth3 * mask1 + (1.0-mask1)*(mask0*((1.0-weid3)*depth3 + weid3*depth2) + (1.0-mask0)*depth2) + 0.01

        deptht = torch.zeros_like(depth4)
        deptht[mask4.unsqueeze(0)] = depth4[mask4.unsqueeze(0)]
        deptht = deptht.squeeze().detach().cpu().numpy()

        depth_i2 = (deptht - deptht.min()) / (deptht.max() - deptht.min() + 1e-20)
        depth_i2 = (depth_i2 * 255).clip(0, 255).astype(np.uint8)
        depth_color_opt = cv2.applyColorMap(depth_i2, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(render_depth_path_opt, view.image_name + ".jpg"), depth_color_opt)
        fid = view.fid
        time_input = fid.unsqueeze(0).to('cuda')
        objects = out["render_object"]
        spe_tint = deform.step(objects, time_input.detach())   

        pred_obj = torch.argmax(spe_tint,dim=0).detach()
        pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))
        Image.fromarray(pred_obj_mask).save(os.path.join(fuse_path_o, view.image_name + ".jpg"))
        
    if volume is not None:
        depths_tsdf_fusion = torch.stack(depths_tsdf_fusion, dim=0)
        for idx, view in enumerate(tqdm(views, desc="TSDF Fusion progress")):
            ref_depth = depths_tsdf_fusion[idx].cuda()

            if view.mask is not None:
                ref_depth[view.mask.squeeze() < 0.5] = 0
            ref_depth[ref_depth>max_depth] = 0
            ref_depth = ref_depth.detach().cpu().numpy()
            
            pose = np.identity(4)
            pose[:3,:3] = view.R.transpose(-1,-2)
            pose[:3, 3] = view.T
            color = o3d.io.read_image(os.path.join(render_path, view.image_name + ".jpg"))
            depth = o3d.geometry.Image((ref_depth*1000).astype(np.uint16))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1000.0, depth_trunc=max_depth, convert_rgb_to_intensity=False)
            volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(W, H, view.Fx, view.Fy, view.Cx, view.Cy),
                pose)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                 max_depth : float, voxel_size : float, use_depth_filter : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        deform = DeformModel()
        deform.load_weights(dataset.model_path)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]   
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        print(f"TSDF voxel_size {voxel_size}")
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=4.0*voxel_size,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), scene, gaussians, pipeline, background, deform, 
                       max_depth=max_depth, volume=volume, use_depth_filter=use_depth_filter)
            print(f"extract_triangle_mesh")
            mesh = volume.extract_triangle_mesh()

            path = os.path.join(dataset.model_path, "possion_mesh")
            os.makedirs(path, exist_ok=True)
            
            o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion.ply"), mesh, 
                                       write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
            mesh_post = post_process_mesh(mesh, 1)
            o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion_post.ply"), mesh_post, 
                                       write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), scene, gaussians, pipeline, background, deform)

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_depth", default=5.0, type=float)
    parser.add_argument("--voxel_size", default=0.008, type=float)
    parser.add_argument("--use_depth_filter", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    print(f"multi_view_num {model.multi_view_num}")
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.max_depth, args.voxel_size, args.use_depth_filter)