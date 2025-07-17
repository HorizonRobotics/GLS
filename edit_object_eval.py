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

import torch
from scene import Scene, DeformModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import json
import open3d as o3d
import cv2
from sklearn.decomposition import PCA
import copy
from collections import deque
from autodecoder.model import Autoencoder
from scipy.spatial import ConvexHull, Delaunay
from render import feature_to_rgb, visualize_obj
import torch.nn.functional as F
from glob import glob
from ext.grounded_sam import grouned_sam_output, load_model_hf, select_obj_ioa
from segment_anything import sam_model_registry, SamPredictor

def points_inside_convex_hull(point_cloud, mask, remove_outliers=True, outlier_factor=1.0):
    """
    Given a point cloud and a mask indicating a subset of points, this function computes the convex hull of the 
    subset of points and then identifies all points from the original point cloud that are inside this convex hull.
    
    Parameters:
    - point_cloud (torch.Tensor): A tensor of shape (N, 3) representing the point cloud.
    - mask (torch.Tensor): A tensor of shape (N,) indicating the subset of points to be used for constructing the convex hull.
    - remove_outliers (bool): Whether to remove outliers from the masked points before computing the convex hull. Default is True.
    - outlier_factor (float): The factor used to determine outliers based on the IQR method. Larger values will classify more points as outliers.
    
    Returns:
    - inside_hull_tensor_mask (torch.Tensor): A mask of shape (N,) with values set to True for the points inside the convex hull 
                                              and False otherwise.
    """

    # Extract the masked points from the point cloud
    masked_points = point_cloud[mask].cpu().numpy()

    # Remove outliers if the option is selected
    if remove_outliers:
        Q1 = np.percentile(masked_points, 25, axis=0)
        Q3 = np.percentile(masked_points, 75, axis=0)
        IQR = Q3 - Q1
        outlier_mask = (masked_points < (Q1 - outlier_factor * IQR)) | (masked_points > (Q3 + outlier_factor * IQR))
        filtered_masked_points = masked_points[~np.any(outlier_mask, axis=1)]
    else:
        filtered_masked_points = masked_points

    # Compute the Delaunay triangulation of the filtered masked points
    delaunay = Delaunay(filtered_masked_points)

    # Determine which points from the original point cloud are inside the convex hull
    points_inside_hull_mask = delaunay.find_simplex(point_cloud.cpu().numpy()) >= 0

    # Convert the numpy mask back to a torch tensor and return
    inside_hull_tensor_mask = torch.tensor(points_inside_hull_mask, device='cuda')

    return inside_hull_tensor_mask

import open_clip
def cos_loss(network_output, gt):
    return F.cosine_similarity(network_output, gt, dim=-1).clip(0.0, 1.0)

def removal_setup(opt, model_path, iteration, view, gaussians, gaussians0, pipeline, background, deform, label, cameras_extent, removal_thresh, model, groundingdino_model, sam_predictor):
    model_clip, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16",  # e.g., ViT-B-16
            pretrained="laion2b_s34b_b88k",  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
    model_clip = model_clip.to("cuda")
    model_clip.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    text = tokenizer([label]).to("cuda")

    results0 = render(view, gaussians0, pipeline, background)
    rendering0 = results0["render"]
    rendering_obj0 = results0["render_object"]
    
    fid = view.fid
    time_input = fid.unsqueeze(0).to('cuda')
    logits = deform.step(rendering_obj0, time_input.detach()) 
    pred_obj = torch.argmax(logits,dim=0).detach()
    image = (rendering0.permute(1,2,0) * 255).cpu().numpy().astype('uint8')
    text_mask = grouned_sam_output(groundingdino_model, sam_predictor, label, image)
    selected_obj_ids = select_obj_ioa(pred_obj, text_mask)
    print(selected_obj_ids)

    with torch.no_grad():
        fid = view.fid
        time_input = fid.unsqueeze(0).to('cuda')
        logits3d = deform.step(gaussians._objects_dc.permute(2,0,1), time_input.detach())
        prob_obj3d = torch.softmax(logits3d, dim=0)
        mask = prob_obj3d[selected_obj_ids, :, :] > 0.2
        mask3d1 = mask.any(dim=0).squeeze()
        mask3d_convex = points_inside_convex_hull(gaussians._xyz.detach(), mask3d1, outlier_factor=1.0)
        mask3d1 = torch.logical_or(mask3d1, mask3d_convex)
        mask3d1 = mask3d1.float()[:, None, None]

        text_features = model_clip.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        objects2 = gaussians._objects_dc.permute(2,0,1).squeeze()
        objects2 = objects2 / (objects2.norm(dim=0, keepdim=True) + 1e-5)
        objects2 = objects2.permute(1,0)
        objects3 = model.decode(objects2).half()
        
        text_features = text_features.expand(objects3.shape)
        text_probs = cos_loss(objects3, text_features)
        mask3d = (text_probs>0.83*text_probs.max()).float()[:,None,None] * mask3d1

        mask3d = 1.0 - mask3d

    # fix some gaussians
    gaussians.removal_setup(opt,mask3d)

    return gaussians

def render_set(model_path, name, iteration, views, gaussians, gaussians0, pipeline, background, deform, view_idx, label, groundingdino_model, sam_predictor):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_select_obj_pip")
    makedirs(render_path, exist_ok=True)

    view = views[view_idx-1]
    img_name = view.image_name.split('.jpg')[0][6:] + f'_{label}.jpg'

    results = render(view, gaussians, pipeline, background)
    rendering = results["render"].squeeze().permute(1,2,0).cpu().numpy()
    rendering_obj = results["render_object"]
    similarity = results["rendered_alpha"].float().squeeze().cpu().numpy()

    results0 = render(view, gaussians0, pipeline, background)
    rendering0 = results0["render"].squeeze().permute(1,2,0).cpu().numpy()

    similarity_flat =  similarity.flatten()
    th_norm = np.percentile(similarity_flat, 90)
    th_mask = np.percentile(similarity_flat, 99)
    similarity[similarity<th_norm] = th_norm
    mask = similarity > th_mask

    similarity_map = cv2.normalize(similarity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    similarity_map = 255 - similarity_map
    heatmap = cv2.applyColorMap(similarity_map, cv2.COLORMAP_TURBO)
    heatmap = np.float32(heatmap) / 255

    alpha = 0.4 
    blender_rgbs = rendering0*alpha + heatmap*(1-alpha)
    cv2.imwrite(os.path.join(render_path, img_name), (blender_rgbs[:,:,::-1]*255).astype(np.uint8))

def render_set2(opt, model_path, name, iteration, views, gaussians0, pipeline, background, deform, view_idx, label, groundingdino_model, sam_predictor, max_depth=None, volume=None, use_depth_filter=False, threshold=0.1):
    pred_obj_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_pred")
    makedirs(pred_obj_path, exist_ok=True)

    view = views[view_idx-1]
    img_name = view.image_name.split('.jpg')[0][6:] + f'_{label}.png'

    results = render(view, gaussians0, pipeline, background)
    rendering = results["render"].detach()
    rendering_obj = results["render_object"].detach()
    
    fid = view.fid
    time_input = fid.unsqueeze(0).to('cuda')
    logits = deform.step(rendering_obj, time_input.detach()) 
    pred_obj = torch.argmax(logits,dim=0).detach()

    image = (rendering.permute(1,2,0) * 255).cpu().numpy().astype('uint8')
    text_mask = grouned_sam_output(groundingdino_model, sam_predictor, label, image)
    selected_obj_ids = select_obj_ioa(pred_obj, text_mask)
    print(selected_obj_ids)
    if len(selected_obj_ids) > 0:
        logits3d = deform.step(gaussians0._objects_dc.permute(2,0,1), time_input.detach())
        prob_obj3d = torch.softmax(logits3d,dim=0)
        mask = prob_obj3d[selected_obj_ids, :, :] > 0.2
        mask3d = mask.any(dim=0).squeeze()
        mask3d_convex = points_inside_convex_hull(gaussians0._xyz.detach(), mask3d, outlier_factor=1.0)
        mask3d = torch.logical_or(mask3d,mask3d_convex)
        mask3d = mask3d.float()[:,None,None]
        mask3d = 1.0 - mask3d

        # fix some gaussians
        gaussians0.removal_setup(opt,mask3d)
        results2 = render(view, gaussians0, pipeline, background)
        # rendering2 = results2["render"]
        rendering_obj2 = results2["render_object"]
        logits2 = deform.step(rendering_obj2, time_input.detach()) 
        prob2 = torch.softmax(logits2,dim=0)
        pred_obj_mask = prob2[selected_obj_ids, :, :] > 0.2
        pred_obj_mask = pred_obj_mask.any(dim=0)
    else:
        # pred_obj_mask = (results["rendered_alpha"] > 0.0).float()
        pred_obj_mask = torch.zeros_like(results["rendered_alpha"]).float()
        
    pred_obj_mask = (pred_obj_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
    
    Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, img_name))

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

def removal(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, opt : OptimizationParams, select_obj_id : int, removal_thresh : float,
                 max_depth : float, voxel_size : float, use_depth_filter : bool):
    with torch.no_grad():
        # 1. load gaussian checkpoint
        num_classes = 256
        print("Num classes: ",num_classes)

        deform = DeformModel()
        deform.load_weights(dataset.model_path)
        deform.deform.eval()

        bg_color = [1,1,1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # grounding-dino
        # Use this command for evaluate the Grounding DINO model
        # Or you can download the model by yourself
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

        # sam-hq
        sam_checkpoint = './ckpts/sam_vit_h_4b8939.pth'
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to("cuda:0")
        sam_predictor = SamPredictor(sam)

        ckpt_path='./autoencoder/ckpt/best_ckpt.pth'
        checkpoint = torch.load(ckpt_path)
        model = Autoencoder([256, 128, 64, 32, 16], [32, 64, 128, 256, 256, 512]).to("cuda:0")
        model.load_state_dict(checkpoint)
        model.eval()

        img_labels = ["00022_toaster.png"]
        
        gaussians0 = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians0, load_iteration=iteration, shuffle=False)

        for img_label in img_labels:
            il_name = img_label.split('/')[-1]
            view_idx = int(il_name.split('_')[0])
            label = il_name.split('_')[1][:-4]

            gaussians = GaussianModel(dataset.sh_degree)
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

            # 2. remove selected object
            gaussians = removal_setup(opt, dataset.model_path, scene.loaded_iter, scene.getTrainCameras()[view_idx-1], gaussians, gaussians0, pipeline, background, deform, label, scene.cameras_extent, removal_thresh, model, groundingdino_model, sam_predictor)
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, gaussians0, pipeline, background, deform, view_idx, label, groundingdino_model, sam_predictor)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_depth", default=10.0, type=float)
    parser.add_argument("--voxel_size", default=0.01, type=float)
    parser.add_argument("--use_depth_filter", action="store_true")

    parser.add_argument("--config_file", type=str, default="config/object_removal/bear.json", help="Path to the configuration file")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Read and parse the configuration file
    try:
        with open(args.config_file, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        exit(1)

    args.num_classes = config.get("num_classes", 200)
    args.removal_thresh = config.get("removal_thresh", 0.3)
    args.select_obj_id = config.get("select_obj_id", [0])

    # Initialize system state (RNG)
    safe_state(args.quiet)

    removal(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, opt.extract(args), args.select_obj_id, args.removal_thresh, args.max_depth, args.voxel_size, args.use_depth_filter)


