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
from datetime import datetime
import torch
import random
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, lncc, get_img_grad_weight, inverse_depth_smoothness_loss,loss_cls_3d,norm_smoothness_loss,ScaleAndShiftInvariantLoss,LabelSmoothingCrossEntropy
from utils.graphics_utils import patch_offsets, patch_warp
from gaussian_renderer import render, network_gui
import sys, time
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state
import cv2
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, erode
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.app_model import AppModel
from scene.cameras import Camera
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import time
import torch.nn.functional as F

import colorsys
from sklearn.decomposition import PCA

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(22)

def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    rx, ry, rz = np.deg2rad(rotation_perturbation)
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)
    virtul_cam = Camera(100000, Rt[:3, :3].transpose(), Rt[:3, 3], cam.FoVx, cam.FoVy,
                        cam.image_width, cam.image_height,
                        cam.image_path, cam.image_name, 100000,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                        preload_img=False, data_device = "cuda")
    return virtul_cam

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
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

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

def ghost_loss_func(_rgb, bg, _acc, den_penalty=0.0):
    _bg = bg.detach()
    ghost_mask = torch.mean(torch.square(_rgb-_bg), 0)
    ghost_mask = torch.sigmoid(ghost_mask*-1.0) + den_penalty
    ghost_alpha = ghost_mask * _acc
    return torch.mean(torch.square(ghost_alpha))

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

from embedder import get_embedder

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # backup main code
    cmd = f'cp ./train.py {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./arguments {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./gaussian_renderer {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./scene {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./utils {dataset.model_path}/'
    os.system(cmd)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    deform = DeformModel()
    deform.train_setting(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_single_view_for_log = 0.0
    ema_multi_view_geo_for_log = 0.0
    ema_multi_view_pho_for_log = 0.0
    normal_loss, geo_loss, ncc_loss = None, None, None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    debug_path = os.path.join(scene.model_path, "debug")
    os.makedirs(debug_path, exist_ok=True)

    embed_fn, input_ch = get_embedder(24, input_dims=16)

    num_classes = opt.num_classes
    depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)
    cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')

    for iteration in range(first_iter, opt.iterations + 1):
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gt_image, gt_image_gray = viewpoint_cam.get_image()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg,
                            return_plane=True, return_depth_normal=True)
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        fid = viewpoint_cam.fid
        time_input = fid.unsqueeze(0).to('cuda')
        vel_duration = opt.deform_lr_max_steps
        objects = render_pkg["render_object"]
        objects2 = objects / (objects.norm(dim=0, keepdim=True) + 1e-5)
        spe_tint = deform.step(objects, time_input.detach()) 

        # Loss
        ssim_loss = (1.0 - ssim(image, gt_image))
        Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss = image_loss.clone()
        
        # scale loss
        if visibility_filter.sum() > 0:
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]
            loss += 10.0*min_scale_loss.mean()

        loss_obj = 0.0 
        depth_loss_s2 = 0.0
        if opt.vel_weight > 0.0:
            # pass
            gt_obj = viewpoint_cam.objects.long()
            obj_mask = (gt_obj > -1).detach()
            loss_obj = cls_criterion(spe_tint.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze()[obj_mask].mean()
            loss_obj = loss_obj / torch.log(torch.tensor(num_classes))  # normalize to (0,1)
            loss = loss + opt.vel_weight * loss_obj
            # if iteration % 1000 == 0:
            #     prob_obj3d = gaussians._objects_dc.permute(2,0,1)
            #     prob_obj3d = deform.step(prob_obj3d, torch.zeros_like(time_input).detach())
            #     prob_obj3d = torch.softmax(prob_obj3d,dim=0).squeeze().permute(1,0)
            #     loss_obj_3d = loss_cls_3d(gaussians._xyz.squeeze().detach(), prob_obj3d)
            #     loss = loss + 0.1*loss_obj_3d
            gt_language_feature, language_feature_mask, gt_language_feature2, gt_language_feature3 = viewpoint_cam.get_language_feature(language_feature_dir=dataset.lf_path)
            Ll1_fea = l1_loss(objects2*language_feature_mask, gt_language_feature*language_feature_mask)
            loss = loss + Ll1_fea

        r_depth1 = render_pkg['rendered_distance']
        depth1 = r_depth1.unsqueeze(0)
        depth2 = render_pkg['plane_depth'].unsqueeze(0)

        # just for blender
        # bg_color2 = torch.ones_like(image) if dataset.white_background else torch.zeros_like(image)
        # ghost_loss = ghost_loss_func(image, bg_color2, render_pkg["rendered_alpha"])
        # loss += 0.01*ghost_loss

        weightn = opt.single_view_weight
        depth_normal = render_pkg["depth_normal"]
        normal = render_pkg["rendered_normal"]
        wein = render_pkg["rendered_alpha"].detach()

        depth_loss_4 = 0.0
        depth4 = torch.zeros_like(depth2)
        if iteration > opt.single_view_weight_from_iter:
            normal_error = (1 - (depth_normal * viewpoint_cam.normal).sum(dim=0))[None]
            normal_loss = weightn * ((wein*normal_error).sum() / (wein.sum() + 1e-5))
            loss += normal_loss

            obj_mask2 = language_feature_mask.unsqueeze(0) * gt_language_feature2.unsqueeze(0)
            depth_loss_s2 = norm_smoothness_loss(render_pkg["depth_normal"].unsqueeze(0)*obj_mask2, gt_language_feature*obj_mask2)
            loss += 0.5*depth_loss_s2

            #depth2: pgsr采用的无偏深度
            temp_y = torch.zeros_like(viewpoint_cam.normal)
            temp_y[1,:,:] = -torch.ones_like(viewpoint_cam.normal[1,:,:]) #得到-y向量：[0,-1,0]
            temp_yn0 = (temp_y * viewpoint_cam.normal).sum(0)
            temp_yn1 = (temp_y * normal).sum(0)
            mask0 = (temp_yn1 > temp_yn0).float().detach() 
            mask1 = (temp_yn0 <= 0.0).float().detach() 
            weid3 = render_pkg["rendered_alpha"].detach()
            depth3 = get_depth3(viewpoint_cam, viewpoint_cam.normal, depth2)            
            depth4 = depth3 * mask1 + (1.0-mask1)*(mask0*((1.0-weid3)*depth3 + weid3*depth2) + (1.0-mask0)*depth2)

            # depth5 = viewpoint_cam.depth
            # maskd = (depth5 > 0.0).detach() & (normal_error <= 0.01).squeeze().detach()
            # depth_error = torch.abs((render_pkg['plane_depth'].squeeze()[maskd] - depth5[maskd]))
            # depth_error, _ = torch.topk(depth_error, int(0.95*depth_error.size(0)), largest=False)
            # depth_loss_5 = depth_error.mean() 

            loss += 0.01*depth_loss_4 #+ 0.3*depth_loss_5
 
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id,1)[0]]
            if iteration % 200 == 0:
                gt_img_show = ((gt_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                img_show = ((image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                normal_show = (((normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                depth_normal_show = (((depth_normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)

                depth = render_pkg['plane_depth'].squeeze().detach().cpu().numpy()
                depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                
                depth4_s = depth4.squeeze().detach().cpu().numpy()
                depth4_i = (depth4_s - depth4_s.min()) / (depth4_s.max() - depth4_s.min() + 1e-20)
                depth4_i = (depth4_i * 255).clip(0, 255).astype(np.uint8)
                depth4_color = cv2.applyColorMap(depth4_i, cv2.COLORMAP_JET)

                row0 = np.concatenate([gt_img_show, img_show, normal_show], axis=1)
                row1 = np.concatenate([depth4_color, depth_color, depth_normal_show], axis=1)

                pred_obj = torch.argmax(spe_tint,dim=0).detach()
                pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))
                gt_objects = viewpoint_cam.objects.detach()
                gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint8))
                rgb_mask = feature_to_rgb(render_pkg["render_object"].detach())
                row2 = np.concatenate([gt_rgb_mask, pred_obj_mask, rgb_mask], axis=1)

                image_to_show = np.concatenate([row0, row1, row2], axis=0)
                cv2.imwrite(os.path.join(debug_path, "%05d"%iteration + "_" + viewpoint_cam.image_name + ".jpg"), image_to_show)

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * image_loss.item() + 0.6 * ema_loss_for_log
            # ema_single_view_for_log = 0.4 * normal_loss.item()
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Class": f"{loss_obj:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)
                    
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                mask = (render_pkg["out_observe"] > 0) & visibility_filter
                gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], radii[mask])
                viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
                gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold, 
                                                opt.opacity_cull_threshold, scene.cameras_extent, size_threshold)

            # reset_opacity
            if iteration < opt.densify_until_iter:
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                if iteration < vel_duration:
                    deform.optimizer.step()
                    deform.optimizer.zero_grad()
                    deform.update_learning_rate(iteration)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            if iteration % 500 == 0:
                torch.cuda.empty_cache()
    
    torch.cuda.empty_cache()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-100)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[1_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
