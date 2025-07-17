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

import argparse
import math
import os
import time
from typing import Tuple

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import viser

import cv2
from utils.general_utils import safe_state
from autodecoder.model import Autoencoder

from ext.grounded_sam import grouned_sam_output, load_model_hf, select_obj_ioa
from segment_anything import sam_model_registry, SamPredictor
import open_clip

from copy import deepcopy
from gaussian_renderer import render
from scene.dataset_readers import CameraInfo
from gaussian_renderer import GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.camera_utils import Camera

from gpt import GPT
import torchvision
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

def cos_loss(network_output, gt):
    return F.cosine_similarity(network_output, gt, dim=-1).clip(0.0, 1.0)

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).astype(np.float32)

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).astype(np.float32)

target_text = ""
def main(args, gaussians, pipeline):
    torch.manual_seed(42)
    device = torch.device("cuda")
    loaded_iter = args.iteration
    print("Loading trained model at iteration {}".format(loaded_iter))

    gaussians.load_ply(os.path.join(args.model_path, "point_cloud",
                                                     "iteration_" + str(loaded_iter),
                                                     "point_cloud.ply"))

    model_clip, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16",  # e.g., ViT-B-16
        pretrained="laion2b_s34b_b88k",  # e.g., laion2b_s34b_b88k
        precision="fp16",
    )
    model_clip = model_clip.to("cuda")
    model_clip.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizer = open_clip.get_tokenizer('ViT-B-16')

    ae_model = Autoencoder(
            [256, 128, 64, 32, 16], 
            [32, 64, 128, 256, 256, 512]).to("cuda")
    checkpoint = torch.load('')
    ae_model.load_state_dict(checkpoint)
    ae_model.eval()

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gpt = GPT(api_key="", version="")

    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(camera_state: nerfview.CameraState, img_wh: Tuple[int, int], output_idx=0, prompt='', gpt_flag=False):
        width, height = img_wh
        c2w = np.array(camera_state.c2w).astype(np.float32)
        c2w = rot_phi(90/180.*np.pi) @ c2w
        # c2w = rot_theta(90/180.*np.pi) @ c2w

        K = np.array(camera_state.get_K(img_wh)).astype(np.float32)
        # viewmat = np.linalg.inv(c2w)
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        # c2w[:3, 1:3] *= -1
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)  
        # R = w2c[:3,:3]
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        fov_y = focal2fov(K[1,1], height)
        fov_x = focal2fov(K[0,0], width)
        camera_pack = Camera(colmap_id=0, R=R, T=T, FoVx=fov_x, FoVy=fov_y,
                            image_width=width, image_height=height, image_path='', 
                            image_name='viser_viewer_fake_img.jpg', uid=0, preload_img=False)

        out = render(camera_pack, gaussians, pipeline, background)
        
        render_rgbs = out["render"].clamp(0.0, 1.0).permute(1,2,0).cpu().numpy()
        render_rgbso = (render_rgbs*255).astype(np.uint8)
        
        render_depth = out['plane_depth'].squeeze()
        langs_fea = out["render_object"]
        render_normal = out["depth_normal"].permute(1,2,0)
        render_normal = render_normal.detach().cpu().numpy()
        render_normal = ((render_normal+1) * 127.5).clip(0, 255).astype(np.uint8)

        # Reshape and normalize the depth map for visualization
        render_depth = render_depth.detach().cpu().numpy()
        render_depth = np.clip(render_depth, 0.0, 20.0).astype(np.float32)
        inv_depth_map = cv2.normalize(render_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        color_depth_image = cv2.applyColorMap(inv_depth_map, cv2.COLORMAP_TURBO)
        color_depth_image = np.float32(color_depth_image) / 255

        if prompt != '' and  output_idx==2:
            global target_text

            if gpt_flag and target_text == "":
                torchvision.utils.save_image(out["render"].clamp(0.0, 1.0), "./temp.png")
                obj_caption_prompt_payload = gpt.payload_get_target_object_caption(img_path="./temp.png", require_prompt=prompt)
                gpt_text_response = gpt(payload=obj_caption_prompt_payload, verbose=True)
                if gpt_text_response is None:
                    print("Failed, terminate early")

                target_text = gpt_text_response[1:-1]

                print(target_text)

            #CLIP
            text = tokenizer([target_text]).to("cuda")
            objects2 = langs_fea / (langs_fea.norm(dim=0, keepdim=True) + 1e-5)
            objects2 = objects2.permute(1,2,0)
            objects3 = ae_model.decode(objects2).half()
            text_features = model_clip.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.unsqueeze(0).expand(objects3.shape)
            similarity = cos_loss(objects3, text_features).detach().cpu().numpy().astype(np.float32)

            similarity_flat =  similarity.flatten()
            th_norm = np.percentile(similarity_flat, 90)
            th_mask = np.percentile(similarity_flat, 99)
            similarity[similarity<th_norm] = th_norm
            mask = similarity > th_mask

            similarity_map = cv2.normalize(similarity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            similarity_map = 255 - similarity_map
            heatmap = cv2.applyColorMap(similarity_map, cv2.COLORMAP_TURBO)
            heatmap = np.float32(heatmap) / 255

            alpha = 0.4  # 设置透明度
            blender_rgbs = render_rgbs*alpha + heatmap*(1-alpha)
            render_objs = np.zeros_like(render_rgbs)
            render_objs[mask] = blender_rgbs[mask]
            render_objs[~mask] = 0.5*blender_rgbs[~mask]
            render_objs = (render_objs*255).astype(np.uint8)

        if output_idx==0:
            target_text = ""
            return render_rgbso
        elif output_idx==1:
            target_text = ""
            return color_depth_image
        elif output_idx==2:
            return render_objs
        elif output_idx==3:
            target_text = ""
            return render_normal

    server = viser.ViserServer(port=args.port, verbose=False)

    _ = nerfview.Viewer(
        server=server,
        render_fn=viewer_render_fn,
        mode="rendering",
    )

    print("Viewer running... Ctrl+C to exit.")
    time.sleep(1000000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30_000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--port", type=int, default=8090, help="port for the viewer server"
    )
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    gaussians = GaussianModel(model.extract(args).sh_degree)
    main(args, gaussians, pipeline.extract(args))
