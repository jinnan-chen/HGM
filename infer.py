"""
Cleaned inference script for LGM-SMPL model
"""

import os
import glob
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision
import tqdm
import tyro
import imageio
from PIL import Image
from safetensors.torch import load_file
from copy import deepcopy
import trimesh
import open3d as o3d

# Third-party imports
import kiui
from kiui.op import recenter
from kiui.cam import orbit_camera
import rembg
import pytorch3d
import smplx
from accelerate import Accelerator
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from diffusers import ControlNetModel, DiffusionPipeline
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)
from pytorch3d.structures import Meshes

# Local imports
from core.options import AllConfigs, Options
from core.lgm_model import LGM_SMPL
from smplx import SMPL
from SMPLer.models.smpler import SMPLer
from SMPLer import config
from SMPLer.models.transformer_basics import TranformerConfig


# Constants
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
SMPL_PATH = ''
NUM_PCA = 45


class IoULoss(nn.Module):
    """IoU Loss for silhouette optimization"""
    
    def __init__(self, weight=None, reduction='mean'):
        super(IoULoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, pred, target):
        intersection = (pred * target).sum((2, 3)).view(-1)
        union = pred.sum((2, 3)).view(-1) + target.sum((2, 3)).view(-1) - intersection
        iou = (intersection / union).clamp(min=1e-6)
        loss = 1 - iou.mean() if self.reduction == 'mean' else 1 - iou.sum()
        return loss


def build_rotation(r):
    """Build rotation matrix from quaternion"""
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])
    q = r / norm[:, None]
    
    R = torch.zeros((q.size(0), 3, 3), device='cuda')
    
    r_q = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r_q * z)
    R[:, 0, 2] = 2 * (x * z + r_q * y)
    R[:, 1, 0] = 2 * (x * y + r_q * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r_q * x)
    R[:, 2, 0] = 2 * (x * z - r_q * y)
    R[:, 2, 1] = 2 * (y * z + r_q * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    
    return R


def fibonacci_sampling_on_sphere(num_samples=1):
    """Generate fibonacci spiral sampling on sphere"""
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians
    
    for i in range(num_samples):
        y = 1 - (i / float(num_samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    
    return np.array(points)


def generate_cameras(r, num_cameras=20, device='cuda:0', pitch=math.pi/8, use_fibonacci=False):
    """Generate camera poses"""
    def normalize_vecs(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True))
    
    t = torch.linspace(0, 1, num_cameras).reshape(-1, 1)
    pitch = torch.zeros_like(t) + pitch
    directions = 2 * math.pi
    yaw = math.pi + directions * t
    
    if use_fibonacci:
        cam_pos = fibonacci_sampling_on_sphere(num_cameras)
        cam_pos = torch.from_numpy(cam_pos).float().to(device) * r
    else:
        z = r * torch.sin(pitch)
        x = r * torch.cos(pitch) * torch.cos(yaw)
        y = r * torch.cos(pitch) * torch.sin(yaw)
        cam_pos = torch.stack([x, y, z], dim=-1).reshape(z.shape[0], -1).to(device)
    
    forward_vector = normalize_vecs(-cam_pos)
    up_vector = torch.tensor([0, 0, -1], dtype=torch.float, device=device).reshape(-1).expand_as(forward_vector)
    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))
    
    rotate = torch.stack((left_vector, up_vector, forward_vector), dim=-1)
    
    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = rotate
    
    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = cam_pos
    
    cam2world = translation_matrix @ rotation_matrix
    return cam2world


def resize_for_condition_image(input_image: Image, resolution: int):
    """Resize image for ControlNet conditioning"""
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


def setup_transforms():
    """Setup image transforms"""
    image_transforms = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    to_tensor = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    
    return image_transforms, to_tensor


def setup_models_and_devices(opt):
    """Setup models, device, and accelerator"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup accelerator
    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
    )
    
    # Setup evaluation metrics
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    
    # Setup SMPL models
    smpl_model = SMPL('assets/SMPL_NEUTRAL.pkl').cuda()
    m_smplx = smplx.SMPLX(
        os.path.join(SMPL_PATH, 'smplx'),
        gender='male',
        use_pca=True,
        num_pca_comps=NUM_PCA
    ).cuda()
    
    # Setup ControlNet pipeline
    controlnet = ControlNetModel.from_pretrained(
        'lllyasviel/control_v11f1e_sd15_tile',
        torch_dtype=torch.float16
    )
    
    pipe = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        custom_pipeline="stable_diffusion_controlnet_img2img",
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to('cuda')
    pipe.enable_xformers_memory_efficient_attention()
    pipe.safety_checker = lambda images, clip_input: (images, None)
    
    return device, accelerator, lpips, ms_ssim, smpl_model, m_smplx, pipe


def load_dataset(opt):
    """Load appropriate dataset based on configuration"""
    dataset_mapping = {
        'ntu_render': 'core.provider_objaverse:NtuDataset_render',
        'ntu_cam': 'core.provider_objaverse:NtuDataset',
        'thu2': 'core.provider_objaverse:Thuman2Dataset_reverse',
        'thu1': 'core.provider_objaverse:ObjaverseDataset',
        'custom': 'core.provider_objaverse:CustomDataset',
        'thu2.1': 'core.provider_objaverse:Thuman2Dataset'
    }
    
    if opt.dataset in dataset_mapping:
        module_path, class_name = dataset_mapping[opt.dataset].split(':')
        module = __import__(module_path, fromlist=[class_name])
        Dataset = getattr(module, class_name)
    else:
        raise ValueError(f"Unknown dataset: {opt.dataset}")
    
    return Dataset


def load_models(opt, device):
    """Load and setup main models"""
    # Load main model
    model = LGM_SMPL(opt).to(device)
    
    if opt.resume is not None:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')
        
        state_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict and state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
        
        print(f'[INFO] Loaded checkpoint from {opt.resume}')
    else:
        print(f'[WARN] Model randomly initialized, are you sure?')
    
    model.eval()
    
    # Load secondary model
    opt2 = deepcopy(opt)
    opt2.num_input_views = 2
    model2 = LGM_SMPL(opt2).to(device)
    
    ckpt2_path = ''
    ckpt2 = load_file(ckpt2_path, device='cuda')
    model2.load_state_dict(ckpt2, strict=False)
    model2.eval()
    
    return model, model2


def optimize_smpl_with_silhouette(model2, data, xyz, num_iterations=20):
    """Optimize SMPL parameters using silhouette loss"""
    # Setup renderer
    num_views = 12
    elev = torch.linspace(0, 0, num_views)
    azim = torch.linspace(0, 330, num_views)
    R, T = look_at_view_transform(dist=1.5, elev=elev, azim=azim)
    
    fov_angle = 2 * torch.atan(torch.tensor([256/409])) / torch.pi * 180
    cameras = FoVPerspectiveCameras(device='cuda', R=R, T=T, fov=fov_angle)
    
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    raster_settings_silhouette = RasterizationSettings(
        image_size=512,
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        faces_per_pixel=50,
    )
    
    lights = PointLights(device='cuda', location=[[0.0, 0.0, -3.0]])
    renderer_silhouette = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings_silhouette),
        shader=SoftSilhouetteShader()
    )
    
    # Setup optimization
    iou_loss = IoULoss()
    smpler = model2.smpler
    
    for param in smpler.parameters():
        param.requires_grad_(True)
    
    optimizer_smpl = torch.optim.Adam(smpler.parameters(), lr=1e-7, betas=(0.9, 0.999), amsgrad=True)
    scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_smpl, mode="min", factor=0.5, verbose=0, min_lr=1e-5
    )
    
    # Get ground truth mask
    bg_color = torch.ones(3, dtype=torch.float32, device='cuda')
    gaussians_, gaussians2_, gaussians3_ = model2.forward_gaussians_wild(data['input'], xyz)
    results = model2.gs.render(gaussians3_, data['cam_view'].cuda(), data['cam_view_proj'].cuda(), data['cam_pos'].cuda(), bg_color=bg_color)
    gt_msk = results['alpha'][0].detach()
    
    # Load SMPL face indices
    smpl_face_path = 'faces.npy'
    smpl_face = torch.from_numpy(np.load(smpl_face_path).astype(np.int32))
    
    xyz_opt = xyz
    
    # Optimization loop
    for ii in tqdm.tqdm(range(num_iterations)):
        optimizer_smpl.zero_grad()
        
        src_mesh = Meshes(
            verts=[xyz_opt[0]] * 12,
            faces=[smpl_face.cuda()] * 12,
        )
        
        images_predicted = renderer_silhouette(src_mesh, cameras=cameras, lights=lights)
        predicted_silhouette = images_predicted[..., 3]
        
        loss_silhouette = iou_loss(predicted_silhouette[:, None], gt_msk)
        chamfer = pytorch3d.loss.chamfer_distance(gaussians3_.detach()[:, :, :3], xyz_opt.float())[0]
        
        loss = loss_silhouette + chamfer
        loss.backward(retain_graph=True)
        optimizer_smpl.step()
        scheduler_smpl.step(loss)
        
        # Update SMPL parameters
        image = data['img_input'].flatten(0, 1)[:1]
        img = model2.smpler_transform(image)
        pred_smpl_dict = smpler(img)[-1]
        dicts = smpler.smpl_handler(pred_smpl_dict['theta'], pred_smpl_dict['beta'], theta_form='rot-matrix', cam=pred_smpl_dict['cam'])
        
        pred_vertices = dicts['vertices']
        pred_vertices = pred_vertices - dicts['smpl_joints'][:, 0]
        xyz = pred_vertices * 0.75
        xyz[:, :, 1:] = -xyz[:, :, 1:]
        xyz_opt = xyz
    
    return xyz_opt


def rgbd_to_mesh(data, images, depths, c2ws, fov, mesh_path, cam_elev_thr=0):
    """Convert RGBD images to mesh using TSDF fusion"""
    voxel_length = 3.0 / 512.0
    sdf_trunc = 1 * 0.05
    color_type = o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=color_type,
    )
    
    for i in range(c2ws.shape[0]):
        camera_to_world = c2ws[i]
        world_to_camera = np.linalg.inv(camera_to_world)
        camera_position = camera_to_world[:3, 3]
        camera_elevation = np.rad2deg(np.arcsin(camera_position[2]))
        
        if camera_elevation < cam_elev_thr:
            continue
        
        color_image = o3d.geometry.Image(np.ascontiguousarray(images[i]))
        depth_image = o3d.geometry.Image(np.ascontiguousarray(depths[i]))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image, depth_image, depth_scale=1.0, depth_trunc=4.0, convert_rgb_to_intensity=False
        )
        
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
        fx = fy = images[i].shape[1] / 2. / np.tan(np.deg2rad(fov / 2.0))
        cx = cy = images[i].shape[1] / 2.
        h = images[i].shape[0]
        w = images[i].shape[1]
        
        camera_intrinsics.set_intrinsics(w, h, fx, fy, cx, cy)
        volume.integrate(rgbd_image, camera_intrinsics, world_to_camera)
    
    fused_mesh = volume.extract_triangle_mesh()
    
    # Clean up mesh
    triangle_clusters, cluster_n_triangles, cluster_area = fused_mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < 200
    fused_mesh.remove_triangles_by_mask(triangles_to_remove)
    fused_mesh.remove_unreferenced_vertices()
    
    fused_mesh = fused_mesh.filter_smooth_simple(number_of_iterations=5)
    fused_mesh = fused_mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(mesh_path, fused_mesh)


def process_data_item(data, model, model2, opt, device, lpips, ms_ssim, pipe):
    """Process a single data item"""
    print(f"Processing: {data['name'][0]}")
    
    # Save input image
    os.makedirs(os.path.join(opt.workspace, 'inputs'), exist_ok=True)
    torchvision.utils.save_image(
        data['img_input'][0],
        os.path.join(opt.workspace, 'inputs', str(data['name'][0]) + 'ori.jpg')
    )
    
    # Get SMPL vertices
    xyz = get_smpl_vertices(data, opt, model2)
    
    # Process with model
    with torch.no_grad():
        if not opt.fitted:
            out = model(data)
            tensor_img = out['image'][:, 6] if opt.dataset == 'ntu_cam' else out['image'][:, 6]
        else:
            gt_images = data['images_output']
            gt_masks = data['masks_output']
            bg_color = torch.ones(3, dtype=torch.float32, device='cuda')
            gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)
            
            gaussians_, gaussians2_, gaussians3_ = model.forward_gaussians_wild(data['input'], xyz)
            results = model.gs.render(gaussians_, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
            tensor_img = results['image'][:, 6]
    
    # Generate enhanced image using ControlNet
    enhanced_image = generate_enhanced_image(tensor_img, pipe, opt)
    
    # Generate final results
    results = generate_final_results(data, model2, enhanced_image, xyz, opt)
    
    return results


def get_smpl_vertices(data, opt, model2):
    """Get SMPL vertices based on configuration"""
    

    # Use SMPL model parameters
    smpl_data = data['xyz_path']
    poses = smpl_data['body_pose']
    B = poses.shape[0]
    trans = smpl_data['transl'].float()
    
    if opt.smplx:
        xyz = data['smplx_xyz']
    else:
        from smplx import SMPL
        smpl_model = SMPL('assets/SMPL_NEUTRAL.pkl').cuda()
        smpl_out = smpl_model(
            betas=smpl_data['betas'].reshape(B, 10),
            body_pose=poses.reshape(B, 69).float(),
            transl=trans.reshape(B, 3),
            global_orient=smpl_data['global_orient'].float().reshape(B, 3)
        )
        xyz = smpl_out.vertices
    
    # Apply transformations
    if 'global' in data:
        if opt.smplx and not '_' in data['name'][0]:
            # Handle SMPLX case
            pass
        else:
            joints = smpl_out.joints
            xyz = xyz - smpl_data['transl'].reshape(B, 1, 3)
            joints = joints - smpl_data['transl'].reshape(B, 1, 3)
            xyz = xyz - joints[0, 0]
            xyz = xyz @ data['global']

    # Apply scaling
    if 'smpl_scale' in data:
        xyz = xyz * data['smpl_scale'][:, None, None]
    
    if (not opt.fitted) and (not opt.smpler):
        if opt.dataset in ['ntu_cam', 'ntu_render', 'thu2', 'thu2.1', 'custom']:
            if 'scale' in data:
                xyz = xyz * 1.5 / data['scale'][:, None, None]
    
    # Add additional vertices if needed
    if opt.smpl_add and not opt.smplx:
        new = np.load('smpl_new_index.npy')
        add = xyz[:, new, :].mean(dim=2)
        
        smpl_face_path = 'faces.npy'
        smpl_face = torch.from_numpy(np.load(smpl_face_path).astype(np.int32))
        add_face = xyz[:, smpl_face, :].mean(dim=2)
        xyz = torch.cat([xyz, add, add_face], dim=1).cuda()
    
    return xyz


def generate_enhanced_image(tensor_img, pipe, opt):
    """Generate enhanced image using ControlNet"""
    torchvision.utils.save_image(tensor_img, opt.workspace + '/' + 'original.jpg')
    
    # Convert tensor to PIL Image
    tensor = tensor_img.cpu().clone().squeeze(0).permute(1, 2, 0)
    image = (tensor.numpy() * 255).astype(np.uint8)
    image = Image.fromarray(image)
    
    # Generate enhanced image
    re_image = pipe(
        prompt="best quality",
        negative_prompt="blur, lowres, bad anatomy, bad hands, cropped, worst quality",
        image=image,
        controlnet_conditioning_image=image,
        width=image.size[0],
        height=image.size[1],
        strength=0.2,
        generator=torch.manual_seed(22),
        num_inference_steps=100,
        guidance_scale=5.0,
    ).images[0]
    
    # Convert back to tensor
    back = torch.from_numpy(np.array(re_image) / 255.0)[None].permute(0, 3, 1, 2).cuda()
    torchvision.utils.save_image(back, opt.workspace + '/' + 'enhanced.jpg')
    
    return TF.normalize(back, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)


def generate_final_results(data, model2, enhanced_image, xyz, opt):
    """Generate final results including videos and meshes"""
    # Prepare input
    if opt.dataset == 'ntu_cam':
        ray_emb = data['ray']
    else:
        ray_emb = torch.load('ray_emb.pt')
    
    ft = data['input'][:, :1, :3]
    ft_bk = torch.cat([ft, enhanced_image[None]], dim=1)
    images_input = torch.cat([ft_bk, ray_emb], dim=2)
    
    # Generate Gaussians
    if opt.mode == '5':
        gaussians, gaussians2, gaussians3 = model2.forward_gaussians_wild(images_input.float(), xyz)
    else:
        gaussians = model2.forward_gaussians_wild(images_input.float(), xyz)
    
    # Save Gaussians
    model2.gs.save_ply(gaussians, opt.workspace + '/' + data['name'][0] + '_gau.ply')
    
    # Render and compute metrics
    bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)
    results = model2.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
    
    # Generate 360-degree video
    generate_360_video(model2, gaussians, gaussians2 if opt.mode in ['2', '5'] else None, 
                      gaussians3 if opt.mode in ['2', '3', '5'] else None, opt, data)
    
    return results


def generate_360_video(model, gaussians, gaussians2, gaussians3, opt, data):
    """Generate 360-degree rotation video"""
    images = []
    images2 = []
    images3 = []
    
    elevation = 0
    azimuth = np.arange(0, 360, 2, dtype=np.int32)
    
    # Setup projection matrix
    tan_half_fov = np.tan(0.5 * np.deg2rad(60))
    proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device='cuda')
    proj_matrix[0, 0] = 1 / tan_half_fov
    proj_matrix[1, 1] = 1 / tan_half_fov
    proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[3, 2] = -(opt.zfar * opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[2, 3] = 1
    
    for azi in tqdm.tqdm(azimuth):
        cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).cuda()
        cam_poses[:, :3, 1:3] *= -1  # invert up & forward direction
        
        cam_view = torch.inverse(cam_poses).transpose(1, 2)
        cam_view_proj = cam_view @ proj_matrix
        cam_pos = -cam_poses[:, :3, 3]
        
        res = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)
        image = res['image']
        images.append((image.squeeze(1).permute(0, 2, 3, 1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
        
        if gaussians2 is not None:
            image2 = model.gs.render(gaussians2, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
            images2.append((image2.squeeze(1).permute(0, 2, 3, 1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
        
        if gaussians3 is not None:
            image3 = model.gs.render(gaussians3, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
            images3.append((image3.squeeze(1).permute(0, 2, 3, 1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
    
    # Save videos
    images = np.concatenate(images, axis=0)
    os.makedirs(opt.workspace, exist_ok=True)
    imageio.mimwrite(
        os.path.join(opt.workspace, f"{data['name'][0]}_{data['frame_id'][0]}_{model.mode}.mp4"),
        images, fps=30
    )
    
    if gaussians2 is not None:
        images2 = np.concatenate(images2, axis=0)
        imageio.mimwrite(
            os.path.join(opt.workspace, f"{data['name'][0]}_{data['frame_id'][0]}_{model.mode}_2.mp4"),
            images2, fps=30
        )
    
    if gaussians3 is not None:
        images3 = np.concatenate(images3, axis=0)
        imageio.mimwrite(
            os.path.join(opt.workspace, f"{data['name'][0]}_{data['frame_id'][0]}_{model.mode}_3.mp4"),
            images3, fps=30
        )

def compute_evaluation_metrics(pred_images, gt_images, lpips, ms_ssim, opt):
    """Compute PSNR, LPIPS, and SSIM metrics"""
    if opt.dataset in ['thu2', 'ntu_render']:
        psnr = -10 * torch.log10(torch.mean((pred_images[:, ::3].detach() - gt_images[:, ::3]) ** 2))
        _lpips = lpips(pred_images[0, ::3].detach(), gt_images[0, ::3])
        _ssim = ms_ssim(pred_images[0, ::3].detach(), gt_images[0, ::3])
    else:
        psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
        _lpips = lpips(pred_images[0, ::3].detach(), gt_images[0, ::3])
        _ssim = ms_ssim(pred_images[0, ::3].detach(), gt_images[0, ::3])
    
    return psnr, _lpips, _ssim


def main_process(opt: Options):
    """Main processing function"""
    # Setup
    device, accelerator, lpips, ms_ssim, smpl_model, m_smplx, pipe = setup_models_and_devices(opt)
    Dataset = load_dataset(opt)
    model, model2 = load_models(opt, device)
    
    # Setup dataset and dataloader
    test_dataset = Dataset(opt, training=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_dataloader = accelerator.prepare(test_dataloader)
    
    # Create workspace
    os.makedirs(opt.workspace, exist_ok=True)
    
    # Evaluation metrics
    total_psnr = 0
    total_lpips = 0
    total_ssim = 0
    n_valid = 0
    
    # Process each data item
    for num, data in enumerate(test_dataloader):
        try:
            print(f"Processing item {num}: {data['name'][0]}")
            
            # Get SMPL vertices
            xyz = get_smpl_vertices(data, opt, model2)
            if xyz is None:
                continue
            
            # Process with model
            with torch.no_grad():
                if not opt.fitted:
                    out = model(data)
                    if opt.dataset == 'ntu_cam':
                        tensor_img = out['image'][:, 5]
                    else:
                        tensor_img = out['image'][:, 6]
                else:
                    gt_images = data['images_output']
                    gt_masks = data['masks_output']
                    bg_color = torch.ones(3, dtype=torch.float32, device='cuda')
                    gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)
                    
                    gaussians_, gaussians2_, gaussians3_ = model.forward_gaussians_wild(data['input'], xyz)
                    results = model.gs.render(gaussians_, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
                    tensor_img = results['image'][:, 6]
                
                # Save original image
                torchvision.utils.save_image(tensor_img, os.path.join(opt.workspace, f"{data['name'][0]}_ori.jpg"))
                
                # Generate enhanced image using ControlNet
                enhanced_image = generate_enhanced_image(tensor_img, pipe, opt)
                
                # Prepare input for final processing
                if opt.dataset == 'ntu_cam':
                    ray_emb = data['ray']
                else:
                    ray_emb = torch.load('ray_emb.pt')
                
                ft = data['input'][:, :1, :3]
                ft_bk = torch.cat([ft, enhanced_image[None]], dim=1)
                images_input = torch.cat([ft_bk, ray_emb], dim=2)
                
                # Generate final Gaussians
                if opt.mode == '5':
                    gaussians, gaussians2, gaussians3 = model2.forward_gaussians_wild(images_input.float(), xyz)
                else:
                    gaussians = model2.forward_gaussians_wild(images_input.float(), xyz)
                
                # Save Gaussians
                model2.gs.save_ply(gaussians, os.path.join(opt.workspace, f"{data['name'][0]}_gau.ply"))
                
                # Render and compute metrics
                bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)
                results = model2.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
                
                gt_images = data['images_output']
                gt_masks = data['masks_output']
                gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)
                
                pred_images = results['image']
                psnr, _lpips, _ssim = compute_evaluation_metrics(pred_images, gt_images, lpips, ms_ssim, opt)
                
                total_psnr += psnr
                total_lpips += _lpips
                total_ssim += _ssim
                n_valid += 1
                
                # Print metrics
                avg_psnr = total_psnr / n_valid
                avg_ssim = total_ssim / n_valid
                avg_lpips = total_lpips / n_valid
                
                print(f"Current PSNR: {psnr:.4f}, SSIM: {_ssim:.4f}, LPIPS: {_lpips:.4f}")
                print(f"Average PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")
                
                # Generate 360-degree video
                if opt.mode == '5':
                    generate_360_video(model2, gaussians, gaussians2, gaussians3, opt, data)
                elif opt.mode in ['2', '3']:
                    generate_360_video(model2, gaussians, None, gaussians3, opt, data)
                else:
                    generate_360_video(model2, gaussians, None, None, opt, data)
                
                # Generate mesh
                generate_mesh_from_gaussians(model2, gaussians, opt, data)
                
        except Exception as e:
            print(f"Error processing {data['name'][0] if 'name' in data else 'unknown'}: {e}")
            continue
    
    # Final average metrics
    if n_valid > 0:
        print(f"\nFinal Results:")
        print(f"Average PSNR: {total_psnr / n_valid:.4f}")
        print(f"Average SSIM: {total_ssim / n_valid:.4f}")
        print(f"Average LPIPS: {total_lpips / n_valid:.4f}")
        print(f"Processed {n_valid} valid samples")


if __name__ == "__main__":
    # Parse command line arguments
    opt = tyro.cli(AllConfigs)
    
    # Run main processing
    with torch.no_grad():
        main_process(opt)
