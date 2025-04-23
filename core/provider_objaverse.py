import os
import cv2
import random
import numpy as np
from pytorch3d.structures import Meshes
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import pickle
import kiui
from core.options import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter
from scipy.spatial.transform import Rotation as R_
import os.path as osp
import json

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
voxel_size=[0.005, 0.005, 0.005]

def transform_can_smpl(xyz, rot_ratio=0.0):
    center = np.array([0, 0, 0]).astype(np.float32)
    rot = np.array([[np.cos(0), -np.sin(0)], [np.sin(0), np.cos(0)]])
    rot = rot.astype(np.float32)
    trans = np.array([0, 0, 0]).astype(np.float32)
    if np.random.uniform() > rot_ratio:
        return xyz, center, rot, trans

    xyz = xyz.copy()

    # rotate the smpl
    rot_range = np.pi / 32
    t = np.random.uniform(-rot_range, rot_range)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    rot = rot.astype(np.float32)
    center = np.mean(xyz, axis=0)
    xyz = xyz - center
    xyz[:, [0, 2]] = np.dot(xyz[:, [0, 2]], rot.T)
    xyz = xyz + center

    # translate the smpl
    x_range = 0.05
    z_range = 0.025
    x_trans = np.random.uniform(-x_range, x_range)
    z_trans = np.random.uniform(-z_range, z_range)
    trans = np.array([x_trans, 0, z_trans]).astype(np.float32)
    xyz = xyz + trans

    return xyz, center, rot, trans
def prepare_input( seq_path, i):
    # read xyz, normal, color from the ply file
    vertices_path = os.path.join(seq_path, 'vertices',
                                    '{}.npy'.format(i))
    xyz = np.load(vertices_path).astype(np.float32)
    nxyz = np.zeros_like(xyz).astype(np.float32)

    # obtain the original bounds for point sampling
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    min_xyz[2] -= 0.05
    max_xyz[2] += 0.05
    can_bounds = np.stack([min_xyz, max_xyz], axis=0)

    # transform smpl from the world coordinate to the smpl coordinate
    params_path = os.path.join(seq_path, 'params',
                                '{}.npy'.format(i))
    params = np.load(params_path, allow_pickle=True).item()
    Rh = params['Rh']
    R = cv2.Rodrigues(Rh)[0].astype(np.float32)
    Th = params['Th'].astype(np.float32)
    xyz = np.dot(xyz - Th, R)

    # transformation augmentation
    xyz, center, rot, trans = transform_can_smpl(xyz)

    # obtain the bounds for coord construction
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    min_xyz[2] -= 0.05
    max_xyz[2] += 0.05
    bounds = np.stack([min_xyz, max_xyz], axis=0)

    cxyz = xyz.astype(np.float32)
    nxyz = nxyz.astype(np.float32)
    feature = np.concatenate([cxyz, nxyz], axis=1).astype(np.float32)

    # construct the coordinate
    dhw = xyz[:, [2, 1, 0]]
    min_dhw = min_xyz[[2, 1, 0]]
    max_dhw = max_xyz[[2, 1, 0]]
    coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

    # construct the output shape
    out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
    x = 32
    # mask sure the output size is N times of 32
    out_sh = (out_sh | (x - 1)) + 1

    return feature, coord, out_sh, can_bounds, bounds, Rh, Th, center, rot, trans
def get_reprojection(points3d, intrinsics ,rotation_mtx , tvec) :
    """
    Projects the 3d points back into the image and computes the residuals between each pair of points2d[i] and its
    corresponding reprojected point from points3d[i]


    Args:
        points2d: N x 2 array of 2d image points
        points3d: N x 3 arrya of 3d world points where points3d[i] coresponds to points2d[i] when reprojected.
        intrinsics: 3 x 3 camera intrinsic matrix
        rotation_mtx: 3 x 3 rotation matrix
        tvec: 3-dim rotation vector

    Returns:
        N array of residuals which is the euclidean distance between the points2d and their reprojected points.

    """
    trans = torch.hstack([rotation_mtx, tvec]).float()
    points3d_homo = torch.hstack([points3d, torch.ones( (points3d.shape[0], 1)).cuda() ]).float()
    points2d_re = intrinsics.float() @(trans @points3d_homo.T) 
    points2d_re = points2d_re[:2, :] / points2d_re[2:3, :]
    points2d_re=points2d_re .permute(1,0)
    return points2d_re

def getProjectionMatrix_refine(K: torch.Tensor, H, W, znear=0.001, zfar=1000):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    s = K[0, 1]
    P = torch.zeros(4, 4, dtype=K.dtype, device=K.device)
    z_sign = 1.0

    P[0, 0] = 2 * fx / W
    P[0, 1] = 2 * s / W
    P[0, 2] = -1 + 2 * (cx / W)

    P[1, 1] = 2 * fy / H
    P[1, 2] = -1 + 2 * (cy / H)

    P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    P[2, 3] = -1 * z_sign * 2 * zfar * znear / (zfar - znear) # z_sign * 2 * zfar * znear / (zfar - znear)
    P[3, 2] = z_sign

    return P


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


class Thuman2Dataset(Dataset):

    def __init__(self, opt: Options, training=True ,cam_radius=1.5):
        self.opt=opt
        self.ratio=self.opt.input_size/1024
        self.scale=2.0
        self.cam_radius=cam_radius
        self.opt = opt
        self.training = training
        
        self.split='train' if training else 'val'
        self.data_root='xxx'
        self.total_cam_num=12
        self.all_names=[]

        train_list=np.arange()
        test_list=np.arange()
        if training:
            self.all_names=train_list
        else:
            self.all_names=test_list

        print('Length',training,len(self.all_names))
        
    def __len__(self):
        return len(self.all_names)
    def __getitem__(self, idx):
        idx= idx % len(self.all_names)
        name = str(int(self.all_names[idx])).zfill(4)
        results = {}
        img_file=osp.join(self.data_root,'img',name)
        imgs_list=[]
        for num in range(self.total_cam_num):
            imgs_list.append(img_file+'_'+str(num).zfill(3) )

        images = []
        masks = []
        Ks=[]
        cam_poses = []
        w2c_oris=[]
        depths=[]
        normals=[]
        smpl_path='...'

        f=open(smpl_path,'rb')
        data=pickle.load(f)
        data['transl']=torch.zeros(3)
        data['v_shaped']=data['vertices']
        global_rs=data['global_orient'][0,0].cpu().detach()
        data['faces']=torch.from_numpy(data['faces'].astype(np.int8) )
        for key in data:
            data[key]=data[key].cpu().detach()
        interval= self.total_cam_num //self.opt.num_input_views

        depth_path=self.data_root+'/depth/'+name+'_000'
        smpl_scale= torch.tensor(float(os.listdir(depth_path)[0].split('.png')[0]))
        for vid in imgs_list:
            image_path = vid+'/0.jpg'
            msk_path =   image_path .replace('img','mask').replace('jpg','png')
            dep_root=self.data_root+'/depth/'+vid.split('/')[-1]
            dep_path=os.path.join(dep_root,os.listdir(dep_root)[0])
         
            # K_path = os.path.join(self.data_root,'parm',name,vid.split('.')[0]+'_intrinsic.npy')
            E_path =  image_path.replace('img','parm').replace('.jpg','_extrinsic.npy')
            K_path =  image_path.replace('img','parm').replace('.jpg','_intrinsic.npy')
            # E_path =  image_path.replace('img','parm').replace('0.jpg','0_1.json')
            E = np.load(E_path)
            K = np.load(K_path)
            K[:2] *= self.ratio

            image =  cv2.imread(image_path)
            depth=torch.from_numpy(cv2.imread(dep_path, cv2.IMREAD_UNCHANGED).astype(np.float32)) / (2**15)
            depth=depth[:,:,None]


            image = torch.from_numpy(image.astype(np.float32) / 255) # [512, 512, 4] in [0, 1]
            proj_matrix=getProjectionMatrix_refine(torch.from_numpy(K),self.opt.output_size,self.opt.output_size).transpose(0, 1).float()
            msk = torch.from_numpy(cv2.imread(msk_path)).float()
            msk[msk>0]=1
            R=E[:3,:3]
            T=E[:3,3]

            w2c_ori=getWorld2View2(np.transpose(R),T)
            c2w=torch.from_numpy(np.linalg.inv(w2c_ori))
        
            # TODO: you may have a different camera system
            # blender world + opencv cam --> opengl world & cam
            c2w[1] *= -1
            c2w[[1, 2]] = c2w[[2, 1]]
            c2w[:3, 1:3] *= -1 # invert up and forward direction

            # scale up radius to fully use the [-1, 1]^3 space!
            c2w[:3, 3] *= self.cam_radius /self.scale # 1.5 is the default scale
            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = msk.permute(2, 0, 1)
            depth=depth.permute(2, 0, 1)
            depth = depth * mask 
            depth[depth>0]=1/ depth[depth>0]
        
            image = image * mask + (1 - mask) # [3, 512, 512], to white bg
            image = image[[2,1,0]].contiguous() # bgr to rgb
            images.append(image)
            depths.append(depth)
            masks.append(mask[0])
            cam_poses.append(c2w)
            Ks.append(torch.from_numpy(K))
            w2c_oris.append(torch.from_numpy(w2c_ori))

        images = torch.stack(images, dim=0) # [V, C, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]
        w2c_oris=torch.stack(w2c_oris,dim=0)
        Ks=torch.stack(Ks,dim=0)
        depths=torch.stack(depths,dim=0)
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]
        shift=0
        images_=torch.roll(images,shift,0)
        images_input = F.interpolate(images_[::interval].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        cam_poses_input = cam_poses[::interval].clone()
        # data augmentation
        if self.opt.num_input_views>1:
            if self.training:
                # apply random grid distortion to simulate 3D inconsistency
                if random.random() < self.opt.prob_grid_distortion:
                    images_input[1:] = grid_distortion(images_input[1:])
                # apply camera jittering (only to input!)
                if random.random() < self.opt.prob_cam_jitter:
                    cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])
        images_input_ori = F.interpolate(images_[::interval].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # resize render ground-truth images, range still in [0, 1]
        results['images_output'] = torch.roll( F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False),shift,0) # [V, C, output_size, output_size]
        results['depths'] = F.interpolate(depths, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) [:,0][:,None]
        # results['depths'][results['depths']>0]=1/ results['depths'][results['depths']>0]
        results['depths']=  torch.roll(results['depths']/ self.scale*1.5,shift,0)
        results['masks_output'] = torch.roll(F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False),shift,0)# [V, 1, output_size, output_size]
        # build rays for input views
        rays_embeddings = []

        for i in range(self.opt.num_input_views):
            focal= K[1,1]
            fovy=2*math.atan(self.opt.input_size/(2*focal))/math.pi*180
            rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
        results['input'] = final_input

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]

        cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
        cam_pos = -cam_poses[:, :3, 3] # [V, 3]
        results['verts']=verts
        results['faces']=faces
        results['global']=global_rs
        results['xyz_path']= data
        results['Ks']= Ks
        results['shift']= shift
        results['smpl_scale']= smpl_scale
        results['cam_view'] = cam_view
        results['cam_view_proj'] =cam_view_proj
        results['cam_pos'] = cam_pos
        results['name']=name
        results['scale']=self.scale
        results['w2c_oris']=w2c_oris
        results['img_input']= images_input_ori 
        results['total_cam_num']=self.total_cam_num
        return results

