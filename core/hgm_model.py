import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kiui
from kiui.lpips import LPIPS
from networks import MultiHeadAttention, SparseConvNet
from core.unet import UNet,MixNet,UNet_feats
import spconv.pytorch as spconv
from core.options import Options
from core.gs import GaussianRenderer
import pytorch3d
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d import loss
from smplx import SMPL,SMPLX
import torchvision
import os
import smplx
import sys
import util.util as util
from safetensors.torch import load_file
import scipy.sparse as sp
from chumpy.utils import row, col
from networks.UNet import ResUNet
from torchvision import transforms

smpl_model=SMPL('assets/SMPL_NEUTRAL.pkl')
SMPL_PATH='XXX/body_models'

def grid_sample_2d(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1);
    iy = ((iy + 1) / 2) * (IH-1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().reshape(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().reshape(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().reshape(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().reshape(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.reshape(N, C, H, W) * nw.reshape(N, 1, H, W) + 
               ne_val.reshape(N, C, H, W) * ne.reshape(N, 1, H, W) +
               sw_val.reshape(N, C, H, W) * sw.reshape(N, 1, H, W) +
               se_val.reshape(N, C, H, W) * se.reshape(N, 1, H, W))

    return out_val

class HGM(nn.Module):
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()
        self.smpl_num= 6890
        self.smplx_num= 10475
        self.opt = opt
        self.code_dim=32
        self.voxel_size=[0.005, 0.005, 0.005],
        self.voxel_size=torch.tensor(self.voxel_size).cuda()
        self.xyzc_net = SparseConvNet(n_layers=4, in_dim=self.code_dim,out_dim=[32, 32, 32, 32])
        self.unet2 =ResUNet(out_ch=32) 
        if self.opt.smplx:
            self.c = nn.Embedding(self.smplx_num,code_dim)
            self.m_smplx = smplx.SMPLX( os.path.join(SMPL_PATH, 'smplx'), gender='male', use_pca=True, num_pca_comps=num_pca).cuda()
        else:
            self.c = nn.Embedding(self.smpl_num,code_dim)
        attn_n_heads=8
        self.xyzc_attn = MultiHeadAttention(attn_n_heads, code_dim, code_dim//attn_n_heads,
                                            code_dim//attn_n_heads, kv_dim= in_feat_ch, sum=False)
           
    
        self.unet = UNet_feats(
            9, 14, 
            down_channels=self.opt.down_channels,
            down_attention=self.opt.down_attention,
            mid_attention=self.opt.mid_attention,
            up_channels=self.opt.up_channels,
            up_attention=self.opt.up_attention,
            num_frames=self.opt.num_input_views,
        )
        self.res=MixNet(in_channels=259)
        self.conv = nn.Conv2d(14, 14, kernel_size=1) # NOTE: maybe remove it if train again
        self.gs = GaussianRenderer(opt)

        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)

    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict

    def entropy_loss(self,opacity):
        loss = (- opacity * torch.log(opacity + 1e-6) - \
            (1 - opacity) * torch.log(1 - opacity + 1e-6)).mean()
        return loss
    def prepare_default_rays(self, device, elevation=0):
        
        from kiui.cam import orbit_camera
        from core.utils import get_rays

        cam_poses = np.stack([
            orbit_camera(elevation, 0, radius=self.opt.cam_radius),
            orbit_camera(elevation, 120, radius=self.opt.cam_radius),
            orbit_camera(elevation, 240, radius=self.opt.cam_radius),
            orbit_camera(elevation, 270, radius=self.opt.cam_radius),
        ], axis=0) # [4, 4, 4]
        cam_poses = torch.from_numpy(cam_poses)
        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(cam_poses[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]
    
        return rays_embeddings
   
    def get_reprojection(self,points3d, intrinsics ,rotation_mtx , tvec) :
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
    def get_reprojection_batch(self,points3d, intrinsics ,w2c) :
        """
        Projects the 3d points back into the image and computes the residuals between each pair of points2d[i] and its
        corresponding reprojected point from points3d[i]


        Args:
            points2d: B*N x 2 array of 2d image points
            points3d: B*N x 3 arrya of 3d world points where points3d[i] coresponds to points2d[i] when reprojected.
            intrinsics: B*3 x 3 camera intrinsic matrix
            rotation_mtx: B*3 x 3 rotation matrix
            tvec: B*3-dim rotation vector

        Returns:
            N array of residuals which is the euclidean distance between the points2d and their reprojected points.

        """
        # residuals = np.zeros(points2d.shape[0])
        """ YOUR CODE HERE """
        # trans = torch.hstack([rotation_mtx, tvec]).float()
        trans=w2c
        points3d_homo = torch.cat([points3d, torch.ones( (points3d.shape[0],points3d.shape[1], 1)).cuda() ] ,dim=-1).float()
        points2d_re = intrinsics.float()[:,:,None] @( trans[:,:,None] @ points3d_homo[:,None,:,:,None]) 
        points2d_re = points2d_re[:,:,:,:2, 0] / points2d_re[:,:,:,2:3, 0]
        # points2d_re=points2d_re .permute(-1,-2)
        return points2d_re

    def normalize(self, pixel_locations, h, w):
        resize_factor = torch.tensor([w-1., h-1.]).to(pixel_locations.device)[None, None, :]
        normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.  # [n_views, n_points, 2]
        return normalized_pixel_locations
    
    def get_grid_coords(self, pts, out_sh, bounds,vol_size):
        # convert xyz to the voxel coordinate dhw
        dhw = pts[..., [2, 1, 0]]
        min_dhw = bounds[:, 0, [2, 1, 0]]
        dhw = dhw - min_dhw[:, None, None]
        dhw = dhw / vol_size
        # convert the voxel coordinate to [-1, 1]

        out_sh = out_sh.to(dhw)
        dhw = dhw / out_sh * 2 - 1
        # convert dhw to whd, since the occupancy is indexed by dhw
        grid_coords = dhw[..., [2, 1, 0]]
        return grid_coords

    def forward_gaussians(self, images,xyz,data):
        # images: [B, 4, 9, H, W]
        # return: Gaussians3: unet [B, dim_t] Gaussians2: smpl [B, dim_t] Gaussians: merged [B, dim_t]

        B, V, C, H, W = images.shape

        images = images.view(B*V, C, H, W)
        if 'smpl_scale' in data:
            xyz=xyz*data['smpl_scale'][:,None,None]

        if 'scale' in data:
            xyz=xyz*1.5/data['scale'][:,None,None]


        x ,feats= self.unet(images) # [B*4, 14, h, w]
        x = self.conv(x) # [B*4, 14, h, w]
        x = x.reshape(B, self.opt.num_input_views, 14, self.opt.splat_size, self.opt.splat_size)
        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
        pos= self.pos_act(x[..., 0:3]) # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])
        gaussians3 = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
      
        if self.opt.smplx:
            code = self.c(torch.arange(0, self.smplx_num).to(images.device))
        else:
            code = self.c(torch.arange(0, self.smpl_num).to(images.device))
        
        code_query = code.unsqueeze(1)

        new_base=new_base.float()
        interval=data['total_cam_num'][0]//self.opt.num_input_views
        smpl_pixel_locations=self.get_reprojection_batch(new_base,torch.roll(data['Ks'],data['shift'].item(),1) [:,::interval],torch.roll(data['cam_view'].transpose(2,3)[:,:,:3,:],data['shift'].item(),1) [:,::interval] )
        smpl_normalized_pixel_locations = torch.clamp(self.normalize(smpl_pixel_locations, self.opt.input_size, self.opt.input_size,) ,-1,1)  # [n_views, n_rays, n_samples, 2]


        features=self.unet2( images)

        smpl_feat_sampled = grid_sample_2d(features, smpl_normalized_pixel_locations.flatten(0,1) [:,None] )

        smpl_feat_sampled=smpl_feat_sampled.view(B,V,features.shape[1],-1 ).permute(0,-1,1,2).flatten(0,1)
        n_points=new_base.shape[1]
        code_query=code_query.repeat(B,1,1)
        smpl_feat_sampled = self.xyzc_attn(code_query, smpl_feat_sampled, smpl_feat_sampled)[0].squeeze() .view(B, n_points,-1)

        new_input=torch.cat([new_base.float() ,smpl_feat_sampled ],dim=-1)
        y=self.net(new_input.permute(0,2,1).clone() )
        y=y.permute(0,2,1)
    
        pos_=self.pos_act (new_base+self.pos_act(y[..., 0:3]))
        opacity_ = self.opacity_act(y[..., 3:4])
        scale_ = self.scale_act(y[..., 4:7])
        rotation_ = self.rot_act(y[..., 7:11])
        rgbs_ = self.rgb_act(y[..., 11:])        
        gaussians2=torch.cat([pos_, opacity_, scale_, rotation_, rgbs_], dim=-1) # [B, N, 14]

     
    
        batch_size= new_base.shape[0]

        new_base=gaussians2[:,:,:3]

        min_xyz = torch.min(new_base, dim=1)[0]-0.05
        max_xyz = torch.max(new_base, dim=1)[0]+0.05

        dhw = new_base[:,: ,[2, 1, 0]]
        min_dhw = min_xyz[:, [2, 1, 0] ]
        max_dhw = max_xyz[ :,[2, 1, 0]]
        coord = torch.round((dhw - min_dhw) / self.voxel_size)


        sh = coord.shape  # smpl coordinate
        idx = [torch.full([sh[1]], i) for i in range(sh[0])]

        idx = torch.cat(idx).to(coord)
        coord =coord.view(-1, sh[-1] )
        coord = torch.cat([idx[:, None], coord], dim=1)


        bounds = torch.stack([min_xyz, max_xyz], axis=0).permute(1,0,2)
        
        # construct the output shape
        out_sh = torch.ceil((max_dhw - min_dhw) / self.voxel_size).int()
        x = 32
        # mask sure the output size is N times of 32
        out_sh = (out_sh | (x - 1)) + 1

        grid_coords = torch.clamp(self.get_grid_coords(pos, out_sh,bounds,self.voxel_size),-1,1)


        interval=data['total_cam_num'][0]//self.opt.num_input_views
        
        new_pixel_locations=self.get_reprojection_batch(new_base,torch.roll(data['Ks'],data['shift'].item(),1) [:,::interval],torch.roll(data['cam_view'].transpose(2,3)[:,:,:3,:],data['shift'].item(),1) [:,::interval] )

        new_normalized_pixel_locations = torch.clamp( self.normalize(new_pixel_locations, self.opt.input_size, self.opt.input_size,) ,-1,1)  # [n_views, n_rays, n_samples, 2]
        
        smpl_feat_sampled = grid_sample_2d(features, new_normalized_pixel_locations.flatten(0,1) [:,None] )
 

        smpl_feat_sampled=smpl_feat_sampled.view(B,V,features.shape[1],-1 ).permute(0,-1,1,2).flatten(0,1)
        n_points=new_base.shape[1]
    

        new_feat_sampled = self.xyzc_attn(code_query, smpl_feat_sampled, smpl_feat_sampled)[0].squeeze() .view(B, n_points,-1)
)
        code = new_feat_sampled.squeeze().float()

        xyzc = spconv.SparseConvTensor(code, torch.clamp(coord.int(),0, 400), out_sh[0], batch_size)

        xyzc_features = self.xyzc_net(xyzc, grid_coords[0][:, None,None]) # [batchsize, channels, point_nums]
 
        xyzc_features = xyzc_features.permute(0, 2, 1).float()

        new=torch.cat([feats.reshape(xyzc_features.shape[0],feats.shape[1],-1).permute(0,2,1), xyzc_features ],dim=-1)

        new_input=torch.cat([pos,new] ,dim=-1)
        x=self.res(new_input.permute(0,2,1).clone() )
        x=x.permute(0,2,1)

        pos=pos+self.pos_act(x[..., 0:3]) # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])
        gaussians=torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1)

        return gaussians, gaussians2 ,gaussians3

    def forward(self, data):
        # data: output of the dataloader
        # return: loss
        results = {}
        loss = 0
        images = data['input'] # [B, 4, 9, h, W], input features
        smpl_model.to(images.device)
        # use the first view to predict gaussians
   
        smpl_data=data['xyz_path']
        poses= smpl_data['body_pose']
        B=poses.shape[0]
        trans=smpl_data['transl'].float()
    
        smpl_out=smpl_model(betas=smpl_data['betas'].reshape(B,10).float(),  body_pose= poses.reshape(B,69).float(),transl=trans.reshape(B,3).float(),global_orient= smpl_data['global_orient'].float() .reshape(B,3))
        xyz=smpl_out.vertices
        
        if 'global' in data:
            joints=smpl_out.joints*smpl_data['scale'].float()
            xyz=xyz-joints[0,0]
            xyz= xyz@data['global'].float()/smpl_data['scale'].float()

        gaussians,gaussians2,gaussians3= self.forward_gaussians(images,xyz,data) # [B, N, 14]
      
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)   
        results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
    
        results2 = self.gs.render(gaussians2, torch.roll(data['cam_view'],data['shift'].item(),1), torch.roll(data['cam_view_proj'],data['shift'].item(),1), torch.roll(data['cam_pos'],data['shift'].item(),1), bg_color=bg_color)
        pred_alphas2 = results2['alpha']
        pred_images2 = results2['image'] # [B, V, C, output_size, output_size]
        results['images_supp'] = pred_images2

        results3 = self.gs.render(gaussians3, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
        pred_images3 = results3['image'] # [B, V, C, output_size, output_size]
        results['images_supp3'] = pred_images3
        pred_alphas3 = results3['alpha']
       
        pred_images = results['image'] # [B, V, C, output_size, output_size]
        
        pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas
        gt_images = data['images_output'] # [B, V, 3, output_size, output_size], ground-truth novel views
        gt_masks = data['masks_output'] # [B, V, 1, output_size, output_size], ground-truth masks
        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)

        loss_mse = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)
        loss_mse_u=0
        loss_mse_s =0
        loss_lpips_u=0
        loss_lpips_s=0

      
        loss_mse_u = F.mse_loss(pred_images3, gt_images) + F.mse_loss(pred_alphas3, gt_masks)
        loss_mse_s = F.mse_loss(pred_images2, gt_images) + F.mse_loss(pred_alphas2, gt_masks)
        loss = loss + loss_mse+ loss_mse_u+loss_mse_s #+grid_loss

        if self.opt.lambda_normal > 0:
            og_normal=results['normal']
            or_normal_gt=data['normals']*2-1.0
            mask_out=data['masks_output'].repeat(1,1,3,1,1)
            or_normal_gt[mask_out==0]=0
            og_normal[mask_out==0]=0
            results['gt_normal_cam']=( or_normal_gt+1.0)/2.0
            results['normal']=(  og_normal+1.0)/2.0
            cos_normal = (1. - torch.sum( or_normal_gt* og_normal, dim =2))    .mean()
            l1_normal = torch.abs(og_normal- or_normal_gt).mean()
            normal_loss=(l1_normal+cos_normal) *self.opt.lambda_normal
            loss+=  normal_loss
            results['nor_loss']=normal_loss 
            
        if self.opt.lambda_depth> 0:
            real_depth= data['depths']
            pred_depth=results['depth']
            depth_loss =(torch.abs(real_depth - pred_depth) .mean()) *self.opt.lambda_depth
            loss+=depth_loss
            results['dep_loss']=depth_loss
      
        if self.opt.lambda_lpips > 0:
            loss_lpips = self.lpips_loss(     
                F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
           
            loss_lpips_u = self.lpips_loss(

                    F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                    F.interpolate(pred_images3.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
                ).mean()
            loss_lpips_s= self.lpips_loss(
                    F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                    F.interpolate(pred_images2.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
                ).mean()
            results['loss_lpips'] = loss_lpips
            loss = loss + self.opt.lambda_lpips * (loss_lpips+loss_lpips_u+loss_lpips_s)
     
        results['loss'] = loss

        # metric
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr

            psnr_u = -10 * torch.log10(torch.mean((pred_images3.detach() - gt_images) ** 2))
            results['psnr_u'] = psnr_u

        return results



