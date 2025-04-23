import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from core.options import Options

import kiui
def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]
    # import ipdb
    # ipdb.set_trace()
    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

class GaussianRenderer:
    def __init__(self, opt: Options):
        
        self.opt = opt
        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        
        # intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[2, 3] = 1
        
    def render(self, gaussians, cam_view, cam_view_proj, cam_pos, bg_color=None, scale_modifier=1):
        # gaussians: [B, N, 14]
        # cam_view, cam_view_proj: [B, V, 4, 4]
        # cam_pos: [B, V, 3]

        device = gaussians.device
        B, V = cam_view.shape[:2]

        # loop of loop...
        images = []
        alphas = []
        depths=[]
        normals=[]
        for b in range(B):

            # pos, opacity, scale, rotation, shs
            means3D = gaussians[b, :, 0:3].contiguous().float()
            opacity = gaussians[b, :, 3:4].contiguous().float()
            scales = gaussians[b, :, 4:7].contiguous().float()
            rotations = gaussians[b, :, 7:11].contiguous().float()
            rgbs = gaussians[b, :, 11:].contiguous().float() # [N, 3]
            # import ipdb
            # ipdb.set_trace()
            for v in range(V):
                
                # render novel views
                view_matrix = cam_view[b, v].float()
                view_proj_matrix = cam_view_proj[b, v].float()
                campos = cam_pos[b, v].float()

                raster_settings = GaussianRasterizationSettings(
                    image_height=self.opt.output_size,
                    image_width=self.opt.output_size,
                    tanfovx=self.tan_half_fov,
                    tanfovy=self.tan_half_fov,
                    bg=self.bg_color if bg_color is None else bg_color,
                    scale_modifier=scale_modifier,
                    viewmatrix=view_matrix,
                    projmatrix=view_proj_matrix,
                    sh_degree=0,
                    campos=campos,
                    prefiltered=False,
                    debug=False,
                )

                rasterizer = GaussianRasterizer(raster_settings=raster_settings)

                # Rasterize visible Gaussians to image, obtain their radii (on screen).
                rendered_image, radii, rendered_depth, rendered_alpha= rasterizer(
                    means3D=means3D,
                    means2D=torch.zeros_like(means3D, dtype=torch.float32, device=device),
                    shs=None,
                    colors_precomp=rgbs,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=None,
                )

                # if self.opt.lambda_normal>0:
                # rotations_mat = build_rotation(rotations)
                # scales = scales
                # min_scales = torch.argmin(scales, dim=1)
                # indices = torch.arange(min_scales.shape[0])
                # normal = rotations_mat[indices, :, min_scales]
                # # convert normal direction to the camera; calculate the normal in the camera coordinate
                # view_dir = means3D -cam_pos[b, v]
                # normal   = normal * ((((view_dir * normal).sum(dim=-1) < 0) * 1 - 0.5) * 2)[...,None]

                # R_w2c =cam_view[b, 0].float().T [:3,:3]
                # R_w2c =view_matrix.T [:3,:3]
                # normal[..., ] *= -1
                # normal = (R_w2c @ normal.transpose(0, 1)).transpose(0, 1)
                # if self.opt.lambda_normal>0:
                #     render_normal, _,_,_= rasterizer(
                #         means3D = means3D,
                #         means2D=torch.zeros_like(means3D, dtype=torch.float32, device=device),
                #         shs = None,
                #         colors_precomp = -normal.float(),
                #         opacities = opacity,
                #         scales = scales,
                #         rotations = rotations,
                #         cov3D_precomp=None)
                #     render_normal = F.normalize(render_normal, dim = 0)                                                                                                                                                   
                    # return_dict.update({'render_normal': render_normal})
                    # import torchvision   
                    # vis_normal=(render_normal +1)/2.0
                    # normals.append( render_normal )
                # torchvision.utils.save_image(vis_normal[None],'r_normal.jpg')

                rendered_image = rendered_image.clamp(0, 1)
                images.append(rendered_image)
                alphas.append(rendered_alpha)
                # depths.append(rendered_depth)
         
        images = torch.stack(images, dim=0).view(B, V, 3, self.opt.output_size, self.opt.output_size)
        alphas = torch.stack(alphas, dim=0).view(B, V, 1, self.opt.output_size, self.opt.output_size)
        # depths = torch.stack(depths, dim=0).view(B, V, 1, self.opt.output_size, self.opt.output_size)
        # if self.opt.lambda_normal>0:  
        #     normals = torch.stack(normals, dim=0).view(B, V, 3, self.opt.output_size, self.opt.output_size)

        results= {
            "image": images, # [B, V, 3, H, W]
            # "depth": depths, # [B, V, 3, H, W]
            "alpha": alphas, # [B, V, 1, H, W]
        }
        if self.opt.lambda_normal>0:  
            results["normal"]= normals   # [B, V, 3, H, W]

        return results


    def save_ply(self, gaussians, path, compatible=True):
        # gaussians: [B, N, 14]
        # compatible: save pre-activated gaussians as in the original paper

        assert gaussians.shape[0] == 1, 'only support batch size 1'

        from plyfile import PlyData, PlyElement
        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

        # prune by opacity
        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        # invert activation to make it compatible with the original ply format
        if compatible:
            opacity = kiui.op.inverse_sigmoid(opacity)
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / 0.28209479177387814

        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        l = ['x', 'y', 'z','nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotations.shape[1]):
            l.append('rot_{}'.format(i))

        # dtype_full = [(attribute, 'f4') for attribute in l]

        # elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        # elements[:] = list(map(tuple, attributes))
        # el = PlyElement.describe(elements, 'vertex')

        # PlyData([el]).write(path)

        rotations_ = gaussians[:, mask, 7:11].float()
        rotations_mat_ = build_rotation(rotations_[0])
        scales_ = gaussians[0, mask, 4:7].float()
        min_scales = torch.argmin(scales_, dim=1)
        indices = torch.arange(min_scales.shape[0])
        normals = rotations_mat_[indices, :, min_scales].detach().cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in l] + [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        elements = np.empty(xyzs.shape[0], dtype=dtype_full)

        attributes = np.concatenate((xyzs,normals, f_dc, opacities, scales, rotations), axis=1)
        # if self.fea_dim > 0:
        #     feature = self.feature.detach().cpu().numpy()
        #     attributes = np.concatenate((attributes, feature), axis=1)
        colors = torch.clamp_min(torch.tensor(0.28209479177387814 * f_dc) + 0.5, 0.0)
        colors = (colors * 255).int().numpy()
        attributes = np.concatenate((attributes, colors), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


    def load_ply(self, path, compatible=True):

        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        print("Number of points at loading : ", xyz.shape[0])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
          
        gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
        gaussians = torch.from_numpy(gaussians).float() # cpu

        if compatible:
            gaussians[..., 3:4] = torch.sigmoid(gaussians[..., 3:4])
            gaussians[..., 4:7] = torch.exp(gaussians[..., 4:7])
            gaussians[..., 11:] = 0.28209479177387814 * gaussians[..., 11:] + 0.5

        return gaussians