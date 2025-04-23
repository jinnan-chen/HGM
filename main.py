import time
import random
import numpy as np
import torch
import trimesh
from core.options import AllConfigs
from core.hgm_model import HGM
from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_file
import smplx
from smplx import SMPL,SMPLX
import kiui
import tyro
import sys
import os


SMPL_PATH = 'xxx/body_models'

def main():    
    opt = tyro.cli(AllConfigs)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )
    seed=10
    torch.manual_seed(seed)
    # model
    model = HGM(opt)
    if opt.resume is not None:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')
        state_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict: 
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                else:
                    accelerator.print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            else:
                accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')

    from torch.utils.data import ConcatDataset

    if opt.dataset=='ntu_render':
        from core.provider_objaverse import NtuDataset_render as Dataset
    elif opt.dataset=='thu2':
        from core.provider_objaverse import Thuman2Dataset_reverse as Dataset

    train_dataset =Dataset(opt,training=True)
    test_dataset =Dataset(opt, training=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))
    total_steps = opt.num_epochs * len(train_dataloader)

    pct_start = 3000 / total_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=total_steps, pct_start=pct_start)

    # accelerate
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )

    if opt.smplx:
        SMPL_PATH='../SiTH/data/body_models'
        num_pca=45
        m_smplx = smplx.SMPLX( os.path.join(SMPL_PATH, 'smplx'), gender='male', use_pca=True, num_pca_comps=num_pca).cuda()
    else:
        smpl_model=SMPL('assets/SMPL_NEUTRAL.pkl').to( train_dataloader.device)
    # # loopimport os
    eval_during_train=True

    for epoch in range(opt.num_epochs):
        # train
        model.train()
        total_loss = 0
        total_psnr = 0
        for i, data in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                step_ratio = (epoch + i / len(train_dataloader)) / opt.num_epochs
                # print(data['name'])
                with torch.cuda.amp.autocast():
                    out = model(data, step_ratio)
                    loss = out['loss']

                psnr = out['psnr']
                accelerator.backward(loss)

            # gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

                optimizer.step()
                scheduler.step()

                total_loss += loss.detach()
                total_psnr += psnr.detach()
            # except:
                # continue
            if accelerator.is_main_process:
                # logging
                if i % 10 == 0:
                    mem_free, mem_total = torch.cuda.mem_get_info()    
                    print(f"[INFO] {i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.7f} step_ratio: {step_ratio:.4f} grid_loss: {grid_loss.item():.6f} dep_loss: {dep_loss.item():.6f} nor_loss: {nor_loss.item():.6f} flat_loss: {flat_loss.item():.6f} bce_loss: {bce_loss.item():.6f} sm_loss: {sm_loss.item():.6f} smpler_shape: {smpler_shape_loss.item():.6f} smpler_pose: {smpler_pose_loss.item():.6f}  smpler_kp3: {smpler_kp3_loss.item():.6f}  smpler_kp2: {smpler_kp2_loss.item():.6f} loss: {loss.item():.6f}")
                if i % 1000 == 0:
                    accelerator.save_model(model, opt.workspace)
                # save log images
                if i % 10 == 0:
                    gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                    kiui.write_image(f'{opt.workspace}/train_gt_images_{epoch}.jpg', gt_images)
                    pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/train_pred_images_{epoch}.jpg', pred_images)
                    if 'images_supp' in out:
                        pred_images_s = out['images_supp'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        pred_images_s = pred_images_s.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images_s.shape[1] * pred_images_s.shape[3], 3)
                        kiui.write_image(f'{opt.workspace}/train_pred_supps_{epoch}.jpg', pred_images_s)
                    if 'images_supp3' in out:    
                        pred_images_s3 = out['images_supp3'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        pred_images_s3 = pred_images_s3.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images_s3.shape[1] * pred_images_s3.shape[3], 3)
                        kiui.write_image(f'{opt.workspace}/train_pred_supps3_{epoch}.jpg', pred_images_s3)

        total_loss = accelerator.gather_for_metrics(total_loss).mean()
        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
      
        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            total_psnr /= len(train_dataloader)
            accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f} dep_loss:{total_dep_loss.item():.4f} nor_loss:{total_nor_loss.item():.4f} ")
        accelerator.wait_for_everyone()

if __name__ == "__main__":
    
    main()
