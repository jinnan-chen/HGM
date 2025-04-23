import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
from typing import Tuple, Literal
from functools import partial

from core.attention import MemEffAttention,MemEffCrossAttention

class MVAttention_pt(nn.Module):
    def __init__(
        self, 
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        groups: int = 32,
        eps: float = 1e-5,
        residual: bool = True,
        skip_scale: float = 1,
        num_frames: int = 1, # WARN: hardcoded!
    ):
        super().__init__()

        self.residual = residual
        self.skip_scale = skip_scale
        self.num_frames = num_frames

        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim, eps=eps, affine=True)
        self.attn = MemEffAttention(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop)

    def forward(self, x):
        # x: [B*V, C, H, W]

        BV, C, N= x.shape
        B = BV // self.num_frames # assert BV % self.num_frames == 0

        res = x
        x = self.norm(x)

        x = x.reshape(B, self.num_frames, C, N).reshape(B, -1, C)
        x = self.attn(x)
        x = x.reshape(B, self.num_frames, N,C).reshape(BV, C, N)

        if self.residual:
            x = (x + res) * self.skip_scale
        return x
    
class MVAttention(nn.Module):
    def __init__(
        self, 
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        groups: int = 32,
        eps: float = 1e-5,
        residual: bool = True,
        skip_scale: float = 1,
        num_frames: int = 1, # WARN: hardcoded!
    ):
        super().__init__()

        self.residual = residual
        self.skip_scale = skip_scale
        self.num_frames = num_frames

        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim, eps=eps, affine=True)
        self.attn = MemEffAttention(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop)

    def forward(self, x):
        # x: [B*V, C, H, W]
        # import ipdb
        # ipdb.set_trace()
        BV, C, H, W = x.shape
        B = BV // self.num_frames # assert BV % self.num_frames == 0
        # import ipdb
        # ipdb.set_trace()
        res = x
        x = self.norm(x)

        x = x.reshape(B, self.num_frames, C, H, W).permute(0, 1, 3, 4, 2).reshape(B, -1, C)
        x = self.attn(x)
        x = x.reshape(B, self.num_frames, H, W, C).permute(0, 1, 4, 2, 3).reshape(BV, C, H, W)

        if self.residual:
            x = (x + res) * self.skip_scale
        return x

class ResptsBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resample: Literal['default', 'up', 'down'] = 'default',
        groups: int = 32,
        eps: float = 1e-5,
        skip_scale: float = 1, # multiplied to output
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_scale = skip_scale

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.act = F.silu

        self.resample = None
        if resample == 'up':
            self.resample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
        elif resample == 'down':
            self.resample = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.shortcut = nn.Identity()
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)

    
    def forward(self, x):
        res = x

        x = self.norm1(x)
        x = self.act(x)

        if self.resample:
            res = self.resample(res)
            x = self.resample(x)
        
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x = (x + self.shortcut(res)) * self.skip_scale

        return x

class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resample: Literal['default', 'up', 'down'] = 'default',
        groups: int = 32,
        eps: float = 1e-5,
        skip_scale: float = 1, # multiplied to output
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_scale = skip_scale

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.act = F.silu

        self.resample = None
        if resample == 'up':
            self.resample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
        elif resample == 'down':
            self.resample = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.shortcut = nn.Identity()
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    
    def forward(self, x):
        res = x

        x = self.norm1(x)
        x = self.act(x)

        if self.resample:
            res = self.resample(res)
            x = self.resample(x)
        
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x = (x + self.shortcut(res)) * self.skip_scale

        return x

class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        downsample: bool = True,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
        num_frames=2,
    ):
        super().__init__()
 
        nets = []
        attns = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            nets.append(ResnetBlock(in_channels, out_channels, skip_scale=skip_scale))
            if attention:
                attns.append(MVAttention(out_channels, attention_heads, skip_scale=skip_scale,num_frames=num_frames))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

        self.downsample = None
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        xs = []

        for attn, net in zip(self.attns, self.nets):
            x = net(x)
            if attn:
                # import ipdb
                # ipdb.set_trace()
                x = attn(x)
            xs.append(x)

        if self.downsample:
            x = self.downsample(x)
            xs.append(x)
  
        return x, xs


class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
        num_frames=2
    ):
        super().__init__()

        nets = []
        attns = []
        # first layer
        nets.append(ResnetBlock(in_channels, in_channels, skip_scale=skip_scale))
        # more layers
        for i in range(num_layers):
            nets.append(ResnetBlock(in_channels, in_channels, skip_scale=skip_scale))
            if attention:
                attns.append(MVAttention(in_channels, attention_heads, skip_scale=skip_scale,num_frames=num_frames))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)
        
    def forward(self, x):
        x = self.nets[0](x)
        for attn, net in zip(self.attns, self.nets[1:]):
            if attn:
                x = attn(x)
            x = net(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_out_channels: int,
        out_channels: int,
        num_layers: int = 1,
        upsample: bool = True,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
        num_frames=2
    ):
        super().__init__()

        nets = []
        attns = []
        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            cskip = prev_out_channels if (i == num_layers - 1) else out_channels

            nets.append(ResnetBlock(cin + cskip, out_channels, skip_scale=skip_scale))
            if attention:
                attns.append(MVAttention(out_channels, attention_heads, skip_scale=skip_scale,num_frames=num_frames))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

        self.upsample = None
        if upsample:
            self.upsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, xs):

        for attn, net in zip(self.attns, self.nets):
            res_x = xs[-1]
            xs = xs[:-1]
            x = torch.cat([x, res_x], dim=1)
            x = net(x)
            if attn:
                x = attn(x)
            
        if self.upsample:
            # import ipdb
            # ipdb.set_trace()
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')
            x = self.upsample(x)

        return x


# it could be asymmetric!
class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024),
        down_attention: Tuple[bool, ...] = (False, False, False, True, True),
        mid_attention: bool = True,
        up_channels: Tuple[int, ...] = (1024, 512, 256,128),
        up_attention: Tuple[bool, ...] = (True, True, False,False,False),
        layers_per_block: int = 2,
        skip_scale: float = np.sqrt(0.5),
        num_frames : int=1 ,
    ):
        super().__init__()

        # first
        self.conv_in = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, stride=1, padding=1)
        self.grad_checkpointing=False
        # down
        down_blocks = []
        cout = down_channels[0]
        for i in range(len(down_channels)):
            cin = cout
            cout = down_channels[i]

            down_blocks.append(DownBlock(
                cin, cout, 
                num_layers=layers_per_block, 
                downsample=(i != len(down_channels) - 1), # not final layer
                attention=down_attention[i],
                skip_scale=skip_scale,
                num_frames= num_frames
            ))
        self.down_blocks = nn.ModuleList(down_blocks)

        # mid
        self.mid_block = MidBlock(down_channels[-1], attention=mid_attention, skip_scale=skip_scale, num_frames= num_frames)

        # up
        up_blocks = []
        cout = up_channels[0]
        for i in range(len(up_channels)):
            cin = cout
            cout = up_channels[i]
            cskip = down_channels[max(-2 - i, -len(down_channels))] # for assymetric

            up_blocks.append(UpBlock(
                cin, cskip, cout, 
                num_layers=layers_per_block + 1, # one more layer for up
                upsample=(i != len(up_channels) - 1), # not final layer
                attention=up_attention[i],
                skip_scale=skip_scale,
                num_frames= num_frames
            ))
        self.up_blocks = nn.ModuleList(up_blocks)

        # last
        self.norm_out = nn.GroupNorm(num_channels=up_channels[-1], num_groups=32, eps=1e-5)

        self.conv_out = nn.Conv2d(up_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)

    def set_grad_checkpointing(self, flag=True):
        self.grad_checkpointing = flag
    def forward(self, x):

        # x: [B, Cin, H, W]
        if self.grad_checkpointing:
        # first
            x = checkpoint(self.conv_in, x,)
        else:
            x = self.conv_in(x)
        # down
        xss = [x]


        for block in self.down_blocks:
            if self.grad_checkpointing:
                x, xs = checkpoint(block,x)
            else:
                x, xs = block(x)
            xss.extend(xs)
        
        # mid
        if self.grad_checkpointing:
            x = checkpoint(self.mid_block, x)
        else:
            x = self.mid_block(x)
        # import ipdb
        # ipdb.set_trace()
        # up

        for block in self.up_blocks:

            xs = xss[-len(block.nets):]
            xss = xss[:-len(block.nets)]
            if self.grad_checkpointing:
                x= checkpoint(block,x,xs)
            else:
                x = block(x, xs)

        # last
        if self.grad_checkpointing:
            x=checkpoint(self.norm_out,x)
            x = F.silu(x)
            x = checkpoint(self.conv_out,x)
            x=x.view(x.shape[0],14,x.shape[-2],x.shape[-1] )
        else:
            x = self.norm_out(x)
            x = F.silu(x)
    # import ipdb
    # ipdb.set_trace()
            x = self.conv_out(x).view(x.shape[0],14,x.shape[-2],x.shape[-1] ) # [B, Cout, H', W']

        return x
    

class UNet_feats(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024),
        down_attention: Tuple[bool, ...] = (False, False, False, True, True),
        mid_attention: bool = True,
        up_channels: Tuple[int, ...] = (1024, 512, 256,128,64),
        up_attention: Tuple[bool, ...] = (True, True, False),
        layers_per_block: int = 2,
        skip_scale: float = np.sqrt(0.5),
        num_frames: int =1
    ):
        super().__init__()
        self.grad_checkpointing = False
        # first
        self.conv_in = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, stride=1, padding=1)

        # down
        down_blocks = []
        cout = down_channels[0]
        for i in range(len(down_channels)):
            cin = cout
            cout = down_channels[i]

            down_blocks.append(DownBlock(
                cin, cout, 
                num_layers=layers_per_block, 
                downsample=(i != len(down_channels) - 1), # not final layer
                attention=down_attention[i],
                skip_scale=skip_scale,
                num_frames= num_frames
            ))
        self.down_blocks = nn.ModuleList(down_blocks)

        # mid
        self.mid_block = MidBlock(down_channels[-1], attention=mid_attention, skip_scale=skip_scale,num_frames= num_frames)

        # up
        up_blocks = []
        cout = up_channels[0]
        for i in range(len(up_channels)):
            cin = cout
            cout = up_channels[i]
            cskip = down_channels[max(-2 - i, -len(down_channels))] # for assymetric

            up_blocks.append(UpBlock(
                cin, cskip, cout, 
                num_layers=layers_per_block + 1, # one more layer for up
                upsample=(i != len(up_channels) - 1), # not final layer
                attention=up_attention[i],
                skip_scale=skip_scale,
                num_frames= num_frames
            ))
        self.up_blocks = nn.ModuleList(up_blocks)

        # last
        self.norm_out = nn.GroupNorm(num_channels=up_channels[-1], num_groups=32, eps=1e-5)
        self.conv_out = nn.Conv2d(up_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)
        self.norm_out_2 = nn.GroupNorm(num_channels=up_channels[-1]*2, num_groups=32, eps=1e-5)
        self.conv_out_2 = nn.Conv1d(up_channels[-1]*2, out_channels, kernel_size=3, stride=1, padding=1)

    def set_grad_checkpointing(self, flag=True):
        self.grad_checkpointing = flag
    def forward(self, x):

        # x: [B, Cin, H, W]
        if self.grad_checkpointing:
        # first
            x = checkpoint(self.conv_in, x)
        else:
            x = self.conv_in(x)
        # down
        xss = [x]


        for block in self.down_blocks:
            if self.grad_checkpointing:
                x, xs = checkpoint(block,x)
            else:
                x, xs = block(x)
            xss.extend(xs)
        
        # mid
        if self.grad_checkpointing:
            x = checkpoint(self.mid_block, x)
        else:
            x = self.mid_block(x)
        # import ipdb
        # ipdb.set_trace()
        # up

        for block in self.up_blocks:

            xs = xss[-len(block.nets):]
            xss = xss[:-len(block.nets)]
            if self.grad_checkpointing:
                x= checkpoint(block,x,xs)
            else:
                x = block(x, xs)
        feats=x
        # last
        if self.grad_checkpointing:
            x=checkpoint(self.norm_out,x)
            x = F.silu(x)
            x = checkpoint(self.conv_out,x)
            x=x.view(x.shape[0],14,x.shape[-2],x.shape[-1] )
        else:
            
            x = self.norm_out(x)
            x = F.silu(x)
        # import ipdb
        # ipdb.set_trace()
            x = self.conv_out(x).view(x.shape[0],14,x.shape[-2],x.shape[-1] ) # [B, Cout, H', W']

        return x,feats
   
        
class UpBlock_pt(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_out_channels: int,
        out_channels: int,
        num_layers: int = 1,
        upsample: bool = False,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
    ):
        super().__init__()

        nets = []
        attns = []
        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            cskip = prev_out_channels if (i == num_layers - 1) else out_channels

            nets.append(ResnetBlock_pt(cin + cskip, out_channels, skip_scale=skip_scale))
            if attention:
                attns.append(MVAttention_pt(out_channels, attention_heads, skip_scale=skip_scale))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

        self.upsample = None
        if upsample:
            self.upsample = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, xs):

        for attn, net in zip(self.attns, self.nets):
            res_x = xs[-1]
            xs = xs[:-1]
            # import ipdb
            # ipdb.set_trace()
            x = torch.cat([x, res_x], dim=1)
            x = net(x)
            if attn:
                x = attn(x)
            
        if self.upsample:
            # import ipdb
            # ipdb.set_trace()
            # x = F.interpolate(x, size=2*x.shape[-1]-1 if x.shape[-1]!=53433 else 2*x.shape[-1], mode='nearest')
            x = F.interpolate(x, size=2*x.shape[-1]-1 if x.shape[-1] not in [431,3445,8192//2,7846//2,1310//2 ,36864//2,73728 //2,73728 , 5238//2, 20665,8192,18432//2,53433,16384,32768,65536,24576//2,49152//2,98304//2,98304,172402//2,262144//2] else 2*x.shape[-1], mode='nearest')
            # x = F.interpolate(x, size=2*x.shape[-1]-1 if x.shape[-1]!=28857 else 2*x.shape[-1], mode='nearest')
            x = self.upsample(x)

        return x

class MidBlock_pt(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
    ):
        super().__init__()

        nets = []
        attns = []
        # first layer
        nets.append(ResnetBlock_pt(in_channels, in_channels, skip_scale=skip_scale))
        # more layers
        for i in range(num_layers):
            nets.append(ResnetBlock_pt(in_channels, in_channels, skip_scale=skip_scale))
            if attention:
                attns.append(MVAttention_pt(in_channels, attention_heads, skip_scale=skip_scale))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)
        
    def forward(self, x):
        x = self.nets[0](x)
        for attn, net in zip(self.attns, self.nets[1:]):
            if attn:
                x = attn(x)
            x = net(x)
        return x

class DownBlock_pt(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        downsample: bool = False,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
    ):
        super().__init__()
 
        nets = []
        attns = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            nets.append(ResnetBlock_pt(in_channels, out_channels, skip_scale=skip_scale))
            if attention:
                attns.append(MVAttention_pt(out_channels, attention_heads, skip_scale=skip_scale))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

        self.downsample = None
        if downsample:
            self.downsample = nn.Conv1d(out_channels , out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        xs = []

        for attn, net in zip(self.attns, self.nets):

            x = net(x)
            if attn:
                x = attn(x)
            xs.append(x)
        if self.downsample:
            # import ipdb
            # ipdb.set_trace()
            x = self.downsample(x)
            xs.append(x)
  
        return x, xs

class ResnetBlock_pt(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resample: Literal['default', 'up', 'down'] = 'default',
        groups: int = 32,
        eps: float = 1e-5,
        skip_scale: float = 1, # multiplied to output
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_scale = skip_scale

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.act = F.silu

        self.resample = None
        if resample == 'up':
            self.resample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
        elif resample == 'down':
            self.resample = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.shortcut = nn.Identity()
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)

    
    def forward(self, x):
        res = x

        x = self.norm1(x)
        x = self.act(x)

        if self.resample:
            res = self.resample(res)
            x = self.resample(x)
        
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x = (x + self.shortcut(res)) * self.skip_scale

        return x


class MixNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 14,
        down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024),
        down_attention: Tuple[bool, ...] = (False, False, False, True, True),
        mid_attention: bool = True,
        up_channels: Tuple[int, ...] = (1024, 512, 256,128,64),
        up_attention: Tuple[bool, ...] = (True, True, False,False,False),
        layers_per_block: int = 2,
        skip_scale: float = np.sqrt(0.5),
    ):
        super().__init__()

        # first
        self.conv_in = nn.Conv1d(in_channels, down_channels[0], kernel_size=3, stride=1, padding=1)
        self.grad_checkpointing=False
        # down
        down_blocks = []
        cout = down_channels[0]
        for i in range(len(down_channels)):
            cin = cout
            cout = down_channels[i]

            down_blocks.append(DownBlock_pt(
                cin, cout, 
                num_layers=layers_per_block, 
                downsample=(i != len(down_channels) - 1), # not final layer
                # downsample=False,
                attention=down_attention[i],
                skip_scale=skip_scale,
            ))
        self.down_blocks = nn.ModuleList(down_blocks)

        # mid
        self.mid_block = MidBlock_pt(down_channels[-1], attention=mid_attention, skip_scale=skip_scale)

        # up
        up_blocks = []
        cout = up_channels[0]
        for i in range(len(up_channels)):
            cin = cout
            cout = up_channels[i]
            cskip = down_channels[max(-2 - i, -len(down_channels))] # for assymetric

            up_blocks.append(UpBlock_pt(
                cin, cskip, cout, 
                num_layers=layers_per_block + 1, # one more layer for up
                upsample=(i != len(up_channels) - 1), # not final layer
                # upsample=False,
                attention=up_attention[i],
                skip_scale=skip_scale,
            ))
        self.up_blocks = nn.ModuleList(up_blocks)

        # last
        self.norm_out = nn.GroupNorm(num_channels=up_channels[-1], num_groups=32, eps=1e-5)
        self.conv_out = nn.Conv1d(up_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)

    def set_grad_checkpointing(self, flag=True):
        self.grad_checkpointing = flag
        

    def forward(self, x):
        # import ipdb
        # ipdb.set_trace()
        if self.grad_checkpointing:
            x=checkpoint(self.conv_in,x)
        else:
            x = self.conv_in(x)
        
        xss = [x]
        for block in self.down_blocks:

            if self.grad_checkpointing:
                x, xs =checkpoint(block,x)
            else:
                x, xs = block(x)
            # ipdb.set_trace()
            xss.extend(xs)
        
        # mid
        x = self.mid_block(x)

        for block in self.up_blocks:
            
            xs = xss[-len(block.nets):]
            xss = xss[:-len(block.nets)]
            if self.grad_checkpointing:
                x =checkpoint(block,x,xs)
            else:
                x = block(x, xs)

        # last
        # ipdb.set_trace()
        if self.grad_checkpointing:
            x =checkpoint(self.norm_out,x)
            x = F.silu(x)
            x =checkpoint(self.conv_out,x)
            
        else:
            x = self.norm_out(x)
            x = F.silu(x)
            x = self.conv_out(x) # [B, Cout, H', W']

        return x
    



class MVAttention_cross(nn.Module):
    def __init__(
        self, 
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        groups: int = 32,
        eps: float = 1e-5,
        residual: bool = True,
        skip_scale: float = 1,
        num_frames: int = 1, # WARN: hardcoded!
    ):
        super().__init__()

        self.residual = residual
        self.skip_scale = skip_scale
        self.num_frames = num_frames

        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim, eps=eps, affine=True)
        self.attn = MemEffCrossAttention(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop)

    def forward(self, x,y):
        # x: [B*V, C, H, W]
        # import ipdb
        # ipdb.set_trace()
        BV, C, N= x.shape
        B = BV // self.num_frames # assert BV % self.num_frames == 0

        res = x
        x = self.norm(x)

        x = x.reshape(B, self.num_frames, C, N).reshape(B, -1, C)
        x = self.attn(x)
        x = x.reshape(B, self.num_frames, N,C).reshape(BV, C, N)

        if self.residual:
            x = (x + res) * self.skip_scale
        return x