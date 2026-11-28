"""Tensor helpers shared by the Fable View GPU stages.

Images are float32 NCHW tensors in [0, 1], channel order B, G, R to match the
OpenCV frames they are uploaded from.  Pixel centres sit at integer
coordinates.  A 3x3 matrix always maps *output* pixel coordinates to *input*
pixel coordinates when it is used to build a sampling grid.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

LUMA_BGR = (0.114, 0.587, 0.299)


class BackendError(RuntimeError):
    """The requested compute backend is unavailable or failed."""


def select_device(device='auto', require_mps=False):
    """Resolve ``auto|mps|cpu`` to a torch device.  Never silently downgrade
    when the caller requires the Apple GPU."""
    want = str(device or 'auto').lower()
    mps_ok = bool(torch.backends.mps.is_available())
    if want == 'mps' or (want == 'auto' and mps_ok):
        if not mps_ok:
            raise BackendError('Apple GPU (MPS) was requested but is not available')
        return torch.device('mps')
    if require_mps:
        raise BackendError('Apple GPU (MPS) is required; refusing to run on %s' % want)
    return torch.device('cpu')


def sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()


def upload_bgr(frame, device):
    """uint8 HxWx3 BGR -> float32 1x3xHxW in [0, 1] on ``device``."""
    tensor = torch.from_numpy(np.ascontiguousarray(frame)).to(device)
    return tensor.permute(2, 0, 1).unsqueeze(0).to(torch.float32).mul_(1.0 / 255.0)


def download_bgr(tensor):
    """float32 1x3xHxW in [0, 1] -> uint8 HxWx3 BGR numpy."""
    out = tensor.clamp(0.0, 1.0).mul(255.0).add_(0.5).to(torch.uint8)
    return out.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()


def download_gray(tensor):
    """float32 1x1xHxW in [0, 1] -> uint8 HxW numpy."""
    out = tensor.clamp(0.0, 1.0).mul(255.0).add_(0.5).to(torch.uint8)
    return out.squeeze(0).squeeze(0).contiguous().cpu().numpy()


def luma(image):
    """Rec.601 luma of a BGR tensor, 1x1xHxW."""
    b, g, r = LUMA_BGR
    return image[:, 0:1] * b + image[:, 1:2] * g + image[:, 2:3] * r


class GridCache:
    """Builds homography sampling grids, caching the pixel-coordinate planes."""

    def __init__(self, device):
        self.device = device
        self._planes = {}

    STEP = 8

    def planes(self, height, width):
        """Homogeneous pixel coordinates of a coarse node lattice spanning an
        output raster exactly (first and last nodes on the border pixels)."""
        key = (int(height), int(width))
        if key not in self._planes:
            ny = max(2, (key[0] - 1) // self.STEP + 2)
            nx = max(2, (key[1] - 1) // self.STEP + 2)
            ys = torch.linspace(0.0, key[0] - 1.0, ny, device=self.device)
            xs = torch.linspace(0.0, key[1] - 1.0, nx, device=self.device)
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            self._planes[key] = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)
        return self._planes[key]

    def grid(self, matrix, out_hw, in_hw):
        """Sampling grid and in-bounds mask.

        ``matrix`` maps output pixel coordinates to input pixel coordinates.
        Returns ``(grid 1xHxWx2, valid 1x1xHxW float)``.

        The projective map is evaluated on an 8-pixel lattice and bilinearly
        interpolated.  That is exact for affine motion and within about
        0.001 px for the perspective range the registration accepts.
        """
        out_h, out_w = int(out_hw[0]), int(out_hw[1])
        in_h, in_w = int(in_hw[0]), int(in_hw[1])
        norm = np.array([[2.0 / in_w, 0.0, 1.0 / in_w - 1.0],
                         [0.0, 2.0 / in_h, 1.0 / in_h - 1.0],
                         [0.0, 0.0, 1.0]], dtype=np.float64)
        m = norm @ np.asarray(matrix, dtype=np.float64)
        mt = torch.as_tensor(m.T, dtype=torch.float32, device=self.device)
        uvw = self.planes(out_h, out_w) @ mt
        w = uvw[..., 2:3]
        coarse = torch.where(w > 1e-6, uvw[..., :2] / w, torch.full_like(uvw[..., :2], 4.0))
        grid = F.interpolate(coarse.permute(2, 0, 1).unsqueeze(0), size=(out_h, out_w),
                             mode='bilinear', align_corners=True)
        valid = ((grid[:, 0:1].abs() <= 1.0 - 1.0 / in_w) &
                 (grid[:, 1:2].abs() <= 1.0 - 1.0 / in_h)).to(torch.float32)
        return grid.permute(0, 2, 3, 1), valid


def sample(image, grid, mode='bilinear'):
    return F.grid_sample(image, grid, mode=mode, padding_mode='border', align_corners=False)


def gaussian_kernel1d(sigma, device, radius=None):
    radius = int(radius if radius is not None else max(1, math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def gaussian_blur(image, sigma, radius=None):
    """Separable Gaussian with replicate borders; works on any channel count."""
    if sigma <= 0:
        return image
    k = gaussian_kernel1d(sigma, image.device, radius)
    r = (k.numel() - 1) // 2
    c = image.shape[1]
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    out = F.conv2d(F.pad(image, (r, r, 0, 0), mode='replicate'), kx, groups=c)
    return F.conv2d(F.pad(out, (0, 0, r, r), mode='replicate'), ky, groups=c)


def box_blur(image, radius):
    """Mean filter of side ``2*radius+1`` with replicate borders."""
    if radius <= 0:
        return image
    k = 2 * radius + 1
    return F.avg_pool2d(F.pad(image, (radius, radius, radius, radius), mode='replicate'), k, 1)


def guided_self(image, radius, eps):
    """Edge-preserving self-guided filter.  ``eps`` may be a tensor (per pixel
    variance threshold): structure with local variance well above ``eps`` is
    preserved, fluctuation below it is averaged."""
    mean = box_blur(image, radius)
    var = (box_blur(image * image, radius) - mean * mean).clamp_min(0.0)
    a = var / (var + eps)
    b = mean * (1.0 - a)
    return box_blur(a, radius) * image + box_blur(b, radius)


def guided_joint(guide, source, radius, eps):
    """Guided filter of ``source`` (N-channel) steered by a 1-channel guide."""
    mean_g = box_blur(guide, radius)
    mean_s = box_blur(source, radius)
    cov = box_blur(guide * source, radius) - mean_g * mean_s
    var = (box_blur(guide * guide, radius) - mean_g * mean_g).clamp_min(0.0)
    a = cov / (var + eps)
    b = mean_s - a * mean_g
    return box_blur(a, radius) * guide + box_blur(b, radius)


def smoothstep(x, lo, hi):
    t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def subsample_numpy(tensor, stride):
    """Strided CPU copy of a 1xCxHxW tensor as float32 CxH'xW'."""
    return tensor[0, :, ::stride, ::stride].contiguous().cpu().numpy()


def to_grid_h(matrix_out_to_in):
    return np.asarray(matrix_out_to_in, dtype=np.float64)


def translation(tx, ty):
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float64)


def scale_about_centres(factor):
    """Maps coordinates on a ``factor``-times finer pixel grid to the coarse
    grid they were pooled from (pixel-centre aware)."""
    return np.array([[factor, 0.0, (factor - 1.0) * 0.5],
                     [0.0, factor, (factor - 1.0) * 0.5],
                     [0.0, 0.0, 1.0]], dtype=np.float64)
