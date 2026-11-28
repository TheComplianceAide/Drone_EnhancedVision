"""Optional Apple GPU implementation of Motion Thread's observation map."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from motion_thread import TILE, _observations as _cpu_observations


_CANDIDATE_CAPACITY = 65536
_LOCAL_SHAPE = (7, 12)
_METAL = r"""
#include <metal_stdlib>
using namespace metal;

inline int reflect101(int position, int limit) {
    return position < 0 ? -position : (position >= limit ? 2 * limit - position - 2 : position);
}

inline float gaussian(device const float *source, device const float *weights,
                      int x, int y, int width, int height, int radius, bool vertical) {
    float result = 0.0f;
    for (int d = -radius; d <= radius; ++d) {
        const int xx = vertical ? x : reflect101(x + d, width);
        const int yy = vertical ? reflect101(y + d, height) : y;
        result = fma(source[yy * width + xx], weights[d + radius], result);
    }
    return result;
}

kernel void blur(device const float *source, device const float *weights, device float *target,
                 constant int &width, constant int &height, constant int &radius,
                 constant int &vertical, uint index [[thread_position_in_grid]]) {
    int x = index % width, y = index / width;
    target[index] = gaussian(source, weights, x, y, width, height, radius, vertical != 0);
}

kernel void bandpass(device const float *low, device const float *horizontal, device const float *weights,
                     device float *target, constant int &width, constant int &height,
                     constant int &radius, uint index [[thread_position_in_grid]]) {
    int x = index % width, y = index / width;
    target[index] = low[index] - gaussian(horizontal, weights, x, y, width, height, radius, true);
}

kernel void tile_scatter(device const float *core, device float *scatter, constant int &width,
                         constant int &tile_cols, constant float &floor_value,
                         uint tile [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float samples[256];
    const int x = (tile % tile_cols) * 64 + (lane % 16) * 4;
    const int y = (tile / tile_cols) * 64 + (lane / 16) * 4;
    samples[lane] = fabs(core[y * width + x]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Each lane writes only its own item. Read pairs from the prior stage.
    for (uint span = 2; span <= 256; span *= 2) {
        for (uint stride = span / 2; stride; stride /= 2) {
            const uint peer = lane ^ stride;
            const float a = samples[lane], b = samples[peer];
            const bool ascending = (lane & span) == 0;
            const bool lower_lane = lane < peer;
            const float next = ascending == lower_lane ? min(a, b) : max(a, b);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            samples[lane] = next;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (lane == 0) {
        scatter[tile] = max((samples[127] + samples[128]) * .74f, floor_value);
    }
}

inline float grid_value(device const float *grid, int x, int y, int width, int height,
                        int cols, int rows) {
    const float sx = clamp((float(x) + .5f) * float(cols) / float(width) - .5f, 0.0f, float(cols - 1));
    const float sy = clamp((float(y) + .5f) * float(rows) / float(height) - .5f, 0.0f, float(rows - 1));
    const int left = int(sx), top = int(sy);
    const int right = min(left + 1, cols - 1), bottom = min(top + 1, rows - 1);
    return mix(mix(grid[top * cols + left], grid[top * cols + right], sx - left),
               mix(grid[bottom * cols + left], grid[bottom * cols + right], sx - left), sy - top);
}

kernel void score_points(device const float *gray, device const float *smooth, device const float *core,
                         device const uchar *valid, device const float *scatter, device const float *nuisance,
                         device float *scores, constant int &width, constant int &height,
                         constant int &tile_cols, constant int &tile_rows, constant int &has_nuisance,
                         constant float &floor_value, uint index [[thread_position_in_grid]]) {
    const float compact = gray[index] - smooth[index], change = core[index];
    if (!valid[index] || change * compact <= 0.0f || fabs(compact) <= floor_value * .70f) {
        scores[index] = 0.0f;
        return;
    }
    const int x = index % width, y = index / width;
    float high = smooth[index], low = high;
    for (int yy = max(y - 2, 0); yy <= min(y + 2, height - 1); ++yy) {
        for (int xx = max(x - 2, 0); xx <= min(x + 2, width - 1); ++xx) {
            const float value = smooth[yy * width + xx];
            high = max(high, value);
            low = min(low, value);
        }
    }
    const float edge = min(high - low, 42.0f);
    const float disagreement = has_nuisance ? clamp(grid_value(nuisance, x, y, width, height, 12, 7), 0.0f, 4.0f) : 0.0f;
    const float sigma = grid_value(scatter, x, y, width, height, tile_cols, tile_rows)
                        + edge * (.018f + disagreement * .14f);
    scores[index] = fabs(change) / sigma;
}

kernel void collect_peaks(device const float *scores, device const float *core,
                           device uint *result, constant int &width, constant int &height,
                           constant uint &capacity, constant float &threshold,
                           uint index [[thread_position_in_grid]]) {
    const float score = scores[index];
    if (score < threshold) return;
    const int x = index % width, y = index / width;
    for (int yy = max(y - 2, 0); yy <= min(y + 2, height - 1); ++yy) {
        for (int xx = max(x - 2, 0); xx <= min(x + 2, width - 1); ++xx) {
            if (scores[yy * width + xx] > score) return;
        }
    }
    const uint slot = atomic_fetch_add_explicit((device atomic_uint *)result, 1u, memory_order_relaxed);
    if (slot < capacity) {
        result[2 * slot + 1] = index;
        result[2 * slot + 2] = as_type<uint>(copysign(score, core[index]));
    }
}
"""


def available() -> bool:
    """Report whether the current process can use the Metal kernel."""
    return torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")


class MetalObservations:
    """Callable replacement for _observations, owned by a single frame loop.

    The inputs and returned candidates use NumPy arrays on the CPU. Candidate
    coordinates are in source pixels and obey the CPU path's per-region cap.
    """

    def __init__(self):
        if not available():
            raise RuntimeError("The Apple GPU is unavailable in this process")
        if TILE != 64:
            raise RuntimeError("The Metal observation kernel requires 64-pixel tiles")
        self._program = torch.mps.compile_shader(_METAL)
        self._weights = []
        for filters in (((2, .72), (5, 2.25), (3, 1.3)), ((3, 1.16), (8, 3.35), (4, 1.9))):
            self._weights.append(tuple(
                (radius, torch.from_numpy(cv2.getGaussianKernel(radius * 2 + 1, sigma, cv2.CV_32F).ravel()).to("mps"))
                for radius, sigma in filters
            ))
        self._buffers = {}
        self.fallbacks = 0
        self._nuisance = torch.empty(_LOCAL_SHAPE, dtype=torch.float32, device="mps")
        self._result = torch.empty(2 * _CANDIDATE_CAPACITY + 1, dtype=torch.int32, device="mps")

    def _working_buffers(self, shape, regional):
        existing = self._buffers.get(regional)
        if existing is not None and existing[0].shape == shape:
            return existing
        buffers = tuple(torch.empty(shape, dtype=torch.float32, device="mps") for _ in range(7))
        valid = torch.empty(shape, dtype=torch.uint8, device="mps")
        scatter = torch.empty(tuple(size // TILE for size in shape), dtype=torch.float32, device="mps")
        self._buffers[regional] = (*buffers, valid, scatter)
        return self._buffers[regional]

    def __call__(self, gray, delta, valid, local, dim, *, regional=False, select=None, peek=None):
        """Return (xy, signs, strengths) for float32 luma and residual rasters."""
        if gray.ndim != 2 or min(gray.shape) < TILE or delta.shape != gray.shape or valid.shape != gray.shape:
            raise ValueError("Metal observations require equal two-dimensional rasters at least one tile across")
        if local is not None and local.shape != _LOCAL_SHAPE:
            raise ValueError("Unexpected local disagreement grid")
        image, residual, work, primary, core, smooth, scores, support, scatter = self._working_buffers(gray.shape, regional)
        image.copy_(torch.from_numpy(np.ascontiguousarray(gray, dtype=np.float32)), non_blocking=True)
        residual.copy_(torch.from_numpy(np.ascontiguousarray(delta, dtype=np.float32)), non_blocking=True)
        support.copy_(torch.from_numpy(np.ascontiguousarray(valid, dtype=np.uint8)), non_blocking=True)
        if local is not None:
            self._nuisance.copy_(torch.from_numpy(np.ascontiguousarray(local, dtype=np.float32)), non_blocking=True)
        self._result[:1].zero_()
        height, width = gray.shape
        rows, cols = scatter.shape
        floor = .48 if dim else .63
        program = self._program
        pixels = height * width
        (first_r, first_w), (second_r, second_w), (smooth_r, smooth_w) = self._weights[int(regional)]
        program.blur(residual, first_w, work, width, height, first_r, 0, threads=pixels)
        program.blur(work, first_w, primary, width, height, first_r, 1, threads=pixels)
        program.blur(primary, second_w, work, width, height, second_r, 0, threads=pixels)
        program.bandpass(primary, work, second_w, core, width, height, second_r, threads=pixels)
        program.blur(image, smooth_w, work, width, height, smooth_r, 0, threads=pixels)
        program.blur(work, smooth_w, smooth, width, height, smooth_r, 1, threads=pixels)
        program.tile_scatter(core, scatter, width, cols, floor,
                             threads=rows * cols * 256, group_size=256)
        program.score_points(image, smooth, core, support, scatter,
                             self._nuisance, scores, width, height, cols, rows, int(local is not None),
                             floor, threads=pixels)
        program.collect_peaks(scores, core, self._result, width, height,
                              _CANDIDATE_CAPACITY, 1.90 if regional else 1.54, threads=pixels)
        packed = self._result.cpu().numpy()
        size = int(packed[0])
        if not size:
            return np.empty((0, 2), np.float32), np.empty(0), np.empty(0)
        if size > _CANDIDATE_CAPACITY:
            self.fallbacks += 1
            return _cpu_observations(gray, delta, valid, local, dim, regional=regional)
        entries = packed[1:2 * size + 1].reshape(-1, 2)
        # Atomic collection has no traversal order. Restore CPU row-major
        # order before applying the CPU scorer's tie and per-region rules.
        entries = entries[np.argsort(entries[:, 0])]
        yy, xx = np.divmod(entries[:, 0], width)
        signed = entries[:, 1].copy().view(np.float32)
        if peek is not None:
            peek(xx, yy, signed)
        strength = np.abs(signed)
        if select is not None:
            take = select(xx, yy, strength, width, regional)
            return np.column_stack((xx[take], yy[take])).astype(np.float32), np.sign(signed[take]), strength[take]
        tile_size, per_tile, cap = (75, 2, 82) if regional else (110, 4, 245)
        bins = (yy // tile_size) * ((width + tile_size - 1) // tile_size) + xx // tile_size
        used = np.zeros(int(bins.max()) + 1, np.int16)
        take = []
        for item in np.argsort(strength)[::-1]:
            group = bins[item]
            if used[group] < per_tile:
                used[group] += 1
                take.append(item)
                if len(take) == cap:
                    break
        return np.column_stack((xx[take], yy[take])).astype(np.float32), np.sign(signed[take]), strength[take]
