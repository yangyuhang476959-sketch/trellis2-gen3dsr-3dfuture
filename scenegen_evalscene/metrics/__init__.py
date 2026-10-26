from .geom_metrics import (
    compute_chamfer_distance,
    compute_fscore,
    compute_volume_iou,
    normalize_points,
    sample_points_from_meshes,
)
from .chamfer_distance import ChamferDistance
from .alignment import point_alignment

def _render_metric_unavailable(*args, **kwargs):
    raise RuntimeError("Render metrics are disabled in this local geometry-only adapter.")

compute_psnr = _render_metric_unavailable
compute_ssim = _render_metric_unavailable
compute_lpips = _render_metric_unavailable
compute_fid = _render_metric_unavailable
compute_clip_similarity = _render_metric_unavailable
compute_dinov2_similarity = _render_metric_unavailable
load_clip_model = _render_metric_unavailable
load_dinov2_model = _render_metric_unavailable
load_lpips_model = _render_metric_unavailable
compute_render_metrics_ = _render_metric_unavailable

__all__ = [
    'compute_chamfer_distance',
    'compute_fscore',
    'compute_volume_iou',
    'normalize_points',
    'sample_points_from_meshes',
    'ChamferDistance',
    'alignment',
    'point_alignment',
    'compute_psnr',
    'compute_ssim',
    'compute_lpips',
    'compute_fid',
    'compute_clip_similarity',
    'compute_dinov2_similarity',
    'load_clip_model',
    'load_dinov2_model',
    'load_lpips_model',
    'compute_render_metrics_',
] 
