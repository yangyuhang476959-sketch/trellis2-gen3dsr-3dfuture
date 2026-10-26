import os
from pathlib import Path
import torchvision.transforms.functional as Ftv
from util import *


def run_perspectivefields(image):
    from perspective2d import PerspectiveFields
    from perspective2d.utils.utils import general_vfov_to_focal

    img_bgr = image[..., ::-1]
    H, W, _ = img_bgr.shape
    pf_model = PerspectiveFields('Paramnet-360Cities-edina-uncentered').eval().cuda()
    pred = pf_model.inference(img_bgr=img_bgr)

    roll = np.radians(pred['pred_roll'].cpu().item())
    pitch = np.radians(pred['pred_pitch'].cpu().item())
    vfov = np.radians(pred['pred_general_vfov'].cpu().item())
    cx_rel = pred['pred_rel_cx'].cpu().item()
    cy_rel = pred['pred_rel_cy'].cpu().item()
    focal_rel = general_vfov_to_focal(cx_rel, cy_rel, 1, vfov, degree=False)
    f = focal_rel * H
    cx = (cx_rel + 0.5) * W
    cy = (cy_rel + 0.5) * H
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])
    return K, roll, pitch


def run_marigold(image:Image.Image, out_dir:Path):
    save_path = (out_dir / "marigold").absolute()
    save_path.mkdir(exist_ok=True)
    image.save(save_path / f"input.png")
    bash_script = (
        'python ../external/Marigold/run.py '
        f'--input_rgb_dir {save_path} '
        f'--output_dir {save_path} '
        '--checkpoint ../external/checkpoints/marigold-lcm-v1-0 '
        '--ensemble_size 5 '
        '--denoise_steps 2'
    )     
    print(bash_script)
    os.system(bash_script)
    relative_depth = np.load(save_path / 'depth_npy' / 'input_pred.npy')
    return relative_depth


def run_depthanything(image:Image.Image, device):
    from zoedepth.utils.config import get_config
    from zoedepth.models.builder import build_model
    config = get_config("zoedepth", "eval", "nyu", pretrained_resource="local::../external/checkpoints/depth_anything_metric_depth_indoor.pt")
    model = build_model(config).to(device)
    return model.infer_pil(image)


def run_entityv2(image: np.ndarray, threshold=0.1, max_size=1500):
    #For cleanup
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.modeling import BACKBONE_REGISTRY, SEM_SEG_HEADS_REGISTRY
    def_keys = list(DatasetCatalog.keys())
    ####

    from detectron2.config import get_cfg
    from demo_cropformer.predictor import VisualizationDemo
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config
    CropFormerCfg = {
        'config_file': "../external/detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/cropformer_hornet_3x.yaml",
        'opts': ["MODEL.WEIGHTS", "../external/checkpoints/CropFormer_hornet_3x_03823a.pth"],
    }
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(CropFormerCfg['config_file'])
    cfg.merge_from_list(CropFormerCfg['opts'])
    cfg.freeze()
    segmentor = VisualizationDemo(cfg)

    # CropFormer expects BGR
    orig_size = image.shape
    if max(image.shape) > max_size:
        scale_factor = max_size / max(image.shape[:-1])
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    img = image.copy()[:, :, ::-1]
    predictions = segmentor.run_on_image(img)

    #hacky cleanup - is there a better way?
    extra_keys = list(DatasetCatalog.keys())
    for key in extra_keys:
        if key not in def_keys:
            DatasetCatalog.remove(key)
            MetadataCatalog.remove(key)
    BACKBONE_REGISTRY._obj_map = {}
    SEM_SEG_HEADS_REGISTRY._obj_map = {}
    #####
    
    pred_masks = predictions["instances"].pred_masks
    pred_scores = predictions["instances"].scores
    selected_indexes = (pred_scores >= threshold)
    selected_masks = pred_masks[selected_indexes]
    selected_masks = Ftv.resize(selected_masks.cpu(), orig_size[:-1]).numpy() > 0.5  # make binary
    return selected_masks


def complete_object(crop, label, model):
    image, mask = np.split(np.array(crop) / 255, (3, ), axis=-1)
    image[mask[:, :, 0] < 0.5] = 0.5
    completed = model(prompt=label, image=image,
                      num_inference_steps=50, image_guidance_scale=1.5,
                      guidance_scale=8.5, num_images_per_prompt=1).images[0]
    return completed


def run_clipseg(image, masks):
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    background_prompts = ["background", "floor", "wall", "curtain", "window", "ceiling", "table"]
    foreground_prompts = ["object", "furniture"]

    img = image
    inputs = processor(
        text=background_prompts + foreground_prompts,
        images=[image] * (len(background_prompts) + len(foreground_prompts)),
        padding="max_length", return_tensors="pt"
    )
    predicted = torch.sigmoid(model(**inputs).logits)
    back_pred = (predicted[:len(background_prompts)] > 0.5).any(dim=0)
    fore_pred = (predicted[-len(foreground_prompts):] > 0.1).any(dim=0)
    foreground_mask = torch.logical_or(~back_pred, fore_pred).numpy()
    foreground_mask = np.array(Image.fromarray(foreground_mask).resize(img.size))
    return filter_component_masks(masks, foreground_mask)


def run_oneformer(image, masks, device):
    predictor, metadata, thing_classes_ids = initialize_oneformer(device)
    W, H = image.size
    f = 640 * 4 / W
    img = np.array(image.resize((int(W * f), int(H * f)), Image.BILINEAR))[:, :, ::-1]  # not sure if RGB or BGR
    predictions = predictor(img, "semantic")['sem_seg'].argmax(dim=0).cpu().numpy()
    is_thing = Image.fromarray(np.isin(predictions, thing_classes_ids))
    is_thing = np.array(is_thing.resize((W, H), Image.NEAREST))
    return filter_component_masks(masks, is_thing)


def run_dreamgaussian(scene_dir, obj_id, elevation):
    save_path = (scene_dir / "object_space" / f"{obj_id}").absolute()
    main_dir_path = Path.cwd().absolute()
    img_path = (scene_dir / "crops" / f"{obj_id}_rgba.png").absolute()
    os.chdir('../external/dreamgaussian/')
    bash_script = f'python main.py --config configs/image.yaml input={img_path} save_path={save_path} elevation={elevation} force_cuda_rast=True'
    print(bash_script)
    os.system(bash_script)
    bash_script = f'python main2.py --config configs/image.yaml input={img_path} save_path={save_path} elevation={elevation} force_cuda_rast=True'
    print(bash_script)
    os.system(bash_script)
    os.chdir(main_dir_path)


def run_ovsam(image: Image.Image, masks, max_samples=5):
    from image_tagger import ImageTagger
    tagger = ImageTagger()
    tags, scores = tagger.infer(image, masks, max_samples=max_samples)
    return tags


def run_metric3d(image: Image.Image, K, device):
    from infer_metric3d import predict_depth_normal
    pred_depth, pred_normal = predict_depth_normal(image, fx=K[0, 0], fy=K[1, 1], c_xy=(K[0, 2], K[1, 2]), device=device)
    return pred_depth


# ================= Gen3DSR local checkpoint override =================
# Added by local setup: force local checkpoints under external/checkpoints.
import os as _os
import shlex as _shlex
from pathlib import Path as _Path

_GEN3DSR_ROOT = _Path(__file__).resolve().parents[1]
_GEN3DSR_CKPT_DIR = _Path(
    _os.environ.get("GEN3DSR_CKPT_DIR", _GEN3DSR_ROOT / "external" / "checkpoints")
).resolve()

_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os.environ.setdefault("HF_HOME", str(_GEN3DSR_CKPT_DIR / "_hf_home"))
_os.environ.setdefault("HF_HUB_CACHE", str(_GEN3DSR_CKPT_DIR / "_hf_home" / "hub"))
_os.environ.setdefault("TRANSFORMERS_CACHE", str(_GEN3DSR_CKPT_DIR / "_hf_home" / "transformers"))
_os.environ.setdefault("TORCH_HOME", str(_GEN3DSR_CKPT_DIR / "_torch_home"))

print(f"[Gen3DSR] Local checkpoint dir: {_GEN3DSR_CKPT_DIR}")

def _gen3dsr_ckpt(name):
    name = str(name)
    direct = _GEN3DSR_CKPT_DIR / name
    if direct.exists():
        return direct

    matches = list(_GEN3DSR_CKPT_DIR.rglob(name))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"[Gen3DSR] checkpoint not found: {name}\n"
        f"Search dir: {_GEN3DSR_CKPT_DIR}\n"
        f"Run download.sh first or put the file under external/checkpoints."
    )

def _gen3dsr_find_by_url_or_name(name_or_url):
    name = str(name_or_url).split("?")[0].rstrip("/").split("/")[-1]
    return str(_gen3dsr_ckpt(name))

try:
    import torch as _torch
    import torch.hub as _torch_hub

    def _gen3dsr_local_load_state_dict_from_url(
        url,
        model_dir=None,
        map_location=None,
        progress=True,
        check_hash=False,
        file_name=None,
        weights_only=False,
    ):
        local_path = _gen3dsr_find_by_url_or_name(file_name or url)
        print(f"[Gen3DSR] torch.hub local checkpoint: {local_path}")
        try:
            return _torch.load(local_path, map_location=map_location, weights_only=weights_only)
        except TypeError:
            return _torch.load(local_path, map_location=map_location)

    _torch_hub.load_state_dict_from_url = _gen3dsr_local_load_state_dict_from_url
except Exception as _e:
    print(f"[Gen3DSR] Warning: torch.hub patch failed: {_e}")

try:
    import huggingface_hub as _hf
    import huggingface_hub.file_download as _hf_file_download

    def _gen3dsr_local_hf_hub_download(repo_id, filename=None, *args, **kwargs):
        if filename is None:
            filename = kwargs.get("filename", None)
        if filename is None and len(args) > 0:
            filename = args[0]
        if filename is None:
            raise FileNotFoundError(f"[Gen3DSR] hf_hub_download called without filename: {repo_id}")

        local_path = _gen3dsr_ckpt(filename)
        print(f"[Gen3DSR] HF local checkpoint: {local_path}")
        return str(local_path)

    def _gen3dsr_local_snapshot_download(repo_id, *args, **kwargs):
        repo_name = str(repo_id).split("/")[-1]
        direct = _GEN3DSR_CKPT_DIR / repo_name
        if direct.exists():
            print(f"[Gen3DSR] HF local snapshot: {direct}")
            return str(direct)

        matches = [p for p in _GEN3DSR_CKPT_DIR.rglob(repo_name) if p.is_dir()]
        if matches:
            print(f"[Gen3DSR] HF local snapshot: {matches[0]}")
            return str(matches[0])

        raise FileNotFoundError(
            f"[Gen3DSR] local snapshot not found for repo: {repo_id}\n"
            f"Search dir: {_GEN3DSR_CKPT_DIR}"
        )

    _hf.hf_hub_download = _gen3dsr_local_hf_hub_download
    _hf.snapshot_download = _gen3dsr_local_snapshot_download
    _hf_file_download.hf_hub_download = _gen3dsr_local_hf_hub_download
except Exception as _e:
    print(f"[Gen3DSR] Warning: huggingface_hub patch failed: {_e}")

# Override functions with robust absolute local paths.

def run_marigold(image: Image.Image, out_dir: Path):
    save_path = (out_dir / "marigold").absolute()
    save_path.mkdir(exist_ok=True)
    image.save(save_path / "input.png")

    ckpt = _gen3dsr_ckpt("marigold-lcm-v1-0")
    bash_script = (
        "python ../external/Marigold/run.py "
        f"--input_rgb_dir {_shlex.quote(str(save_path))} "
        f"--output_dir {_shlex.quote(str(save_path))} "
        f"--checkpoint {_shlex.quote(str(ckpt))} "
        "--ensemble_size 5 "
        "--denoise_steps 2"
    )
    print(bash_script)
    ret = _os.system(bash_script)
    if ret != 0:
        raise RuntimeError(f"Marigold failed with exit code {ret}")

    relative_depth = np.load(save_path / "depth_npy" / "input_pred.npy")
    return relative_depth


def run_depthanything(image: Image.Image, device):
    from zoedepth.utils.config import get_config
    from zoedepth.models.builder import build_model

    ckpt = _gen3dsr_ckpt("depth_anything_metric_depth_indoor.pt")
    config = get_config(
        "zoedepth",
        "eval",
        "nyu",
        pretrained_resource=f"local::{ckpt}",
    )
    model = build_model(config).to(device)
    return model.infer_pil(image)


def run_entityv2(image: np.ndarray, threshold=0.1, max_size=1500):
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.modeling import BACKBONE_REGISTRY, SEM_SEG_HEADS_REGISTRY
    def_keys = list(DatasetCatalog.keys())

    from detectron2.config import get_cfg
    from demo_cropformer.predictor import VisualizationDemo
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config

    weight = _gen3dsr_ckpt("CropFormer_hornet_3x_03823a.pth")
    CropFormerCfg = {
        "config_file": "../external/detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/cropformer_hornet_3x.yaml",
        "opts": ["MODEL.WEIGHTS", str(weight)],
    }

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(CropFormerCfg["config_file"])
    cfg.merge_from_list(CropFormerCfg["opts"])
    cfg.freeze()
    segmentor = VisualizationDemo(cfg)

    orig_size = image.shape
    if max(image.shape) > max_size:
        scale_factor = max_size / max(image.shape[:-1])
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

    img = image.copy()[:, :, ::-1]
    predictions = segmentor.run_on_image(img)

    extra_keys = list(DatasetCatalog.keys())
    for key in extra_keys:
        if key not in def_keys:
            DatasetCatalog.remove(key)
            MetadataCatalog.remove(key)
    BACKBONE_REGISTRY._obj_map = {}
    SEM_SEG_HEADS_REGISTRY._obj_map = {}

    pred_masks = predictions["instances"].pred_masks
    pred_scores = predictions["instances"].scores
    selected_indexes = pred_scores >= threshold
    selected_masks = pred_masks[selected_indexes]
    selected_masks = Ftv.resize(selected_masks.cpu(), orig_size[:-1]).numpy() > 0.5
    return selected_masks


def run_clipseg(image, masks):
    # Only works if you manually placed a local transformers-format CLIPSeg directory here.
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

    local_dir = _GEN3DSR_CKPT_DIR / "clipseg-rd64-refined"
    if not local_dir.exists():
        raise FileNotFoundError(
            f"[Gen3DSR] CLIPSeg local directory not found: {local_dir}\n"
            "Default config usually uses oneformer, not clipseg. "
            "Either avoid run.segmentation.fg_bg=clipseg or place CLIPSeg files locally."
        )

    processor = CLIPSegProcessor.from_pretrained(str(local_dir), local_files_only=True)
    model = CLIPSegForImageSegmentation.from_pretrained(str(local_dir), local_files_only=True)

    background_prompts = ["background", "floor", "wall", "curtain", "window", "ceiling", "table"]
    foreground_prompts = ["object", "furniture"]

    img = image
    inputs = processor(
        text=background_prompts + foreground_prompts,
        images=[image] * (len(background_prompts) + len(foreground_prompts)),
        padding="max_length",
        return_tensors="pt",
    )
    predicted = torch.sigmoid(model(**inputs).logits)
    back_pred = (predicted[:len(background_prompts)] > 0.5).any(dim=0)
    fore_pred = (predicted[-len(foreground_prompts):] > 0.1).any(dim=0)
    foreground_mask = torch.logical_or(~back_pred, fore_pred).numpy()
    foreground_mask = np.array(Image.fromarray(foreground_mask).resize(img.size))
    return filter_component_masks(masks, foreground_mask)

# ================= End Gen3DSR local checkpoint override =================

# ================= Batch-run model caches =================
# These overrides keep heavyweight helpers alive inside one Python process.
# They are intentionally module-local, so normal one-shot run.py behavior is unchanged
# except that repeated calls in the same process avoid reloading weights.

_PERSPECTIVEFIELDS_MODEL = None
_DEPTHANYTHING_MODEL = None
_ENTITYV2_SEGMENTOR = None
_CLIPSEG_PROCESSOR = None
_CLIPSEG_MODEL = None
_ONEFORMER_CACHE = None
_OVSAM_TAGGER = None
_METRIC3D_CACHE = {}


def run_perspectivefields(image):
    global _PERSPECTIVEFIELDS_MODEL
    from perspective2d import PerspectiveFields
    from perspective2d.utils.utils import general_vfov_to_focal

    if _PERSPECTIVEFIELDS_MODEL is None:
        _PERSPECTIVEFIELDS_MODEL = PerspectiveFields(
            "Paramnet-360Cities-edina-uncentered"
        ).eval().cuda()

    img_bgr = image[..., ::-1]
    H, W, _ = img_bgr.shape
    pred = _PERSPECTIVEFIELDS_MODEL.inference(img_bgr=img_bgr)

    roll = np.radians(pred["pred_roll"].cpu().item())
    pitch = np.radians(pred["pred_pitch"].cpu().item())
    vfov = np.radians(pred["pred_general_vfov"].cpu().item())
    cx_rel = pred["pred_rel_cx"].cpu().item()
    cy_rel = pred["pred_rel_cy"].cpu().item()
    focal_rel = general_vfov_to_focal(cx_rel, cy_rel, 1, vfov, degree=False)
    f = focal_rel * H
    cx = (cx_rel + 0.5) * W
    cy = (cy_rel + 0.5) * H
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    return K, roll, pitch


def run_depthanything(image: Image.Image, device):
    global _DEPTHANYTHING_MODEL
    if _DEPTHANYTHING_MODEL is None:
        from zoedepth.utils.config import get_config
        from zoedepth.models.builder import build_model

        ckpt = _gen3dsr_ckpt("depth_anything_metric_depth_indoor.pt")
        config = get_config(
            "zoedepth",
            "eval",
            "nyu",
            pretrained_resource=f"local::{ckpt}",
        )
        _DEPTHANYTHING_MODEL = build_model(config).to(device).eval()
    return _DEPTHANYTHING_MODEL.infer_pil(image)


def run_entityv2(image: np.ndarray, threshold=0.1, max_size=1500):
    global _ENTITYV2_SEGMENTOR
    if _ENTITYV2_SEGMENTOR is None:
        from detectron2.config import get_cfg
        from demo_cropformer.predictor import VisualizationDemo
        from detectron2.projects.deeplab import add_deeplab_config
        from mask2former import add_maskformer2_config

        weight = _gen3dsr_ckpt("CropFormer_hornet_3x_03823a.pth")
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(
            "../external/detectron2/projects/CropFormer/configs/entityv2/"
            "entity_segmentation/cropformer_hornet_3x.yaml"
        )
        cfg.merge_from_list(["MODEL.WEIGHTS", str(weight)])
        cfg.freeze()
        _ENTITYV2_SEGMENTOR = VisualizationDemo(cfg)

    orig_size = image.shape
    if max(image.shape) > max_size:
        scale_factor = max_size / max(image.shape[:-1])
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

    predictions = _ENTITYV2_SEGMENTOR.run_on_image(image.copy()[:, :, ::-1])
    pred_masks = predictions["instances"].pred_masks
    pred_scores = predictions["instances"].scores
    selected_masks = pred_masks[pred_scores >= threshold]
    selected_masks = Ftv.resize(selected_masks.cpu(), orig_size[:-1]).numpy() > 0.5
    return selected_masks


def run_clipseg(image, masks):
    global _CLIPSEG_PROCESSOR, _CLIPSEG_MODEL
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

    if _CLIPSEG_PROCESSOR is None or _CLIPSEG_MODEL is None:
        local_dir = _GEN3DSR_CKPT_DIR / "clipseg-rd64-refined"
        if not local_dir.exists():
            raise FileNotFoundError(
                f"[Gen3DSR] CLIPSeg local directory not found: {local_dir}\n"
                "Either avoid run.segmentation.fg_bg=clipseg or place CLIPSeg files locally."
            )
        _CLIPSEG_PROCESSOR = CLIPSegProcessor.from_pretrained(
            str(local_dir), local_files_only=True
        )
        _CLIPSEG_MODEL = CLIPSegForImageSegmentation.from_pretrained(
            str(local_dir), local_files_only=True
        ).eval()

    background_prompts = ["background", "floor", "wall", "curtain", "window", "ceiling", "table"]
    foreground_prompts = ["object", "furniture"]
    inputs = _CLIPSEG_PROCESSOR(
        text=background_prompts + foreground_prompts,
        images=[image] * (len(background_prompts) + len(foreground_prompts)),
        padding="max_length",
        return_tensors="pt",
    )
    with torch.no_grad():
        predicted = torch.sigmoid(_CLIPSEG_MODEL(**inputs).logits)
    back_pred = (predicted[:len(background_prompts)] > 0.5).any(dim=0)
    fore_pred = (predicted[-len(foreground_prompts):] > 0.1).any(dim=0)
    foreground_mask = torch.logical_or(~back_pred, fore_pred).numpy()
    foreground_mask = np.array(Image.fromarray(foreground_mask).resize(image.size))
    return filter_component_masks(masks, foreground_mask)


def run_oneformer(image, masks, device):
    global _ONEFORMER_CACHE
    if _ONEFORMER_CACHE is None:
        _ONEFORMER_CACHE = initialize_oneformer(device)
    predictor, metadata, thing_classes_ids = _ONEFORMER_CACHE

    W, H = image.size
    f = 640 * 4 / W
    img = np.array(image.resize((int(W * f), int(H * f)), Image.BILINEAR))[:, :, ::-1]
    predictions = predictor(img, "semantic")["sem_seg"].argmax(dim=0).cpu().numpy()
    is_thing = Image.fromarray(np.isin(predictions, thing_classes_ids))
    is_thing = np.array(is_thing.resize((W, H), Image.NEAREST))
    return filter_component_masks(masks, is_thing)


def run_ovsam(image: Image.Image, masks, max_samples=5):
    global _OVSAM_TAGGER
    if _OVSAM_TAGGER is None:
        from image_tagger import ImageTagger

        _OVSAM_TAGGER = ImageTagger()
    tags, scores = _OVSAM_TAGGER.infer(image, masks, max_samples=max_samples)
    return tags


def _get_metric3d_model(model_selection, device):
    key = (model_selection, str(device))
    if key in _METRIC3D_CACHE:
        return _METRIC3D_CACHE[key]

    try:
        from mmcv.utils import Config
    except Exception:
        from mmengine import Config
    from mono.model.monodepth_model import get_configured_monodepth_model
    from mono.utils.running import load_ckpt

    if model_selection == "vit-small":
        cfg = Config.fromfile("../external/Metric3D/mono/configs/HourglassDecoder/vit.raft5.small.py")
        ckpt = str(_gen3dsr_ckpt("metric_depth_vit_small_800k.pth"))
    elif model_selection == "vit-large":
        cfg = Config.fromfile("../external/Metric3D/mono/configs/HourglassDecoder/vit.raft5.large.py")
        ckpt = str(_gen3dsr_ckpt("metric_depth_vit_large_800k.pth"))
    elif model_selection == "vit-giant":
        cfg = Config.fromfile("../external/Metric3D/mono/configs/HourglassDecoder/vit.raft5.giant2.py")
        ckpt = str(_gen3dsr_ckpt("metric_depth_vit_giant2_800k.pth"))
    else:
        raise NotImplementedError(f"Unknown Metric3D model: {model_selection}")

    model = get_configured_monodepth_model(cfg)
    model, _, _, _ = load_ckpt(ckpt, model, strict_match=False)
    model.eval().to(device)
    _METRIC3D_CACHE[key] = (model, cfg)
    return _METRIC3D_CACHE[key]


def run_metric3d(image: Image.Image, K, device, model_selection="vit-large"):
    from mono.utils.do_test import transform_test_data_scalecano, get_prediction

    model, cfg = _get_metric3d_model(model_selection, device)
    cv_image = np.array(image)
    img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    intrinsic = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
    rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(
        img, intrinsic, cfg.data_basic
    )

    with torch.no_grad():
        pred_depth, pred_depth_scale, scale, output, confidence = get_prediction(
            model=model,
            input=rgb_input,
            cam_model=cam_models_stacks,
            pad_info=pad,
            scale_info=label_scale_factor,
            gt_depth=None,
            normalize_scale=cfg.data_basic.depth_range[1],
            ori_shape=[img.shape[0], img.shape[1]],
        )

    pred_depth = pred_depth.squeeze().cpu().numpy()
    pred_depth[pred_depth < 0] = 0
    return pred_depth

# ================= End batch-run model caches =================
