import argparse
import contextlib
import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf
from PIL import Image
from scipy.ndimage import binary_opening

from dataset import get_scene
from run import complete_crop, get_camera, get_depth, reconstruct_object
from util import (
    align_to_depth,
    align_to_depth_rep,
    clean,
    crop_object,
    depth_to_points,
    estimate_elevation,
    get_crop_calibration,
    initialize_acompletion,
    initialize_zero123,
    reproject_pixels,
)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            if getattr(stream, "closed", False):
                continue
            try:
                stream.write(data)
                stream.flush()
            except ValueError:
                continue

    def flush(self):
        for stream in self.streams:
            if getattr(stream, "closed", False):
                continue
            try:
                stream.flush()
            except ValueError:
                continue


class SharedModels:
    def __init__(self, device):
        self.device = device
        self.acompletion = None
        self.zero123 = None

    def get_acompletion(self):
        if self.acompletion is None:
            self.acompletion = initialize_acompletion(self.device)
        return self.acompletion

    def get_zero123(self):
        if self.zero123 is None:
            self.zero123 = initialize_zero123(self.device)
        return self.zero123


def numeric_scene_dirs(data_root):
    return sorted(
        [p for p in data_root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )


def make_scene_opt(base_opt, img_path, out_dir):
    opt = OmegaConf.create(OmegaConf.to_container(base_opt, resolve=True))
    opt.scene.attributes.img_path = str(img_path)
    opt.scene.save_dir = str(out_dir)
    opt.run.only_foreground = True
    opt.run.clip_bg = False
    return opt


def add_mesh_to_scene(scene, mesh):
    if isinstance(mesh, trimesh.Scene):
        scene.add_geometry(mesh.dump(concatenate=False))
    else:
        scene.add_geometry([mesh])


def load_category_names(gt_json):
    if gt_json is None:
        return {}, {}

    with open(gt_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_id_by_name = {img["file_name"]: img["id"] for img in data["images"]}
    category_by_id = {cat["id"]: cat["category"] for cat in data["categories"]}
    annotations_by_image = {}
    for ann in data["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)
    return image_id_by_name, {
        image_name: [
            category_by_id.get(ann["category_id"], "object")
            for ann in annotations_by_image.get(image_id, [])
        ]
        for image_name, image_id in image_id_by_name.items()
    }


def clean_label(label):
    return (
        str(label)
        .replace(" / ", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )


def load_dataset_masks(scene_dir, scene_id, labels_by_scene):
    masks = []
    labels = []
    for mask_path in sorted(scene_dir.glob("*_mask.png"), key=lambda p: int(p.stem.split("_")[0])):
        idx = int(mask_path.stem.split("_")[0])
        mask_img = Image.open(mask_path)
        mask_np = np.array(mask_img)
        if mask_np.ndim == 3:
            mask_np = mask_np[..., -1] if mask_np.shape[-1] == 4 else mask_np.max(axis=-1)
        mask = mask_np > 0
        masks.append(mask)

        scene_labels = labels_by_scene.get(scene_id, [])
        label = scene_labels[idx] if idx < len(scene_labels) else "object"
        labels.append(label)

    if not masks:
        raise FileNotFoundError(f"No *_mask.png files found in {scene_dir}")

    return np.stack(masks), labels


def reconstruct_scene_with_dataset_masks(
    opt,
    scene_dir,
    scene_id,
    labels_by_scene,
    device,
    crop_size,
    shared_models,
):
    print("Running Gen3DSR with SceneGen dataset masks")
    print(opt)

    scene = get_scene(opt.scene.type, opt.scene.attributes)
    out_dir = Path(opt.scene.save_dir)
    print(f"Saving to {out_dir}")
    out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / "crops").mkdir(exist_ok=True)
    (out_dir / "object_space").mkdir(exist_ok=True)
    (out_dir / "reconstruction").mkdir(exist_ok=True)

    OmegaConf.save(config=opt, f=out_dir / "config.yaml")
    scene.image_pil.save(out_dir / "input.png")

    K_img, pose = get_camera(scene, opt.run.calibration)
    depth_map = get_depth(scene, opt.run.depth, K_img, out_dir, device)
    masks, instance_labels = load_dataset_masks(scene_dir, scene_id, labels_by_scene)
    print(f"Dataset foreground masks: {len(masks)}")

    pts3d = depth_to_points(depth_map[None], K_img)
    trimesh.PointCloud(
        pts3d.reshape(-1, 3), scene.image_np.reshape(-1, 3)
    ).export(out_dir / "depth_scene.ply")
    with open(out_dir / "cam_params.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "K": K_img.tolist(),
                "c2w": pose.tolist(),
                "W": scene.image_pil.width,
                "H": scene.image_pil.height,
            },
            fp,
        )

    K_crop = get_crop_calibration(crop_size)
    acompletion_p = shared_models.get_acompletion()
    zero123_p = shared_models.get_zero123()
    scene_mesh = trimesh.Scene()

    for i in range(len(masks) - 1, -1, -1):
        label = instance_labels[i]
        obj_id = f"{i}_{clean_label(label)}"

        mask = binary_opening(masks[i], np.ones((7, 7)))
        if mask.sum() < 400:
            print(f"Skipped too small object: {obj_id}")
            continue

        if opt.run.reproject:
            crop, normalization_mat, out_depth, c2w_def, mesh_pc = reproject_pixels(
                scene.image_np,
                mask,
                pts3d,
                K_crop,
                K_img,
                crop_size,
                opt.run.depth_edge_threshold,
            )
        else:
            crop, crop_params = crop_object(scene.image_np, mask, crop_size)
        crop.save(out_dir / "crops" / f"{obj_id}_reproj.png")

        full_crop = complete_crop(crop, label, acompletion_p, opt.run.amodal_completion)
        full_crop.save(out_dir / "crops" / f"{obj_id}_rgba.png")

        estimate_elevation(
            np.array(full_crop.resize((256, 256))),
            out_dir / "object_space" / f"{obj_id}",
            zero123_p,
        )
        obj_elevation = np.load(
            out_dir / "object_space" / f"{obj_id}" / "estimated_elevation.npy"
        )

        try:
            clean()
            obj_mesh = reconstruct_object(
                opt.run, out_dir, obj_id, obj_elevation, opt.run.reproject
            )
            clean()
        except Exception:
            print(f"Failed to reconstruct object {obj_id}")
            traceback.print_exc()
            continue

        if opt.run.reproject:
            transform = align_to_depth_rep(
                obj_mesh, normalization_mat, out_depth, c2w_def, K_crop, crop_size
            )
        else:
            transform = align_to_depth(mask, pts3d, obj_mesh, K_crop, crop_params, crop_size)
        obj_mesh.apply_transform(transform)
        obj_mesh.apply_transform(pose)
        obj_mesh.export(out_dir / "reconstruction" / f"{obj_id}.glb")
        add_mesh_to_scene(scene_mesh, obj_mesh)
        print(f"Saved, {obj_id}.glb")

    scene_mesh.export(out_dir / "reconstruction" / "full_scene.glb")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--config", default="./configs/image.yaml")
    parser.add_argument("--gt_json", default=None)
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--start_id", default=None)
    parser.add_argument("--end_id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args, extras = parser.parse_known_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"DATA_ROOT not found: {data_root}")
    if not Path(args.config).is_file():
        raise FileNotFoundError(f"config not found: {args.config}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    gt_json = Path(args.gt_json) if args.gt_json else None
    labels_by_scene = {}
    if gt_json is not None and gt_json.is_file():
        _, labels_by_scene = load_category_names(gt_json)
    elif gt_json is not None:
        print(f"[WARN] gt_json not found, using generic labels: {gt_json}")

    device = f"cuda:{args.gpu_idx}"
    base_opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    scenes = numeric_scene_dirs(data_root)
    if args.start_id is not None:
        scenes = [p for p in scenes if p.name >= args.start_id]
    if args.end_id is not None:
        scenes = [p for p in scenes if p.name <= args.end_id]
    if args.limit is not None:
        scenes = scenes[: args.limit]

    out_root.mkdir(parents=True, exist_ok=True)
    shared_models = SharedModels(device)
    counts = {"success": 0, "failed": 0, "skipped": 0}

    print("============================================================")
    print(f"[INFO] DATA_ROOT = {data_root}")
    print(f"[INFO] OUT_ROOT  = {out_root}")
    print(f"[INFO] CONFIG    = {args.config}")
    print(f"[INFO] GT_JSON   = {gt_json}")
    print(f"[INFO] SCENES    = {len(scenes)}")
    print("============================================================")

    for scene_dir in scenes:
        scene_id = scene_dir.name
        img_path = scene_dir / "scene.jpg"
        out_dir = out_root / scene_id
        final_glb = out_dir / "reconstruction" / "full_scene.glb"
        log_dir = out_dir / "logs"
        log_file = log_dir / "run.log"

        print("============================================================")
        print(f"[INFO] Scene: {scene_id}")
        print(f"[INFO] IMG  : {img_path}")
        print(f"[INFO] OUT  : {out_dir}")

        if not img_path.is_file():
            print(f"[SKIP] scene.jpg not found: {img_path}")
            counts["skipped"] += 1
            continue

        if not args.force and final_glb.is_file():
            print(f"[SKIP] already done: {final_glb}")
            counts["skipped"] += 1
            continue

        if out_dir.exists():
            print(f"[RESET] incomplete scene output removed: {out_dir}")
            shutil.rmtree(out_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        opt = make_scene_opt(base_opt, img_path, out_dir)
        try:
            with open(log_file, "w", encoding="utf-8") as fp:
                tee = Tee(sys.stdout, fp)
                with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                    reconstruct_scene_with_dataset_masks(
                        opt,
                        scene_dir,
                        scene_id,
                        labels_by_scene,
                        device,
                        args.crop_size,
                        shared_models,
                    )
            if final_glb.is_file():
                print(f"[OK] {scene_id}")
                counts["success"] += 1
            else:
                print(f"[FAIL] {scene_id}, final glb missing")
                counts["failed"] += 1
        except Exception as exc:
            print(f"[FAIL] {scene_id}: {exc}")
            traceback.print_exc()
            counts["failed"] += 1
            clean()

    print("==================== SUMMARY ====================")
    print(f"SUCCESS : {counts['success']}")
    print(f"FAILED  : {counts['failed']}")
    print(f"SKIPPED : {counts['skipped']}")
    print(f"TOTAL   : {len(scenes)}")
    print(f"OUT_ROOT: {out_root}")
    print("=================================================")


if __name__ == "__main__":
    main()
