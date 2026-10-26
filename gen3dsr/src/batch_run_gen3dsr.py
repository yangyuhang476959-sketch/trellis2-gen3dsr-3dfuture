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
from scipy.ndimage import binary_opening

from dataset import get_scene
from run import (
    complete_crop,
    get_background_mesh,
    get_camera,
    get_components_masks,
    get_depth,
    reconstruct_object,
)
from util import (
    align_to_depth,
    align_to_depth_rep,
    clean,
    clip_object_with_bg,
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


def scene_status(out_dir):
    return out_dir / "logs" / "status.json"


def write_status(out_dir, **payload):
    path = scene_status(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def make_scene_opt(base_opt, img_path, out_dir):
    opt = OmegaConf.create(OmegaConf.to_container(base_opt, resolve=True))
    opt.scene.attributes.img_path = str(img_path)
    opt.scene.save_dir = str(out_dir)
    return opt


def add_mesh_to_scene(scene, mesh):
    if isinstance(mesh, trimesh.Scene):
        scene.add_geometry(mesh.dump(concatenate=False))
    else:
        scene.add_geometry([mesh])


def reconstruct_scene(opt, device, crop_size, shared_models):
    print("Running Gen3DSR with config")
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
    masks, object_ids, background_ids, instance_labels = get_components_masks(
        scene, opt.run.segmentation, out_dir, device
    )

    pts3d = depth_to_points(depth_map[None], K_img)
    trimesh.PointCloud(
        pts3d.reshape(-1, 3), scene.image_np.reshape(-1, 3)
    ).export(out_dir / "depth_scene.ply")
    cam_params = {
        "K": K_img.tolist(),
        "c2w": pose.tolist(),
        "W": scene.image_pil.width,
        "H": scene.image_pil.height,
    }
    with open(out_dir / "cam_params.json", "w", encoding="utf-8") as fp:
        json.dump(cam_params, fp)

    bg_path = out_dir / "reconstruction" / "background.ply"
    if not opt.run.only_foreground:
        background_mesh = get_background_mesh(
            scene.image_np,
            pts3d,
            np.any(masks[background_ids], axis=0),
            K_img,
            depth_map,
            device,
        )
        background_mesh.apply_transform(pose)
        background_mesh.export(bg_path)
    else:
        background_mesh = None

    K_crop = get_crop_calibration(crop_size)
    acompletion_p = shared_models.get_acompletion()
    zero123_p = shared_models.get_zero123()

    if opt.run.single_object:
        masks = np.any(masks[object_ids], axis=0)[None]
        label = ",".join([instance_labels[i] for i in object_ids])
        label = label.replace(" (", ", ").replace(")", "")
        instance_labels = [label]
        object_ids = np.array([0])

    scene_mesh = trimesh.Scene()
    if background_mesh is not None:
        add_mesh_to_scene(scene_mesh, background_mesh)

    for i in range(len(masks[object_ids]) - 1, -1, -1):
        label = instance_labels[object_ids[i]]
        label = label.replace(" (", ", ").replace(")", "")
        obj_id = f"{i}_{label.replace(' ', '_')}"
        final_obj_path = out_dir / "reconstruction" / f"{obj_id}.glb"

        mask = binary_opening(masks[object_ids][i], np.ones((7, 7)))
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

        obj_space_dir = out_dir / "object_space" / f"{obj_id}"
        elev_path = obj_space_dir / "estimated_elevation.npy"
        estimate_elevation(
            np.array(full_crop.resize((256, 256))),
            obj_space_dir,
            zero123_p,
        )
        obj_elevation = np.load(elev_path)

        try:
            clean()
            obj_mesh = reconstruct_object(opt.run, out_dir, obj_id, obj_elevation, opt.run.reproject)
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
        if opt.run.get("clip_bg", False):
            clip_object_with_bg(obj_mesh, background_mesh)
        obj_mesh.export(final_obj_path)
        add_mesh_to_scene(scene_mesh, obj_mesh)
        print(f"Saved, {obj_id}.glb")

    scene_mesh.export(out_dir / "reconstruction" / "full_scene.glb")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--config", default="./configs/image.yaml")
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
    print(f"[INFO] GPU_ID    = {args.gpu_idx}")
    print(f"[INFO] FORCE     = {args.force}")
    print(f"[INFO] SCENES    = {len(scenes)}")
    print("============================================================")

    for scene_dir in scenes:
        scene_id = scene_dir.name
        img_path = scene_dir / "scene.jpg"
        out_dir = out_root / scene_id
        final_glb = out_dir / "reconstruction" / "full_scene.glb"
        log_dir = out_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "run.log"

        print("============================================================")
        print(f"[INFO] Scene: {scene_id}")
        print(f"[INFO] IMG  : {img_path}")
        print(f"[INFO] OUT  : {out_dir}")

        if not img_path.is_file():
            print(f"[SKIP] scene.jpg not found: {img_path}")
            write_status(out_dir, scene_id=scene_id, status="skipped", reason="missing image")
            counts["skipped"] += 1
            continue

        if not args.force and final_glb.is_file():
            print(f"[SKIP] already done: {final_glb}")
            write_status(out_dir, scene_id=scene_id, status="skipped", reason="already done")
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
                    reconstruct_scene(
                        opt,
                        device,
                        args.crop_size,
                        shared_models,
                    )
            if final_glb.is_file():
                print(f"[OK] {scene_id}")
                write_status(out_dir, scene_id=scene_id, status="success")
                counts["success"] += 1
            else:
                print(f"[FAIL] {scene_id}, final glb missing")
                write_status(out_dir, scene_id=scene_id, status="failed", reason="final glb missing")
                counts["failed"] += 1
        except Exception as exc:
            print(f"[FAIL] {scene_id}: {exc}")
            traceback.print_exc()
            write_status(out_dir, scene_id=scene_id, status="failed", reason=repr(exc))
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
