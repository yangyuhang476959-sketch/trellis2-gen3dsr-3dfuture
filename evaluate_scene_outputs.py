import argparse
import csv
import json
import os
import random
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import trimesh
from tqdm import tqdm

EVALSCENE_ROOT = Path(__file__).resolve().parent / "scenegen_evalscene"
sys.path.insert(0, str(EVALSCENE_ROOT))

from systems.scene_system import SceneSystem  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_file(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def valid_glb(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    size = path.stat().st_size
    if size < 12:
        return False
    try:
        with path.open("rb") as f:
            magic, _, declared_len = struct.unpack("<4sII", f.read(12))
            if magic != b"glTF" or declared_len != size:
                return False
            pos = 12
            while pos < size:
                header = f.read(8)
                if len(header) != 8:
                    return False
                chunk_len, _ = struct.unpack("<II", header)
                pos += 8
                if pos + chunk_len > size:
                    return False
                f.seek(chunk_len, os.SEEK_CUR)
                pos += chunk_len
            return pos == size
    except OSError:
        return False


def sorted_scene_ids(root: Path, limit: int) -> list[str]:
    ids = sorted(p.name for p in root.iterdir() if p.is_dir())
    return ids[:limit] if limit and limit > 0 else ids


def read_scene_id_file(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pick_scene_ids(root: Path, limit: int, scene_id_file: Path | None = None) -> list[str]:
    if scene_id_file is not None:
        ids = read_scene_id_file(scene_id_file)
        return ids[:limit] if limit and limit > 0 else ids
    return sorted_scene_ids(root, limit)


def prepare_flat_input(method: str, scene_ids: list[str], source_fn, work_dir: Path) -> Path:
    input_dir = work_dir / method
    reset_dir(input_dir)
    for scene_id in scene_ids:
        src = source_fn(scene_id)
        if valid_glb(src):
            link_file(src, input_dir / f"{scene_id}.glb")
    return input_dir


def _mesh_from_scene_or_mesh(path: Path) -> trimesh.Trimesh | None:
    loaded = trimesh.load(path, force="scene")
    meshes = []
    for node_name in loaded.graph.nodes_geometry:
        transform, geometry_name = loaded.graph.get(node_name)
        geometry = loaded.geometry[geometry_name]
        if isinstance(geometry, trimesh.Trimesh):
            mesh = geometry.copy()
            mesh.apply_transform(transform)
            meshes.append(mesh)
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]


def count_scene_geometries(path: Path) -> int | None:
    if not valid_glb(path):
        return None
    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        return len(loaded.graph.nodes_geometry)
    if isinstance(loaded, trimesh.Trimesh):
        return 1
    return None


def prepare_gen3dsr_ordered_input(
    scene_ids: list[str],
    gen3dsr_dir: Path,
    gt_scene_dir: Path,
    work_dir: Path,
    cache_dir: Path,
    require_gt_object_count_match: bool,
) -> Path:
    input_dir = work_dir / "Gen3DSR_ordered_assets"
    reset_dir(input_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    skipped_count_mismatch = []
    for scene_id in scene_ids:
        reconstruction_dir = gen3dsr_dir / scene_id / "reconstruction"
        if not reconstruction_dir.exists():
            continue
        asset_paths = sorted(
            [
                p
                for p in reconstruction_dir.glob("*.glb")
                if p.name != "full_scene.glb"
                and p.stem.split("_", 1)[0].isdigit()
                and valid_glb(p)
            ],
            key=lambda p: int(p.stem.split("_", 1)[0]),
        )

        gt_count = count_scene_geometries(gt_scene_dir / f"{scene_id}.glb")
        if require_gt_object_count_match and (gt_count is None or len(asset_paths) != gt_count):
            skipped_count_mismatch.append((scene_id, len(asset_paths), gt_count))
            continue

        cache_path = cache_dir / f"{scene_id}.glb"
        if valid_glb(cache_path):
            link_file(cache_path, input_dir / f"{scene_id}.glb")
            continue
        if cache_path.exists():
            cache_path.unlink()

        if not asset_paths:
            full_scene = reconstruction_dir / "full_scene.glb"
            if valid_glb(full_scene):
                link_file(full_scene, cache_path)
                link_file(cache_path, input_dir / f"{scene_id}.glb")
            continue

        scene = trimesh.Scene()
        try:
            for asset_path in asset_paths:
                mesh = _mesh_from_scene_or_mesh(asset_path)
                if mesh is not None:
                    scene.add_geometry(mesh, node_name=asset_path.stem)
            if len(scene.geometry) > 0:
                scene.export(cache_path)
                link_file(cache_path, input_dir / f"{scene_id}.glb")
        except Exception as exc:
            print(f"Skipping Gen3DSR ordered scene {scene_id}: {exc}")
    if skipped_count_mismatch:
        print(
            "Skipped Gen3DSR scenes with asset/GT count mismatch: "
            + ", ".join(f"{sid}({pred}!={gt})" for sid, pred, gt in skipped_count_mismatch)
        )
    return input_dir


def flatten_scene_glb(src: Path, dst: Path) -> None:
    loaded = trimesh.load(src)
    if isinstance(loaded, trimesh.Scene):
        meshes = []
        for node_name in loaded.graph.nodes_geometry:
            transform = loaded.graph.get(node_name)[0]
            geometry_name = loaded.graph.get(node_name)[1]
            geometry = loaded.geometry[geometry_name]
            if isinstance(geometry, trimesh.Trimesh):
                mesh = geometry.copy()
                mesh.apply_transform(transform)
                meshes.append(mesh)
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = loaded
    mesh.export(dst)


def prepare_flattened_gt(scene_ids: list[str], gt_scene_dir: Path, work_dir: Path) -> Path:
    out_dir = work_dir / "gt_flattened"
    reset_dir(out_dir)
    for scene_id in tqdm(scene_ids, desc="Flatten evaluation GT", unit="scene"):
        src = gt_scene_dir / f"{scene_id}.glb"
        if valid_glb(src):
            flatten_scene_glb(src, out_dir / f"{scene_id}.glb")
    return out_dir


def run_official_eval(
    method: str,
    input_dir: Path,
    gt_scene_dir: Path,
    args,
    eval_object_level: bool = True,
    pred_camera_to_meshlab: bool = False,
    scene_alignment: bool | None = None,
    asset_alignment: bool | None = None,
    use_aligned_cache: bool = True,
) -> dict:
    input_scene_count = len(list(input_dir.glob("*.glb")))
    if input_scene_count == 0:
        return {
            "Method": method,
            "CD-S ↓": None,
            "CD-O ↓": None,
            "F-Score-S ↑": None,
            "F-Score-O ↑": None,
            "IoU-B ↑": None,
            "CLIP-S ↑": None,
            "DINO-S ↑": None,
            "Time ↓": None,
            "Scene N": 0,
        }

    save_dir = args.out_dir / "official_eval_raw" / method
    save_dir.mkdir(parents=True, exist_ok=True)

    if scene_alignment is None:
        scene_alignment = args.scene_alignment
    if asset_alignment is None:
        asset_alignment = args.asset_alignment

    cfg = SceneSystem.Config(
        input_dir=str(input_dir),
        output_dir=str(gt_scene_dir),
        save_dir=str(save_dir),
        input_format="glb",
        gt_format="glb",
        gpu_num=1,
        gpu_id=0,
        num_points=args.num_points,
        chunk_size=args.chunk_size,
        eval_scene_level=True,
        eval_object_level=eval_object_level,
        alignment_mode=args.alignment_mode,
        alignment_max_iterations=args.alignment_max_iterations,
        alignment_max_points=args.alignment_max_points,
        alignment_tolerance=args.alignment_tolerance,
        fscore_threshold=args.fscore_threshold,
        scene_alignment=scene_alignment,
        scene_visualize=args.scene_visualize,
        asset_alignment=asset_alignment,
        asset_visualize=args.asset_visualize,
        render_eval=False,
        filename="metrics_scenegen",
        metrics=["scene_cd", "scene_fscore", "object_cd", "object_fscore", "iou_bbox"],
        device=args.device,
        normalize_pred_scene=args.normalize_pred_scene,
        pred_target_extent=args.pred_target_extent,
        pred_camera_to_meshlab=pred_camera_to_meshlab,
        pred_rotate_x_deg=args.pred_rotate_x_deg,
        pred_rotate_z_deg=args.pred_rotate_z_deg,
        save_aligned_limit=args.save_aligned_limit if args.save_aligned_method == "all" or method == args.save_aligned_method else 0,
        save_aligned_dir=str(args.save_aligned_dir / method),
        aligned_cache_dir=str(args.aligned_cache_dir / method),
        use_aligned_cache=use_aligned_cache,
        cleanup_interval=args.cleanup_interval,
    )
    system = SceneSystem(cfg)
    system.setup()
    system.test_directory()
    system.on_test_end()
    scene_count = system.successful_test_count

    csv_path = save_dir / "metrics_scenegen.csv"
    if not csv_path.exists():
        return {"Method": method, "error": f"missing metrics csv: {csv_path}"}
    row = pd.read_csv(csv_path).iloc[0].to_dict()
    out = {
        "Method": method,
        "CD-S ↓": float(row["scene_cd"]) if "scene_cd" in row else None,
        "CD-O ↓": float(row["object_cd"]) if eval_object_level and "object_cd" in row else None,
        "F-Score-S ↑": float(row["scene_fscore"]) * 100 if "scene_fscore" in row else None,
        "F-Score-O ↑": float(row["object_fscore"]) * 100 if eval_object_level and "object_fscore" in row else None,
        "IoU-B ↑": float(row["iou_bbox"]) if eval_object_level and "iou_bbox" in row else None,
        "CLIP-S ↑": None,
        "DINO-S ↑": None,
        "Time ↓": None,
        "Scene N": scene_count,
    }
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("No rows.\n")
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header)) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Thin adapter around copied SceneGen evalscene code.")
    parser.add_argument("--trellis-dir", type=Path, default=Path("outputs/trellis2"))
    parser.add_argument("--gen3dsr-dir", type=Path, default=Path("outputs/gen3dsr"))
    parser.add_argument("--gt-scene-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--scene-id-file", type=Path, default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["TRELLIS2_scene_masked", "Gen3DSR_flipped_aligned"],
        choices=["TRELLIS2_scene_masked", "Gen3DSR_flipped_aligned", "Gen3DSR_raw_aligned"],
    )
    parser.add_argument("--trellis-limit", type=int, default=0)
    parser.add_argument("--gen3dsr-limit", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=10240)
    parser.add_argument("--chunk-size", type=int, default=10240)
    parser.add_argument("--fscore-threshold", type=float, default=0.1)
    parser.add_argument("--alignment-mode", choices=["filterreg", "cpd"], default="filterreg")
    parser.add_argument("--scene-alignment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--asset-alignment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--alignment-max-iterations", type=int, default=200)
    parser.add_argument("--alignment-max-points", type=int, default=0)
    parser.add_argument("--alignment-tolerance", type=float, default=1e-3)
    parser.add_argument("--normalize-pred-scene", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pred-target-extent", type=float, default=1.96)
    parser.add_argument("--pred-rotate-x-deg", type=float, default=0.0)
    parser.add_argument("--pred-rotate-z-deg", type=float, default=0.0)
    parser.add_argument("--save-aligned-limit", type=int, default=0)
    parser.add_argument("--save-aligned-method", default="Gen3DSR_flipped_aligned")
    parser.add_argument("--save-aligned-dir", type=Path, default=None)
    parser.add_argument("--aligned-cache-dir", type=Path, default=Path("outputs/eval_cache/aligned"))
    parser.add_argument("--ordered-gen3dsr-cache-dir", type=Path, default=Path("outputs/eval_cache/ordered_gen3dsr"))
    parser.add_argument("--gen3dsr-require-gt-object-count-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cleanup-interval", type=int, default=30)
    parser.add_argument("--scene-visualize", action="store_true")
    parser.add_argument("--asset-visualize", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.save_aligned_dir is None:
        args.save_aligned_dir = args.out_dir / "aligned_debug"
    for name in ("trellis_dir", "gen3dsr_dir", "gt_scene_dir", "out_dir", "save_aligned_dir", "aligned_cache_dir", "ordered_gen3dsr_cache_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    selected_methods = set(args.methods)
    required_dirs = [args.gt_scene_dir]
    if "TRELLIS2_scene_masked" in selected_methods:
        required_dirs.append(args.trellis_dir)
    if selected_methods - {"TRELLIS2_scene_masked"}:
        required_dirs.append(args.gen3dsr_dir)
    for path in required_dirs:
        if not path.is_dir():
            parser.error(f"Directory not found: {path}")

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    set_seed(args.seed)

    work_dir = args.out_dir / "official_eval_inputs"
    reset_dir(work_dir)

    selected_methods = set(args.methods)
    trellis_ids = pick_scene_ids(args.trellis_dir, args.trellis_limit, args.scene_id_file) if "TRELLIS2_scene_masked" in selected_methods else []
    gen3dsr_ids = pick_scene_ids(args.gen3dsr_dir, args.gen3dsr_limit, args.scene_id_file) if selected_methods - {"TRELLIS2_scene_masked"} else []
    trellis_eval_ids = [
        scene_id
        for scene_id in trellis_ids
        if "TRELLIS2_scene_masked" in selected_methods
        and valid_glb(args.trellis_dir / scene_id / "scene_masked.glb")
        and valid_glb(args.gt_scene_dir / f"{scene_id}.glb")
    ]
    trellis_gt_dir = prepare_flattened_gt(trellis_eval_ids, args.gt_scene_dir, work_dir) if "TRELLIS2_scene_masked" in selected_methods else None

    trellis_methods = []
    if "TRELLIS2_scene_masked" in selected_methods:
        trellis_methods.append(
            (
                "TRELLIS2_scene_masked",
                prepare_flat_input("TRELLIS2_scene_masked", trellis_eval_ids, lambda sid: args.trellis_dir / sid / "scene_masked.glb", work_dir),
            )
        )

    gen3dsr_methods = []
    if "Gen3DSR_flipped_aligned" in selected_methods or "Gen3DSR_raw_aligned" in selected_methods:
        gen3dsr_input = prepare_gen3dsr_ordered_input(
            gen3dsr_ids,
            args.gen3dsr_dir,
            args.gt_scene_dir,
            work_dir,
            args.ordered_gen3dsr_cache_dir,
            args.gen3dsr_require_gt_object_count_match,
        )
        if "Gen3DSR_flipped_aligned" in selected_methods:
            gen3dsr_methods.append(
                (
                    "Gen3DSR_flipped_aligned",
                    gen3dsr_input,
                    True,
                    True,
                    True,
                )
            )
        if "Gen3DSR_raw_aligned" in selected_methods:
            gen3dsr_methods.append(
                (
                    "Gen3DSR_raw_aligned",
                    gen3dsr_input,
                    False,
                    True,
                    True,
                )
            )

    rows = []
    for method, input_dir in trellis_methods:
        rows.append(
            run_official_eval(
                method,
                input_dir,
                trellis_gt_dir,
                args,
                eval_object_level=False,
                pred_camera_to_meshlab=False,
                scene_alignment=True,
                asset_alignment=True,
                use_aligned_cache=True,
            )
        )
    for method, input_dir, pred_camera_to_meshlab, scene_alignment, asset_alignment in gen3dsr_methods:
        rows.append(
            run_official_eval(
                method,
                input_dir,
                args.gt_scene_dir,
                args,
                eval_object_level=True,
                pred_camera_to_meshlab=pred_camera_to_meshlab,
                scene_alignment=scene_alignment,
                asset_alignment=asset_alignment,
                use_aligned_cache=True,
            )
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table_csv = args.out_dir / "scenegen_style_table.csv"
    table_md = args.out_dir / "scenegen_style_table.md"
    summary_json = args.out_dir / "scene_eval_summary.json"
    write_csv(table_csv, rows)
    write_markdown(table_md, rows)
    summary_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"Wrote table:   {table_csv}")
    print(f"Wrote table:   {table_md}")
    print(f"Wrote summary: {summary_json}")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
