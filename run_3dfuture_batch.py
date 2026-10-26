import argparse
import csv
import os
import time
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from PIL import ImageChops
from PIL import Image
import torch

from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel


def find_scenes(input_root: Path, start: int = 0, limit: int = None):
    seen = 0
    yielded = 0
    for scene_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        scene_path = scene_dir / "scene.jpg"
        if not scene_path.exists():
            continue
        if seen < start:
            seen += 1
            continue
        if limit is not None and yielded >= limit:
            return
        seen += 1
        yielded += 1
        yield scene_dir.name, scene_path


def make_masked_scene(scene_dir: Path, scene_path: Path) -> Image.Image:
    scene = Image.open(scene_path).convert("RGBA")
    mask_paths = sorted(scene_dir.glob("*_mask.png"))
    if not mask_paths:
        raise FileNotFoundError(f"No *_mask.png files found in {scene_dir}")

    alpha = Image.new("L", scene.size, 0)
    for mask_path in mask_paths:
        mask = Image.open(mask_path).convert("L")
        if mask.size != scene.size:
            mask = mask.resize(scene.size, Image.Resampling.NEAREST)
        alpha = ImageChops.lighter(alpha, mask)

    alpha = alpha.point(lambda value: 255 if value > 0 else 0)
    black = Image.new("RGBA", scene.size, (0, 0, 0, 0))
    masked = Image.composite(scene, black, alpha)
    masked.putalpha(alpha)
    return masked


def export_mesh(mesh, out_path: Path, texture_size: int, decimation_target: int, remesh: bool):
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=remesh,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    glb.export(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Directory containing 3D-FUTURE masked image folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/trellis2"),
        help="Output directory.",
    )
    parser.add_argument("--pretrained", default="pretrained/TRELLIS.2-4B")
    parser.add_argument("--pipeline-type", default="1024_cascade")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-num-tokens", type=int, default=49152)
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--decimation-target", type=int, default=1000000)
    parser.add_argument("--no-remesh", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.input_root = args.input_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if not args.input_root.is_dir():
        parser.error(f"Input directory not found: {args.input_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    log_path = args.output_root / "run_log.csv"
    timing_log_path = args.output_root / "timing_log.csv"

    print(f"Input:  {args.input_root}", flush=True)
    print(f"Output: {args.output_root}", flush=True)
    print("Loading pipeline once...", flush=True)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.pretrained)
    pipeline.cuda()

    scenes = find_scenes(args.input_root, start=args.start, limit=args.limit)
    planned = "unknown" if args.limit is None else str(args.limit)
    print("Input variants: scene, scene_masked", flush=True)
    print(f"Processing scene stream; planned scene count: {planned}", flush=True)

    with log_path.open("a", newline="") as f, timing_log_path.open("a", newline="") as timing_f:
        writer = csv.DictWriter(f, fieldnames=["scene", "image", "output", "status", "error"])
        if f.tell() == 0:
            writer.writeheader()
        timing_writer = csv.DictWriter(
            timing_f,
            fieldnames=["scene", "variant", "image", "output", "status", "elapsed_seconds", "error"],
        )
        if timing_f.tell() == 0:
            timing_writer.writeheader()

        for idx, (scene_id, image_path) in enumerate(scenes, start=args.start):
            out_dir = args.output_root / scene_id
            out_dir.mkdir(parents=True, exist_ok=True)

            for output_name, image_source in (
                ("scene", lambda: Image.open(image_path)),
                ("scene_masked", lambda: make_masked_scene(image_path.parent, image_path)),
            ):
                out_path = out_dir / f"{output_name}.glb"

                if out_path.exists() and not args.overwrite:
                    print(f"[{idx}] skip existing: {out_path}", flush=True)
                    writer.writerow(
                        {"scene": scene_id, "image": f"{image_path}#{output_name}", "output": str(out_path), "status": "skip", "error": ""}
                    )
                    timing_writer.writerow(
                        {
                            "scene": scene_id,
                            "variant": output_name,
                            "image": f"{image_path}#{output_name}",
                            "output": str(out_path),
                            "status": "skip",
                            "elapsed_seconds": "",
                            "error": "",
                        }
                    )
                    f.flush()
                    timing_f.flush()
                    continue

                print(f"[{idx}] {image_path}#{output_name} -> {out_path}", flush=True)
                start_time = time.perf_counter()
                status = "ok"
                error = ""
                try:
                    image = image_source()
                    input_preview_path = out_dir / f"{output_name}_input.png"
                    image.save(input_preview_path)
                    mesh = pipeline.run(
                        image,
                        seed=args.seed,
                        pipeline_type=args.pipeline_type,
                        max_num_tokens=args.max_num_tokens,
                    )[0]
                    mesh.simplify(16777216)
                    export_mesh(
                        mesh,
                        out_path,
                        texture_size=args.texture_size,
                        decimation_target=args.decimation_target,
                        remesh=not args.no_remesh,
                    )
                    writer.writerow(
                        {"scene": scene_id, "image": f"{image_path}#{output_name}", "output": str(out_path), "status": "ok", "error": ""}
                    )
                except Exception as e:
                    status = "error"
                    error = repr(e)
                    print(f"[{idx}] ERROR {output_name}: {e}", flush=True)
                    writer.writerow(
                        {"scene": scene_id, "image": f"{image_path}#{output_name}", "output": str(out_path), "status": "error", "error": error}
                    )
                finally:
                    elapsed = time.perf_counter() - start_time
                    timing_writer.writerow(
                        {
                            "scene": scene_id,
                            "variant": output_name,
                            "image": f"{image_path}#{output_name}",
                            "output": str(out_path),
                            "status": status,
                            "elapsed_seconds": f"{elapsed:.6f}",
                            "error": error,
                        }
                    )
                    f.flush()
                    timing_f.flush()
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
