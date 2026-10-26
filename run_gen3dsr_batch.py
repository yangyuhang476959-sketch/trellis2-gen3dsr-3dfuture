import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run bundled Gen3DSR with dataset instance masks.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/gen3dsr"))
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--gt-json", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=root / "gen3dsr/src/configs/image.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-id")
    parser.add_argument("--end-id")
    parser.add_argument("--force", action="store_true")
    args, extras = parser.parse_known_args()
    for name in ("input_root", "checkpoint_dir"):
        if not getattr(args, name).expanduser().is_dir():
            parser.error(f"Directory not found: {getattr(args, name)}")
    for name in ("gt_json", "config"):
        if not getattr(args, name).expanduser().is_file():
            parser.error(f"File not found: {getattr(args, name)}")

    command = [sys.executable, "batch_run_gen3dsr_scenegen_masks.py"]
    for option, value in (("data_root", args.input_root), ("out_root", args.output_root), ("gt_json", args.gt_json), ("config", args.config)):
        command.extend([f"--{option}", str(value.expanduser().resolve())])
    for option, value in (("limit", args.limit), ("start_id", args.start_id), ("end_id", args.end_id)):
        if value is not None:
            command.extend([f"--{option}", str(value)])
    if args.force:
        command.append("--force")
    env = os.environ.copy()
    env["GEN3DSR_CKPT_DIR"] = str(args.checkpoint_dir.expanduser().resolve())
    result = subprocess.run(command + extras, cwd=root / "gen3dsr/src", env=env)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
