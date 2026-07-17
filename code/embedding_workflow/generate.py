"""Run one resumable state row-tile through the inference engine."""
import argparse
import subprocess
import sys
import os
from pathlib import Path
from models import get_model

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tile-index", type=int, default=0)
    p.add_argument("--num-tiles", type=int, default=1)
    p.add_argument("--year", type=int, default=2022)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--test-chips", type=int)
    args = p.parse_args()
    spec = get_model(args.model)
    engine = Path(__file__).resolve().parents[1] / "embedding_generation" / "embed.py"
    command = [sys.executable, str(engine), "--model", spec.family, "--input", str(args.input),
               "--output", str(args.output), "--tile-index", str(args.tile_index),
               "--num-tiles", str(args.num_tiles), "--year", str(args.year),
               "--batch-size", str(args.batch_size or spec.batch_size),
               "--checkpoint-every", str(args.checkpoint_every)]
    if spec.variant:
        command += ["--variant", spec.variant]
    if args.test_chips is not None:
        command += ["--test-chips", str(args.test_chips)]
    env = os.environ.copy()
    env.setdefault("EMBEDDING_COG_INTERLEAVE", "band")
    subprocess.run(command, check=True, env=env)

if __name__ == "__main__":
    main()
