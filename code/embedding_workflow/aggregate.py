"""Stable entry point for census-tract aggregation."""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parents[1] / "embedding_generation" / "aggregate.py"), run_name="__main__")
