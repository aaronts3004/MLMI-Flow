from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
GENERATED = DATA / "generated"
CHECKPOINTS = ROOT / "checkpoints"
OUTPUTS = ROOT / "outputs"
CONFIGS = ROOT / "configs"