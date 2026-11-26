"""AnyCam entry point: python run_anycam.py {fuji,m9,m10r,m240p} --raw RAW --dcp DCP [options]."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anycam.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
