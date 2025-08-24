import logging
import os
from visual_radar.cli import main

def setup_logging():
    level_name = (os.getenv("VR_LOG_LEVEL") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname).1s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def run():
    setup_logging()
    main()

if __name__ == "__main__":
    run()