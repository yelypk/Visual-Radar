import logging
from visual_radar.cli import main

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def run():
    setup_logging()
    main()

if __name__ == "__main__":
    run()