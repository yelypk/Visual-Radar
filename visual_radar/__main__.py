from .cli import build_parser, args_to_config, run

def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = args_to_config(args)
    run(cfg)

if __name__ == "__main__":
    main()