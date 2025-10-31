import yaml
from src.experiment import Experiment

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    exp = Experiment(cfg)
    exp.run()

if __name__ == "__main__":
    main()