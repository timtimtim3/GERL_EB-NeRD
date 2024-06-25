import argparse
import subprocess
import os
import sys


def main(cfg):
    fsize_arg = cfg.fsize

    # Base paths from the environment variables
    base_path = cfg.base_path

    python_env = f"{base_path}{cfg.conda_env}"
    scripts_path = f"{base_path}/GERL/src/scripts"

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    # List of commands with f-string for the argument
    commands = [
        f"{python_env} {scripts_path}/extract_behavior_histories.py --fsize {fsize_arg}",
        f"{python_env} {scripts_path}/extract_samples.py --fsize {fsize_arg}",
        f"{python_env} {scripts_path}/build_vocabs.py --fsize {fsize_arg}",
        f"{python_env} {scripts_path}/build_word_emb.py --fsize {fsize_arg}",
        f"{python_env} {scripts_path}/build_doc_emb.py --fsize {fsize_arg}",
        f"{python_env} {scripts_path}/build_neighbors.py --fsize {fsize_arg}",
        f"{python_env} {scripts_path}/build_training_examples.py --fsize {fsize_arg}",
        f"{python_env} {scripts_path}/build_eval_examples.py --fsize {fsize_arg}"
    ]

    # Run each command sequentially
    for command in commands:
        subprocess.run(command, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path options.
    parser.add_argument("--fsize", default="small", type=str,
                        help="Corpus size")
    parser.add_argument("--base_path", default="/home/scur1584", type=str,
                        help="Base path.")
    parser.add_argument("--conda_env", default="/.conda/envs/recsys/bin/python", type=str,
                        help="Path of conda env.")
    args = parser.parse_args()
    
    main(args)