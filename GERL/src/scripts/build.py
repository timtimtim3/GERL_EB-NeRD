import subprocess
import os
import sys

fsize_arg = "large"

# Set GERL environment variable if not set
if "GERL" not in os.environ:
    os.environ["GERL"] = "/home/scur1584"

# Base paths from the environment variables
base_path = os.environ["GERL"]

# Set python environment variable if not set, using GERL base path
if "PYTHON_ENV" not in os.environ:
    os.environ["PYTHON_ENV"] = f"{base_path}/.conda/envs/recsys/bin/python"

python_env = os.environ["PYTHON_ENV"]
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
