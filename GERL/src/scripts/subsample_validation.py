import os
import random
import argparse
import sys
import pandas as pd
import json

# Set GERL environment variable if not set
if "GERL" not in os.environ:
    os.environ["GERL"] = "/home/scur1584"

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

random.seed(7)
ROOT_PATH = os.environ["GERL"]

def main(cfg):
    f_examples = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", "examples", "eval_examples.tsv")

    # Read the eval examples
    df = pd.read_csv(f_examples, sep='\t', header=None, names=["data"])

    # Convert the 'data' column from JSON strings to dictionaries
    df['data'] = df['data'].apply(json.loads)

    # Expand the 'data' column into separate columns
    expanded_df = pd.json_normalize(df['data'])

    # Group by user and subsample a fixed amount of users
    unique_users = expanded_df['user'].unique()
    sampled_users = random.sample(list(unique_users), min(cfg.num_users, len(unique_users)))

    # Select all rows that have user IDs that we sampled
    sampled_df = expanded_df[expanded_df['user'].isin(sampled_users)]

    # Write the filtered data to a new TSV file
    f_output = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", "examples", "eval_examples_subsampled.tsv")
    with open(f_output, 'w', encoding='utf-8') as f:
        for _, row in sampled_df.iterrows():
            row_dict = row.to_dict()
            json_str = json.dumps(row_dict, separators=(',', ':'))
            f.write(f"{json_str}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path options.
    parser.add_argument("--fsize", default="small", type=str,
                        help="Corpus size")
    parser.add_argument("--num_users", default=20000, type=int,
                        help="Number of users to subsample")
    args = parser.parse_args()

    main(args)
