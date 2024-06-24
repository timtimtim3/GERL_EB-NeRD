import argparse
import os
import sys
from collections import defaultdict
import pandas as pd


# Set GERL environment variable if not set
if "GERL" not in os.environ:
    os.environ["GERL"] = "/home/scur1584"

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

ROOT_PATH = os.environ["GERL"]


def get_user_hist_from_behaviors_slow(behavior_df):
    # Initialize a dictionary to store the history for each user
    user_history = defaultdict(set)

    # Iterate over each row to collect 'article_ids_clicked' for each user
    for idx, row in behavior_df.iterrows():
        user_id = row['user_id']
        article_ids_clicked = row['article_ids_clicked']
        
        # Add each clicked article to the user's history set
        for article_id in article_ids_clicked:
            user_history[user_id].add(article_id)

    # Convert sets to lists of integers
    user_history = {user_id: list(article_ids) for user_id, article_ids in user_history.items()}

    # # Example: Print the history for a specific user
    # specific_user_id = 1001055
    # print(f"History for user {specific_user_id}: {user_history.get(specific_user_id, [])}")

    # Optionally, you can convert the user history to a DataFrame for further processing or saving
    user_history_df = pd.DataFrame({
        'uid': list(user_history.keys()),
        'hist': list(user_history.values())
    })
    return user_history_df


def get_user_hist_from_behaviors(behavior_df):
    # Extract the single article_id from the list
    behavior_df['article_ids_clicked'] = behavior_df['article_ids_clicked'].str[0]

    # Group by 'user_id' and aggregate the 'article_ids_clicked' and 'read_time' into lists
    user_history_df = behavior_df.groupby('user_id').agg({
        'article_ids_clicked': list,
        'read_time': list
    }).reset_index()

    # Rename columns for consistency
    user_history_df.rename(columns={'user_id': 'uid', 'article_ids_clicked': 'hist'}, inplace=True)

    return user_history_df


def print_value_counts_and_nans(df, column_name):
    value_counts = df[column_name].value_counts(dropna=False)
    nan_percentage = df[column_name].isna().mean() * 100
    print(f"Value counts for {column_name}:\n{value_counts}\n")
    print(f"Percentage of NaNs for {column_name}: {nan_percentage:.2f}%\n")


def main(cfg):
    f_train_behaviors = os.path.join(ROOT_PATH, f"data/ebnerd_{cfg.fsize}", "train/behaviors.parquet")
    train_behavior = pd.read_parquet(f_train_behaviors)
    f_dev_behaviors = os.path.join(ROOT_PATH, f"data/ebnerd_{cfg.fsize}", "validation/behaviors.parquet")
    dev_behavior = pd.read_parquet(f_dev_behaviors)

    train_behavior['article_ids_clicked_len'] = train_behavior['article_ids_clicked'].apply(len)
    dev_behavior['article_ids_clicked_len'] = dev_behavior['article_ids_clicked'].apply(len)
    print_value_counts_and_nans(train_behavior, 'article_ids_clicked_len')

    train_behavior = train_behavior[train_behavior['article_ids_clicked_len'] == 1]
    dev_behavior = dev_behavior[dev_behavior['article_ids_clicked_len'] == 1]

    # Remove the helper column
    train_behavior.drop(columns=['article_ids_clicked_len'], inplace=True)
    dev_behavior.drop(columns=['article_ids_clicked_len'], inplace=True)

    if cfg.add_21_day_history:
        # TODO: add 21 day history, note we must do the same in build_vocabs.py
        pass

    # Generate user history
    train_hist = get_user_hist_from_behaviors(train_behavior)
    dev_hist = get_user_hist_from_behaviors(dev_behavior)

    # Identify duplicate uids
    duplicate_uids = train_hist[train_hist['uid'].duplicated(keep=False)]

    # Print the duplicate uid values
    print("Duplicate uid values:")
    print(duplicate_uids['uid'].unique())

    # Save the DataFrames to Parquet files
    train_hist_path = os.path.join(ROOT_PATH, f"data/ebnerd_{cfg.fsize}", "train/readtime_histories.parquet")
    dev_hist_path = os.path.join(ROOT_PATH, f"data/ebnerd_{cfg.fsize}", "validation/readtime_histories.parquet")

    train_hist.to_parquet(train_hist_path, index=False)
    dev_hist.to_parquet(dev_hist_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path options.
    parser.add_argument("--fsize", default="small", type=str,
                        help="Corpus size")
    parser.add_argument("--add_21_day_history", action="store_true", default=False,
                        help="Flag to include 21-day click history prior to the behavior logs. Default is False.")
    args = parser.parse_args()

    main(args)
