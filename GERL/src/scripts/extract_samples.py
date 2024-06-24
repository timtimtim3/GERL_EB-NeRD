import argparse
import os
import sys
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Set GERL environment variable if not set
if "GERL" not in os.environ:
    os.environ["GERL"] = "/home/scur1584"

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

ROOT_PATH = os.environ["GERL"]

def print_negative_pool_distribution(df, column_name, root_path):
    """
    Print and visualize the distribution of lengths of sets in a specified column,
    and save the figure to the specified root path.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the column.
    column_name (str): The name of the column containing sets.
    root_path (str): The root directory where the figure will be saved.
    """
    # Calculate the lengths of the sets in the specified column
    negative_pool_lengths = df[column_name].apply(len)

    # Print the distribution of lengths
    length_distribution = negative_pool_lengths.value_counts().sort_index()
    print(length_distribution)

    # Visualize the distribution
    plt.figure(figsize=(10, 6))
    plt.bar(length_distribution.index, length_distribution.values)
    plt.xlabel('Length of Negative Pool')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {column_name} Lengths')

    # Save the figure
    fig_path = os.path.join(root_path, f'distribution_of_{column_name}_lengths.png')
    plt.savefig(fig_path)
    print(f"Figure saved at {fig_path}")
    plt.close()


# Sample negative samples
def sample_negatives(negatives, n_samples, shuffle, with_replacement, seed):
    if len(negatives) == 0:
        return []
    negatives = list(negatives)
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(negatives)
    if with_replacement:
        return list(np.random.choice(negatives, n_samples, replace=True))
    else:
        if len(negatives) < n_samples:
            return list(np.random.choice(negatives, n_samples, replace=True))
        else:
            return list(np.random.choice(negatives, n_samples, replace=False))


def get_negatives(behavior_df, click_history, n_neg_samples, shuffle=True, with_replacement=True, seed=None):
    behavior_df['article_ids_inview_set'] = behavior_df['article_ids_inview'].apply(set)

    # Merge the DataFrames on the user ID
    behavior_df = behavior_df.merge(click_history, left_on='user_id', right_on='uid', how='left')

    # Compute the difference between 'article_ids_inview_set' and 'hist'
    behavior_df['negative_pool'] = behavior_df.apply(
        lambda row: row['article_ids_inview_set'] - row['hist'] if pd.notnull(row['hist']) else row['article_ids_inview_set'],
        axis=1
    )

    behavior_df['negative_samples'] = behavior_df['negative_pool'].apply(
        lambda x: sample_negatives(x, n_neg_samples, shuffle, with_replacement, seed)
    )

    # Drop the 'article_ids_inview_set' column
    behavior_df.drop(columns=['article_ids_inview_set', 'hist', 'article_id', 'impression_time', 
                                'read_time', 'scroll_percentage', 'device_type', 'article_ids_inview', 
                                'is_sso_user', 'gender', 'postcode', 'age', 'is_subscriber', 'session_id', 
                                'next_read_time', 'next_scroll_percentage', 'uid'], inplace=True)
    # behavior_df['uid'] = behavior_df['uid'].astype(str)
    behavior_df = behavior_df.rename(columns={'user_id': 'uid'})
    return behavior_df


def expand_rows_with_multiple_clicks(df):
    # Explode the article_ids_clicked column
    expanded_df = df.explode('article_ids_clicked').reset_index(drop=True)
    return expanded_df


def main(cfg):
    f_train_behaviors = os.path.join(ROOT_PATH, f"data/ebnerd_{cfg.fsize}", "train/behaviors.parquet")
    f_dev_behaviors = os.path.join(ROOT_PATH, f"data/ebnerd_{cfg.fsize}", "validation/behaviors.parquet")

    train_behavior = pd.read_parquet(f_train_behaviors)
    dev_behavior = pd.read_parquet(f_dev_behaviors)

    print(train_behavior.head())
    print(train_behavior.columns)
    # print(len(train_behavior))
    # print(len(train_behavior['user_id'].unique()))
    # print(train_behavior[train_behavior['user_id'] == 1001055]['article_ids_clicked'])

    f_train_hist = os.path.join(ROOT_PATH, f"data/ebnerd_{cfg.fsize}", "train/behavior_histories.parquet")
    f_dev_hist = os.path.join(ROOT_PATH, f"data/ebnerd_{cfg.fsize}", "validation/behavior_histories.parquet")
    
    train_click_hist = pd.read_parquet(f_train_hist)
    dev_click_hist = pd.read_parquet(f_dev_hist)

    # Convert the 'hist' column from lists to sets
    train_click_hist['hist'] = train_click_hist['hist'].apply(set)
    dev_click_hist['hist'] = dev_click_hist['hist'].apply(set)

    def print_diff_lengths(df):
        list_lengths = df['article_ids_clicked'].apply(len)

        # Get the distribution of lengths
        length_distribution = list_lengths.value_counts()

        # Print the distribution
        print("Distribution of lengths of lists in 'article_ids_clicked':")
        print(length_distribution)

        for length in length_distribution.index:
            example_row = df[list_lengths == length].iloc[0]
            print(f"Length: {length}, Frequency: {length_distribution[length]}")
            print(f"Example row:\n{example_row}\n")
    

    # print_diff_lengths(train_behavior)
    # print_diff_lengths(dev_behavior)

    train_behavior = expand_rows_with_multiple_clicks(train_behavior)
    dev_behavior = expand_rows_with_multiple_clicks(dev_behavior)

    # print(train_behavior['article_ids_clicked'].head())

    # Example usage
    train_behavior = get_negatives(train_behavior, train_click_hist, n_neg_samples=4, shuffle=True, with_replacement=False, seed=123)
    dev_behavior = get_negatives(dev_behavior, dev_click_hist, n_neg_samples=4, shuffle=True, with_replacement=False, seed=123)
    print(train_behavior.head())
    print(len(train_behavior))

    train_behavior.drop(columns=['negative_pool'], inplace=True)

    # # Save the DataFrames to Parquet files
    train_path = os.path.join(ROOT_PATH, f"data/ebnerd_{cfg.fsize}", "train/samples.parquet")
    dev_path = os.path.join(ROOT_PATH, f"data/ebnerd_{cfg.fsize}", "validation/samples.parquet")

    train_behavior.to_parquet(train_path, index=False)
    dev_behavior.to_parquet(dev_path, index=False)

    # print_negative_pool_distribution(train_behavior, 'negative_pool', ROOT_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path options.
    parser.add_argument("--fsize", default="large", type=str,
                        help="Corpus size")
    args = parser.parse_args()

    main(args)
