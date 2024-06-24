import argparse
import os
import sys
import pandas as pd

# Set GERL environment variable if not set
if "GERL" not in os.environ:
    os.environ["GERL"] = "/home/scur1584"

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

ROOT_PATH = os.environ["GERL"]

def print_value_counts_and_nans(df, column_name):
    value_counts = df[column_name].value_counts(dropna=False)
    nan_percentage = df[column_name].isna().mean() * 100
    print(f"Value counts for {column_name}:\n{value_counts}\n")
    print(f"Percentage of NaNs for {column_name}: {nan_percentage:.2f}%\n")

def collapse_user_features(df, user_id_col):
    # Forward fill and backward fill the DataFrame
    df = df.sort_values(user_id_col).ffill().bfill()
    # Group by user_id and take the first non-NaN value for each column
    collapsed_df = df.groupby(user_id_col, as_index=False).first()
    return collapsed_df

def main(cfg):
    f_train_behaviors = os.path.join(ROOT_PATH, f"data/ebnerd_{cfg.fsize}", "train/behaviors.parquet")
    f_dev_behaviors = os.path.join(ROOT_PATH, f"data/ebnerd_{cfg.fsize}", "validation/behaviors.parquet")

    train_behavior = pd.read_parquet(f_train_behaviors)
    dev_behavior = pd.read_parquet(f_dev_behaviors)

    columns_to_extract = ['user_id', 'is_sso_user', 'is_subscriber', 'gender', 'postcode', 'age', 
                          'read_time', 'scroll_percentage']

    # Extract user-specific features
    train_user_features = train_behavior[columns_to_extract]
    dev_user_features = dev_behavior[columns_to_extract]

    # # Collapse user-specific features
    # train_user_features_collapsed = collapse_user_features(train_user_features, 'user_id')
    # dev_user_features_collapsed = collapse_user_features(dev_user_features, 'user_id')

    # # Print value counts and percentage of NaNs for each column in training data
    # print("Training Data Value Counts and NaN Percentages:")
    # for column in columns_to_extract[1:]:  # Skip 'user_id'
    #     print_value_counts_and_nans(train_user_features_collapsed, column)
    
    # Print value counts for 'read_time' and 'scroll_percentage'
    print("Training Data Value Counts for 'read_time' and 'scroll_percentage':")
    print_value_counts_and_nans(train_behavior, 'read_time')
    print_value_counts_and_nans(train_behavior, 'scroll_percentage')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path options.
    parser.add_argument("--fsize", default="small", type=str,
                        help="Corpus size")
    args = parser.parse_args()

    main(args)
