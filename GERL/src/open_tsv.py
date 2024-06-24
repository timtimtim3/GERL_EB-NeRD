import pandas as pd

# Load the data from the TSV file
file_path = "/home/scur1584/data/ebnerd_demo/examples/training_examples.tsv"
df = pd.read_json(file_path, lines=True)

# Filter rows where the length of 'target_news' is 1
rows_with_length_one = df[df['target_news'].apply(len) != 5]

# Print the filtered rows
print(rows_with_length_one)

# Load the Parquet file
parquet_file_path = "/home/scur1584/data/ebnerd_demo/train/behaviors.parquet"
parquet_df = pd.read_parquet(parquet_file_path)
print(parquet_df.head()['user_id'])

# Filter rows with user id 540765
user_rows = parquet_df[parquet_df['user_id'] == 255480]['article_ids_inview']

# Print the filtered rows
print(user_rows)
