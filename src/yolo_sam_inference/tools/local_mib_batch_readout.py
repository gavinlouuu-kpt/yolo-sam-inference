# import pandas as pd
# from minio import Minio
# from minio.error import S3Error
# from concurrent.futures import ThreadPoolExecutor
# from tqdm import tqdm
# import sys

# def get_batch_data(project_path: str) -> pd.DataFrame:
#     client = Minio(
#         "localhost:9000",
#         access_key="mibadmin",
#         secret_key="cuhkminio",
#         secure=False
#     )
    
#     bucket_name = "erb-g07"
#     def process_batch(batch_num: int):
#         batch_path = f"{project_path}/batch_{batch_num}/batch_data.csv"
#         try:
#             data = client.get_object(bucket_name, batch_path)
#             df = pd.read_csv(data)
#             print(f"SUCCESS: Loaded {batch_path} with columns: {df.columns.tolist()}", file=sys.stderr)
#             return df
#         except S3Error as e:
#             if "NoSuchKey" in str(e):
#                 return None
#             print(f"ERROR processing {batch_path}: {e}", file=sys.stderr)
#             return None
#         except Exception as e:
#             print(f"CRITICAL ERROR in {batch_path}: {e}", file=sys.stderr)
#             return None

#     try:
#         max_batches = 100
#         batch_nums = range(1, max_batches + 1)

#         print(f"\n=== PHASE 1: Fetching batch files ===", file=sys.stderr)
#         with ThreadPoolExecutor(max_workers=10) as executor:
#             results = list(tqdm(executor.map(process_batch, batch_nums), total=len(batch_nums), file=sys.stderr))

#         print("\n=== PHASE 2: Validating DataFrames ===", file=sys.stderr)
#         valid_dfs = [df for df in results if df is not None]
#         print(f"Found {len(valid_dfs)} valid DataFrames", file=sys.stderr)

#         if not valid_dfs:
#             print("ERROR: No valid DataFrames found", file=sys.stderr)
#             return pd.DataFrame()

#         print("\nFirst DataFrame schema:", file=sys.stderr)
#         print(valid_dfs[0].dtypes, file=sys.stderr)

#         print("\n=== PHASE 3: Concatenation ===", file=sys.stderr)
#         try:
#             combined_df = pd.concat(valid_dfs, ignore_index=True)
#             print(f"Combined DataFrame shape: {combined_df.shape}", file=sys.stderr)

#             if combined_df.empty:
#                 print("WARNING: Combined DataFrame is empty. Columns:", combined_df.columns.tolist(), file=sys.stderr)
#                 print("Sample from first DataFrame:", file=sys.stderr)
#                 print(valid_dfs[0].head(), file=sys.stderr)

#             return combined_df
#         except Exception as e:
#             print(f"CONCATENATION ERROR: {e}", file=sys.stderr)
#             return pd.DataFrame()

#     except Exception as e:
#         print(f"\nFATAL ERROR: {e}", file=sys.stderr)
#         return pd.DataFrame()

# if __name__ == "__main__":
#     project_path = "Qinru/2025_cells/ms_20250326_231_10ul_BONE"
#     df = get_batch_data(project_path)
#     print(df.head(100))

#     if not df.empty:
#         print(f"\nSUCCESS: Final DataFrame has {len(df)} rows", file=sys.stderr)
#     else:
#         print("\nFAILURE: Final DataFrame is empty", file=sys.stderr)

import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import sys
import io
import argparse
import os

def get_batch_data(project_path: str) -> pd.DataFrame:
    def process_batch(batch_num: int):
        batch_path = os.path.join(project_path, f"batch_{batch_num}", "batch_data.csv")
        try:
            if not os.path.exists(batch_path):
                return None
            df = pd.read_csv(batch_path)
            print(f"SUCCESS: Loaded {batch_path} with columns: {df.columns.tolist()}", file=sys.stderr)
            return df
        except Exception as e:
            print(f"CRITICAL ERROR in {batch_path}: {e}", file=sys.stderr)
            return None

    try:
        max_batches = 100
        batch_nums = range(1, max_batches + 1)
        print(f"\n=== PHASE 1: Fetching batch files ===", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(tqdm(executor.map(process_batch, batch_nums), total=len(batch_nums), file=sys.stderr))

        print("\n=== PHASE 2: Validating DataFrames ===", file=sys.stderr)
        valid_dfs = [df for df in results if df is not None]
        print(f"Found {len(valid_dfs)} valid DataFrames", file=sys.stderr)
        if not valid_dfs:
            print("ERROR: No valid DataFrames found", file=sys.stderr)
            return pd.DataFrame()

        print("\nFirst DataFrame schema:", file=sys.stderr)
        print(valid_dfs[0].dtypes, file=sys.stderr)

        print("\n=== PHASE 3: Concatenation ===", file=sys.stderr)
        try:
            combined_df = pd.concat(valid_dfs, ignore_index=True)
            print(f"Combined DataFrame shape: {combined_df.shape}", file=sys.stderr)
            if combined_df.empty:
                print("WARNING: Combined DataFrame is empty. Columns:", combined_df.columns.tolist(), file=sys.stderr)
                print("Sample from first DataFrame:", file=sys.stderr)
                print(valid_dfs[0].head(), file=sys.stderr)
                return pd.DataFrame()

            # Save the combined DataFrame locally
            output_path = os.path.join(project_path, "combined_output.csv")
            combined_df.to_csv(output_path, index=False)
            print(f"\nSUCCESS: Saved combined DataFrame to {output_path}", file=sys.stderr)

            return combined_df
        except Exception as e:
            print(f"CONCATENATION ERROR: {e}", file=sys.stderr)
            return pd.DataFrame()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}", file=sys.stderr)
        return pd.DataFrame()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process batch data from local storage')
    parser.add_argument('project_path', type=str, help='Path to the project directory containing batch folders')
    args = parser.parse_args()

    if not os.path.exists(args.project_path):
        print(f"ERROR: Project path '{args.project_path}' does not exist", file=sys.stderr)
        sys.exit(1)

    df = get_batch_data(args.project_path)
    print(df.head(100))
    if not df.empty:
        print(f"\nSUCCESS: Final DataFrame has {len(df)} rows", file=sys.stderr)
    else:
        print("\nFAILURE: Final DataFrame is empty", file=sys.stderr)