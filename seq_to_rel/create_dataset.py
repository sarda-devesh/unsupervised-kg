import pandas as pd
import os
import argparse
import json

def read_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--load_directory', type= str, required = True, help = "The directory containing the dataset generated from snippets API")
    parser.add_argument('--save_directory', type= str, required = True, help = "The directory where we want to save the dataset")
    return parser.parse_args()

def generate_sequence_dataset(src_file, dst_folder):
    os.makedirs(dst_folder, exist_ok = True)
    src_dir = os.path.dirname(src_file)
    save_file_name = os.path.basename(src_file).replace(".txt", ".tsv")
    save_path = os.path.join(dst_folder, save_file_name)
    
    dfs_to_merge = []
    print("Generating dataset from", os.path.basename(src_file))
    with open(src_file, 'r') as reader:
        for line in reader:
            line = line.strip()
            if len(line) == 0:
                continue
            
            # Load the df
            df_path = os.path.join(src_dir, line)
            df = pd.read_csv(df_path)

            # Create the rows for the sequence df
            seq_rows = []
            for idx, row in df.iterrows():
                text = row["text"].replace("\n", " ").replace("\t", " ").replace("  ", " ")
                r_type = row["type"]
                src_val, dst_val = row["src"], row["dst"]

                # Get the src and dst type
                src_type, dst_type = "", ""
                if "_to_" in r_type:
                    src_type, dst_type = r_type.split("_to_")
                else:
                    src_type, dst_type = "lith", "lith_att"

                seq_relationship = f"{src_val} @{src_type}@ {dst_val} @{dst_type}@ @{r_type}@"
                seq_rows.append([text, seq_relationship])
            
            seq_df = pd.DataFrame(seq_rows, columns = ["text", "relationship"])
            dfs_to_merge.append(seq_df)
        
        # Merge and save the df
        merged_df = pd.concat(dfs_to_merge)
        merged_df.to_csv(save_path, sep='\t', index=False, header=False)

def main():
    args = read_args()
    src_dir, save_dir = args.load_directory, args.save_directory
    generate_sequence_dataset(os.path.join(src_dir, "train.txt"), save_dir)
    generate_sequence_dataset(os.path.join(src_dir, "valid.txt"), save_dir)
    generate_sequence_dataset(os.path.join(src_dir, "test.txt"), save_dir)

def get_example():
    train_file = "paragraph_data/train.tsv"
    df = pd.read_csv(train_file, sep = '\t', header = None, names = ["text", "relationship"])
    df["r_type"] = df["relationship"].str.split(" ").str[-1]
    for relationship, rows in df.groupby(by = ["r_type"]):
        sample = rows.sample(n = 15, weights = df.text.str.len())
        for idx, row in sample.iterrows():
            print("------------ START -------------")
            print("Relationship", row["relationship"])
            print(row["text"])
            print("------------ END -------------")

if __name__ == "__main__":
    get_example()