import json
import os
import pandas as pd

def create_df():
    rows = []

    paragraphs_dir = "formation_sample_paragraphs"
    for file_name in os.listdir(paragraphs_dir):
        if ".json" not in file_name or file_name[0] == '.':
            continue
        
        # Read the formation name
        formation_name = file_name[ : file_name.index(".")].replace("_", " ")
        file_path = os.path.join(paragraphs_dir, file_name)
        data = None
        try:
            with open(file_path, 'r') as reader:
                data = json.load(reader)
        except:
            continue
        
        if data is None or "matching_paragraphs" not in data:
            continue
        
        for para_data in data["matching_paragraphs"]:
            paper_id = para_data["paper_id"]
            paragraph = para_data["paragraph"]
            rows.append([formation_name, paper_id, paragraph])
    
    df = pd.DataFrame(rows, columns = ["formation_name", "paper_id", "paragraph"])
    print("Saving", len(df.index), "example rows")
    df.to_parquet('formation_sample.parquet.gzip', compression='gzip')

if __name__ == "__main__":
    create_df()