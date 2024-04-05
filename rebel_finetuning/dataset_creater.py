import os
import pandas as pd
import numpy as np
import argparse
import requests
from multiprocessing import Pool
import time
import random
import weaviate
from nltk.tokenize import sent_tokenize

def create_tokenizer(graph_path):
    save_path = 'data/archive_token_terms.txt'
    os.makedirs(os.path.dirname(save_path), exist_ok = True)

    if not os.path.exists(save_path):
        database_terms_df = pd.read_csv(graph_path)
        database_terms_df.dropna()
        unique_terms = set(database_terms_df["edge_src"]).union(set(database_terms_df["edge_dst"]))
        terms_arr = np.array(list(unique_terms))
        np.savetxt(save_path, terms_arr, delimiter=" ", fmt="%s") 

def get_relationships_for_edge(src_edge, dst_edge, edge_type):
    try:
        src_name = src_edge.replace(" ", "%20")
        dst_name = dst_edge.replace(" ", "%20")
        request_url = f"https://xdd.wisc.edu/api/snippets?term={src_name}%2C{dst_name}&inclusive=true&clean=true"
        time.sleep(0.5)
        response = requests.get(request_url)
        relations_json = response.json()
        all_pairs = []
        if 'success' in relations_json and 'data' in relations_json['success']: 
            data = relations_json['success']['data']
            for document in data:
                doc_title = document["title"]
                doc_id = document["_gddid"]
                for curr_sentence in document["highlight"]:
                    for piece in curr_sentence.split("."):
                        piece = piece.strip()
                        if src_edge in piece and dst_edge in piece:
                            all_pairs.append([doc_id, doc_title, piece, src_edge, dst_edge, edge_type])
        return all_pairs
    except Exception as e:
        print("Encountered error", e, "for pair", src_edge, "<->", dst_edge)
        return []

def save_rows_as_df(save_path, results_row):
    save_df = pd.DataFrame(results_row, columns = ["doc_id", "title", "text", "src", "dst", "type"])
    print("Saving file", save_path, "with", len(save_df.index), "rows")
    save_df.to_csv(save_path, index = None)

def create_dataset_using_snippets(graph_path, num_workers = 5, pairs_per_file = 2000):
    if os.path.exists("data/archive/0.csv"):
        return

    graph_df = pd.read_csv(graph_path).astype(str)
    file_id = 0
    results_row, prev_count = [], 0

    with Pool(num_workers) as pool:
        results = []
        print("DF has a total of", len(graph_df.index), "rows")
        for _, row in graph_df.iterrows():
            src_txt = str(row["edge_src"])
            dst_txt = str(row["edge_dst"])
            edge_type = str(row["edge_name"])
            if src_txt != 'nan' and dst_txt != 'nan' and edge_type != 'nan':
                results.append(pool.apply_async(get_relationships_for_edge, [src_txt, dst_txt, edge_type]))
        print("Expecting results for", len(results), "rows")

        for result in results:
            pairs = result.get()
            results_row.extend(pairs)
            if (len(results_row) > prev_count):
                prev_count = len(results_row)

            if len(results_row) >= pairs_per_file:
                save_rows_as_df(file_id, results_row)
                file_id += 1
                results_row = []
                prev_count = 0
    
    # Save the last chunk
    save_rows_as_df(file_id, results_row)

client = weaviate.Client(
    "http://cosmos0001.chtc.wisc.edu:8080",
    auth_client_secret=weaviate.auth.AuthApiKey(os.getenv("HYBRID_API_KEY")),
)
HYBRID_ENDPOINT = "http://cosmos0001.chtc.wisc.edu:4502/hybrid"
HYBRID_HEADERS = {"Content-Type": "application/json", "Api-Key": os.getenv("HYBRID_API_KEY")}
POSSIBLE_TOPICS = ["criticalmaas", "dolomites"]

def get_weave_paragraphs(src_node, dst_node, relationship_type, results_per_pair):
    where_filter = {"operator": "And", "operands": [{
        "path": ["text_content"],
        "operator": "ContainsAll",
        "valueString": [src_node, dst_node],
    }]}

    result = client.query.get("Passage", ["text_content", "paper_id"]).with_where(where_filter).with_limit(results_per_pair).do()
    valid_paragraphs = []

    if "data" in result and "Get" in result["data"] and "Passage" in result["data"]["Get"] and result["data"]["Get"]["Passage"] is not None:
        paragraphs = result["data"]["Get"]["Passage"]
        for paragraph_content in paragraphs:
            paper_id, paragraph_text = paragraph_content["paper_id"], paragraph_content["text_content"]
            paragraph_lower = paragraph_text.lower().replace("\n", " ")
            if src_node.lower() in paragraph_lower and dst_node.lower() in paragraph_lower:
                valid_paragraphs.append([paper_id, paper_id, paragraph_lower, src_node, dst_node, relationship_type])
    
    return valid_paragraphs


def create_dataset_using_weave(graph_path, save_dir, results_per_pair = 20, pairs_per_file = 2000):
    # Determine the new file ids 
    graph_df = pd.read_csv(graph_path).astype(str)
    file_id = 0
    for file_name in os.listdir(save_dir):
        if "csv" not in file_name or file_name[0] == '.':
            continue

        file_id += 1
    
    print("Determined initial file id of", file_id)

    result_row = []
    for idx, row in graph_df.iterrows():
        # Get the sentences for this pair
        src_txt = str(row["edge_src"])
        dst_txt = str(row["edge_dst"])
        edge_type = str(row["edge_name"])
        if not edge_type.startswith("att_"):
            continue

        # Get the sentences for this pair
        result_row.extend(get_weave_paragraphs(src_txt, dst_txt, edge_type, results_per_pair))
        time.sleep(0.5)
        print("Updated size to", len(result_row), "after processing idx", idx, "with content", (src_txt, edge_type, dst_txt))
        
        # Checkpoint every so often
        if len(result_row) >= pairs_per_file:
            save_path = os.path.join(save_dir, str(file_id) + ".csv")
            save_rows_as_df(save_path, result_row)
            file_id += 1
            result_row = []

def write_split_to_file(split_files, save_path):
    print("Writing split to", save_path)
    with open(save_path, 'w+') as writer:
        for file_name in split_files:
            writer.write(file_name + "\n")

def create_splits(save_dir):
    # Determine the directory path
    train_path = os.path.join(save_dir, "train.txt")
    test_path = os.path.join(save_dir, "test.txt")
    valid_path = os.path.join(save_dir, "valid.txt")
    
    # Perorm the split
    all_data_files = []
    for file_name in os.listdir(save_dir):
        if file_name[0] != '.' and "csv" in file_name:
            all_data_files.append(file_name)
    random.shuffle(all_data_files)

    # Create the split
    num_train, num_test = int(0.85 * len(all_data_files)), int(0.1 * len(all_data_files))
    train_files = all_data_files[ : num_train]
    test_files = all_data_files[num_train : num_train + num_test]
    valid_files = all_data_files[num_train + num_test : ]

    # Write split out to file
    write_split_to_file(train_files, train_path)
    write_split_to_file(test_files, test_path)
    write_split_to_file(valid_files, valid_path)

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_file', type= str, required = True, help = "The path to the graph file")
    parser.add_argument('--save_dir', type= str, required = True, help = "The directory where we want to save the results in")
    return parser.parse_args()

if __name__ == "__main__":
    args = read_args()
    create_dataset_using_weave(args.graph_file, args.save_dir)
    create_splits(args.save_dir)