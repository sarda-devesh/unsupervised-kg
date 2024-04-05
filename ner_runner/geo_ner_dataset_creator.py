import os
import json
import requests
import pandas as pd
import random
import spacy

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

class Node:

    def __init__(self):
        self.children = {}
        self.possible_labels = set()

def create_trie():
    root = Node()
    terms_path = os.path.join("macrostrat_strat_trees", "terms.csv")
    terms_df = pd.read_csv(terms_path)
    terms_df = terms_df[["term_type", "term"]]
    terms_df = terms_df.drop_duplicates()
    print("Number of terms is", len(terms_df.index))

    for idx, row in terms_df.iterrows():
        term_type, term_name = row["term_type"], row["term"]
        name_parts = term_name.split(" ")

        # Start the process
        current_node = root
        for name in name_parts:
            name = name.strip().lower()
            if len(name) == 0:
                continue
            
            if name not in current_node.children:
                current_node.children[name] = Node()
            current_node = current_node.children[name]

        current_node.possible_labels.add(term_type)
        
    return root, pd.unique(terms_df["term"])

def get_snippets_for_term(term):
    term_name = term.replace(" ", "%20")
    relations_json = requests.get(f"https://xdd.wisc.edu/api/snippets?term={term_name}&inclusive=true&clean=true").json()
    all_paragraphs = []
    if 'success' in relations_json and 'data' in relations_json['success']: 
        data = relations_json['success']['data']
        for document in data:
            doc_id = document["_gddid"]
            for curr_sentence in document["highlight"]:
                if term.lower() in curr_sentence.lower():
                    all_paragraphs.append(curr_sentence.replace("\n", " "))

    return all_paragraphs

nlp = spacy.load("en_core_web_sm")

def get_row_for_paragraph(paragraph, root, writer):
    words = nlp(paragraph)
    words = [token.text.strip() for token in words]
    idx = 0

    while idx < len(words):
        # Perform the strip
        words[idx] = words[idx].strip()
        if len(words[idx]) == 0:
            idx += 1
            continue

        # Check if word is in trie 
        start_idx, most_recent_idx, end_idx = idx, idx, idx
        label = "O"

        curr_node = root
        while end_idx < len(words) and words[end_idx].lower() in curr_node.children:
            curr_node = curr_node.children[words[end_idx].lower()]
            if curr_node.label is not None:
                label = "I-" + str(curr_node.label)
                most_recent_idx = end_idx
            
            end_idx += 1
        
        # Write the result
        for write_idx in range(start_idx, most_recent_idx + 1):
            label_to_write = label
            if write_idx == start_idx:
                label_to_write = label.replace("I-", "B-")
            writer.write(words[write_idx] + " " + label_to_write + "\n")

        idx = most_recent_idx + 1

def create_dataset_from_paragraphs(paragraphs, root, save_path):
    print("Saving", len(paragraphs), "paragraphs to path", save_path)
    with open(save_path, 'w+') as writer:
        for paragraph in paragraphs:
            get_row_for_paragraph(paragraph, root, writer)
            writer.write(" \n")

def main():
    trie, terms = create_trie()
    print("Got a total of", len(terms), "terms")

    # Get all of the paragraphs and save it
    ner_dir = "macrostart_ner"
    term_save_path = os.path.join(ner_dir, "paragraphs.txt")
    if not os.path.exists(term_save_path):
        # Got all of the paragraphs based on the terms
        with open(term_save_path, 'w+') as writer:
            for term in terms:
                matching_paragraphs = get_snippets_for_term(term)
                print("Got", len(matching_paragraphs), "paragraphs for term", term)
        
                # Save the paragraphs
                for paragraph in matching_paragraphs:
                    writer.write(paragraph + "\n")
    
    # Load the paragraphs
    paragraphs = None
    with open(term_save_path, 'r') as reader:
        paragraphs = reader.readlines()
    random.shuffle(paragraphs)

    # Create the train dataset
    num_trains = int(0.85 * len(paragraphs))
    create_dataset_from_paragraphs(paragraphs[ : num_trains], trie, os.path.join(ner_dir, "train.txt"))

    # Create the test dataset
    num_test = int(0.1 * len(paragraphs))
    create_dataset_from_paragraphs(paragraphs[num_trains : num_trains + num_test], trie, os.path.join(ner_dir, "test.txt"))

    # Create the dev dataset
    create_dataset_from_paragraphs(paragraphs[num_trains + num_test : ], trie, os.path.join(ner_dir, "dev.txt"))

def fuzzy_string_matching():
    trie, terms = create_trie()
    print(trie.children["rhyolite"].children.keys())

if __name__ == "__main__":
    fuzzy_string_matching()