import math
import torch
from pyvis.network import Network
import os
import random
import numpy
import multiprocessing
import argparse
import json
import requests
import numpy as np
import pandas as pd
import time

from knowledge_graph import *
from model_wrapper import *

def get_model(model_type, model_path):
    if model_type == "rebel":
        return RebelWrapper(model_path)
    elif model_type == "seq2rel":
        return Seq2RelWrapper(model_path)
    else:
        raise Exception("Invalid model type of " + model_type)

def get_snippets_for_formation(formation_name, article_limit, fragment_limit):
    try:
        formatted_name = formation_name.replace(" ", "%20")
        request_url = f"https://xdd.wisc.edu/api/snippets?term={formatted_name}&inclusive=true&clean=true&article_limit={article_limit}&fragment_limit={fragment_limit}"
        response = requests.get(request_url)
        relations_json = response.json()
        all_snippets = []
        if 'success' in relations_json and 'data' in relations_json['success']: 
            data = relations_json['success']['data']
            for document in data:
                doc_id = document["_gddid"]
                for curr_sentence in document["highlight"]:
                   all_snippets.append((doc_id, curr_sentence))
        return all_snippets
    except Exception as e:
        print("Encountered error", e, "for formation", formation_name)
        return []

def get_models(model_types, model_paths):
    models = []
    num_models = len(model_types)
    for idx in range(num_models):
        models.append(get_model(model_types[idx], model_paths[idx]))
    return models

def get_kg_for_formation(models, formation_name, article_limit = 10, fragment_limit = 5):
    metrics = {}

    try:
        # Load the snippets
        start_time = time.time()
        snippets = get_snippets_for_formation(formation_name, article_limit, fragment_limit)
        if len(snippets) == 0:
            return {
                "result" : "error",
                "reason" : "Failed to get snippets for formation " + str(formation)
            }
        metrics["snippets_fetch_time"] = str(round(time.time() - start_time, 3)) + " seconds"
        
        # Create the kg
        num_lines, total_words = 0, 0
        combined_kg = KG()
        start_time = time.time()
        for article_id, curr_line in snippets:
            curr_line = curr_line.strip()
            for model in models:
                curr_kg = get_kg_for_line(model, curr_line, article_id)
                combined_kg.merge_with_kb(curr_kg)
            num_lines += 1
            total_words += len(curr_line.split(" "))

        # Record the metrics
        metrics["avg_time_to_process_sentence"] = str(round((time.time() - start_time)/num_lines, 3)) + " seconds"
        metrics["num_lines"] = num_lines
        metrics["avg_words_per_line"] = str(round(total_words/num_lines, 3)) + " words"
        
        # Return the result
        return {
            "result" : "sucess",
            "metrics" : metrics,
            "knowledge_graph" : combined_kg.get_json_representation()
        }
    except Exception as e:
        return {
            "result" : "error",
            "reason" : "Encountered error of " + str(e)
        }

def read_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--formation', type= str, required = True, help = "The formation we want to generate the kg for")
    parser.add_argument('--article_limit', type= int, default = 10, help = "The number of articles we should get snippets from")
    parser.add_argument('--fragment_limit', type= int, default = 5, help = "The number of fragments we should get from each article")
    parser.add_argument('--save_path', type = str, default = None, help = "The file path we want to store the results to")
    parser.add_argument('--model_types', nargs='+', default = ["rebel"], help = "The type of models we want to use")
    parser.add_argument('--model_paths', nargs = '+', default = ["Babelscape/rebel-large"], help = "The path to the model weights we want to use")
    return parser.parse_args()

def main():
    # Load the model
    args = read_args()
    models = get_models(args.model_types, args.model_paths)

    # Get the prediction
    result = get_kg_for_formation(models, args.formation, args.article_limit, args.fragment_limit)

    # Save the result
    if args.save_path is not None:
        with open(args.save_path, 'w+', encoding='utf-8') as writer:
            json.dump(result, writer, ensure_ascii=False, indent=4)
        print("Wrote result to", args.save_path)
    else:
        print("Got result of", result)

if __name__ == "__main__":
    main()