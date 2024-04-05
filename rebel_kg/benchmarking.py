import torch
import argparse
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score

from model_wrapper import *

def get_model(model_type, model_path):
    print("Loading model", model_type, "from weights", model_path)
    if model_type == "rebel":
        return RebelWrapper(model_path)
    elif model_type == "seq2rel":
        return Seq2RelWrapper(model_path)
    else:
        raise Exception("Invalid model type of " + model_type)

def read_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_type', type = str, required = True, help = "The type of model we want to use")
    parser.add_argument('--model_path', type = str, required = True, help = "The path to the finetuned model weights we want to use")
    parser.add_argument('--dataset_path', type = str, required = True, help = "The path to the dataset we want to evaluate the model on")
    parser.add_argument('--num_examples', type = int, default = 25, help = "The number of examples to run the benchmark on")
    return parser.parse_args()

possible_relations = [ "strat_name_to_lith",  "lith_to_lith_group",  "lith_to_lith_type",  "att_grains",  "att_lithology", "att_color", 
"att_sed_structure", "att_bedform",  "att_structure"]

def run_benchmarking(args):
    # Load the model and dataset
    model = get_model(args.model_type, args.model_path)
    print("Loading dataset from path", args.dataset_path)
    benchmark_df = pd.read_csv(args.dataset_path, sep = '\t', header = None, names = ["sentence", "relationship"]).sample(frac=1).head(args.num_examples)
    print("Loaded a total of", len(benchmark_df.index), "examples")

    true_labels, predicted_labels = [], []
    for idx, row in benchmark_df.iterrows():
        # Determine the expected relationship
        relationship_parts = row["relationship"].strip().split(" ")
        relationship = relationship_parts[-1].strip()[1 : -1]
        expected_value = possible_relations.index(relationship) + 1
        print("Expected relationship", relationship, "with value", expected_value)

        # Get the prediction
        sentence = row["sentence"].strip()
        predictions = model.get_relations_in_line(sentence)
        print("Sentence", sentence, "has prediction of", predictions)
        prediction_count = {}
        for curr_prediction in predictions:
            prediction_type = curr_prediction["type"]
            # Ensure we only process relationships we care about
            if prediction_type in possible_relations:
                if prediction_type not in prediction_count:
                    prediction_count[prediction_type] = 0
                prediction_count[prediction_type] += 1
        
        # Compute the prediction
        predicted_value = 0
        if len(prediction_count) > 0:
            sorted_pairs = sorted(prediction_count.items(), key=lambda pair: pair[1])
            predicted_relationship = sorted_pairs[0][0]
            predicted_value = possible_relations.index(predicted_relationship) + 1
        print("Got prediction of", predicted_value)
        
        # Record these values
        true_labels.append(expected_value)
        predicted_labels.append(predicted_value)
    
    print("Precision of", 100.0 * precision_score(true_labels, predicted_labels, average='macro'))
    print("Recall of", 100.0 * recall_score(true_labels, predicted_labels, average='macro'))
    print("F1 score of", 100.0 * f1_score(true_labels, predicted_labels, average='macro'))

if __name__ == "__main__":
    args = read_args()
    run_benchmarking(args)