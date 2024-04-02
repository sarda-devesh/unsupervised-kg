import argparse
import logging
import os
import random
import spacy
import time
import json
import pandas as pd

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

from torch.nn import CrossEntropyLoss

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

CLS = "[CLS]"
SEP = "[SEP]"

class RE_Extractor:

    def __init__(self, data_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_tokenizer = BertTokenizer.from_pretrained(data_dir, do_lower_case = False)
        self.model = BertForSequenceClassification.from_pretrained(data_dir, num_labels = 10)
        self.model = self.model.to(self.device)
        with open(os.path.join(data_dir, "label_mapping.json"), "r") as reader:
            self.id_to_label = json.load(reader)["id2label"]
        self.max_seq_length = 512
    
    def extract_features(self, inputs, special_tokens = {}):

        words = inputs["words"]
        def get_special_token(w):
            if w not in special_tokens:
                special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
            return special_tokens[w]

        # Get the tokens
        num_tokens = 0
        tokens = [CLS]
        SUBJECT_START = get_special_token("SUBJ_START")
        SUBJECT_END = get_special_token("SUBJ_END")
        OBJECT_START = get_special_token("OBJ_START")
        OBJECT_END = get_special_token("OBJ_END")

        # Build the initial tokens
        start_idx = min(inputs["span1"][0], inputs["span2"][0])
        end_idx = max(inputs["span1"][1], inputs["span2"][1])
        for i in range(start_idx, end_idx):
            if i == inputs["span1"][0]:
                tokens.append(SUBJECT_START)
            if i == inputs["span2"][0]:
                tokens.append(OBJECT_START)
            for sub_token in self.bert_tokenizer.tokenize(words[i]):
                tokens.append(sub_token)
            if i == inputs["span1"][1]:
                tokens.append(SUBJECT_END)
            if i == inputs["span2"][1]:
                tokens.append(OBJECT_END)

        while (start_idx >= 0 or end_idx < len(words)) and len(tokens) < self.max_seq_length:
            # Include the left word
            if start_idx >= 0:
                left_tokens = [sub_token for sub_token in self.bert_tokenizer.tokenize(words[start_idx])]
                tokens = left_tokens + tokens
                start_idx -= 1
            
            # Include the right word
            if end_idx < len(words):
                right_tokens = [sub_token for sub_token in self.bert_tokenizer.tokenize(words[end_idx])]
                tokens = tokens + right_tokens
                end_idx += 1
        
        tokens.append(SEP)

        # Extract the features
        num_tokens += len(tokens)
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
        
        segment_ids = [0] * len(tokens)
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        # Return it as tensor
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        input_mask = torch.tensor([input_mask], dtype=torch.long).to(self.device)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long).to(self.device)

        return input_ids, input_mask, segment_ids

    def get_prediction(self, inputs):
        input_ids, input_mask, segment_ids = self.extract_features(inputs)
        with torch.no_grad():
            logits = self.model(input_ids, segment_ids, input_mask, labels=None)

        probabilities = torch.nn.functional.softmax(logits, dim = -1).detach().cpu().numpy()[0]
        pred_id = np.argmax(probabilities)
        return self.id_to_label[str(pred_id)], probabilities[pred_id]

def main(args):
    # Load the model
    model = RE_Extractor(args.model_dir)

    # Format the input
    nlp = spacy.load("en_core_web_lg") 
    example_text = "the mount galen volcanics consists of basalt, andesite, dacite, and rhyolite lavas and dacite and rhyolite tuff"
    input = {
        "words" : [str(token) for token in nlp(example_text)],
        "span1" : (6, 7),
        "span2" : (9, 10)
    }
    prediction, probability = model.get_prediction(input)
    print("Got prediction", prediction, "has probability of", probability)

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help = "The directory containing the model")
    return parser.parse_args()

if __name__ == "__main__":
    main(read_args())