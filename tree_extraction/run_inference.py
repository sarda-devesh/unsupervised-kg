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

class InputExample(object):
    """A single training/test example for span pair classification."""

    def __init__(self, guid, sentence, span1, span2, ner1, ner2):
        self.guid = guid
        self.sentence = sentence
        self.span1 = span1
        self.span2 = span2
        self.ner1 = ner1
        self.ner2 = ner2
    
    def get_json(self):
        return {
            "guid" : self.guid,
            "sentence" : self.sentence,
            "span1" : self.span1,
            "span2" : self.span2,
            "ner1" : self.ner1,
            "ner2" : self.ner2,
        }


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = torch.tensor([input_ids], dtype=torch.long)
        self.input_mask = torch.tensor([input_mask], dtype=torch.long)
        self.segment_ids = torch.tensor([segment_ids], dtype=torch.long)
    
    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.input_mask = self.input_mask.to(device)
        self.segment_ids = self.segment_ids.to(device)
    
    def make_prediction_using_model(self, model):
        with torch.no_grad():
            logits = model(self.input_ids, self.segment_ids, self.input_mask, labels=None)

        probabilities = torch.nn.functional.softmax(logits, dim = -1).detach().cpu().numpy()[0]
        pred_id = np.argmax(probabilities)
        return pred_id, probabilities[pred_id]

def convert_examples_to_features(example, max_seq_length, tokenizer, special_tokens = {}, mode='text'):
    """Loads a data file into a list of `InputBatch`s."""

    def get_special_token(w):
        if w not in special_tokens:
            special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
        return special_tokens[w]

    num_tokens = 0
    num_fit_examples = 0
    tokens = [CLS]
    SUBJECT_START = get_special_token("SUBJ_START")
    SUBJECT_END = get_special_token("SUBJ_END")
    OBJECT_START = get_special_token("OBJ_START")
    OBJECT_END = get_special_token("OBJ_END")
    SUBJECT_NER = get_special_token("SUBJ=%s" % example.ner1)
    OBJECT_NER = get_special_token("OBJ=%s" % example.ner2)

    if mode.startswith("text"):
        for i, token in enumerate(example.sentence):
            if i == example.span1[0]:
                tokens.append(SUBJECT_START)
            if i == example.span2[0]:
                tokens.append(OBJECT_START)
            for sub_token in tokenizer.tokenize(token):
                tokens.append(sub_token)
            if i == example.span1[1]:
                tokens.append(SUBJECT_END)
            if i == example.span2[1]:
                tokens.append(OBJECT_END)
        if mode == "text_ner":
            tokens = tokens + [SEP, SUBJECT_NER, SEP, OBJECT_NER, SEP]
        else:
            tokens.append(SEP)
    else:
        subj_tokens = []
        obj_tokens = []
        for i, token in enumerate(example.sentence):
            if i == example.span1[0]:
                tokens.append(SUBJECT_NER)
            if i == example.span2[0]:
                tokens.append(OBJECT_NER)
            if (i >= example.span1[0]) and (i <= example.span1[1]):
                for sub_token in tokenizer.tokenize(token):
                    subj_tokens.append(sub_token)
            elif (i >= example.span2[0]) and (i <= example.span2[1]):
                for sub_token in tokenizer.tokenize(token):
                    obj_tokens.append(sub_token)
            else:
                for sub_token in tokenizer.tokenize(token):
                    tokens.append(sub_token)

        if mode == "ner_text":
            tokens.append(SEP)
            for sub_token in subj_tokens:
                tokens.append(sub_token)
            tokens.append(SEP)
            for sub_token in obj_tokens:
                tokens.append(sub_token)
                
        tokens.append(SEP)
    num_tokens += len(tokens)

    if len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length]
    else:
        num_fit_examples += 1

    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    
    return InputFeatures(input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids)

class RE_Extractor:

    spacy_tokenizer = spacy.load("en_core_web_lg") 

    def __init__(self, data_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_tokenizer = BertTokenizer.from_pretrained(data_dir, do_lower_case = False)
        self.model = BertForSequenceClassification.from_pretrained(data_dir, num_labels = 10)
        self.model = self.model.to(self.device)
        with open(os.path.join(data_dir, "label_mapping.json"), "r") as reader:
            self.id_to_label = json.load(reader)["id2label"]
        self.max_seq_length = 128
    
    def extract_features(input):
        pass


def main(args):
    # Load the model
    print("Loaded model")

    # Format the input
    search_txt = "the rhytholite in the Bonetterre Formation"
    example = InputExample(guid = -1, sentence = tokens, span1 = (1, 2), span2 = (4, 6), ner1 = "lith", ner2 = "strat_name")
    print("Loaded feature")

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help = "The directory containing the model")
    return parser.parse_args()

if __name__ == "__main__":
    main(read_args())