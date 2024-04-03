import os
import json
import pandas as pd
import numpy as np
import argparse
import spacy

class RockTerm:

    def __init__(self, term_type, term_tokens, span):
        self.term_type = term_type
        self.term_tokens = term_tokens
        self.occurences = [span]
        self.children = []
    
    def idx_in_range(self, idx):
        for start_idx, end_idx in self.occurences:
            if idx >= start_idx and idx < end_idx:
                return True
        
        return False

    def add_child(self, child):
        self.children.append(child)
    
    def get_json(self):
        return {
            "term_type" : self.term_type,
            "term_tokens" : self.term_tokens,
            "occurences" : self.occurences,
            "children" : [child.get_json() for child in self.children]
        }

class TrieNode:

    def __init__(self):
        self.label = None
        self.children = {}
    
    def contains_child(self, word):
        return word in self.children

    def add_child(self, word, child_node):
        self.children[word] = child_node
    
    def get_child(self, word):
        return self.children[word]

    def get_all_children(self):
        return list(self.children.keys())

class NERExtractor:

    tokenizer = spacy.load("en_core_web_lg") 

    def __init__(self, data_dir):
        terms_file = os.path.join(data_dir, "all_terms.csv")
        terms_df = pd.read_csv(terms_file)

        self.root_node = TrieNode()
        for idx, row in terms_df.iterrows():
            # Get the current node
            curr_term = row["term"]
            curr_term = curr_term.replace("-", " ").replace("(", "").replace(")", "")
            term_words = [word.lower().strip() for word in curr_term.split(" ")]

            # Update the trie
            curr_node = self.root_node
            for word in term_words:
                if len(word) == 0:
                    continue
                
                if not curr_node.contains_child(word):
                    curr_node.add_child(word, TrieNode())
                curr_node = curr_node.get_child(word)
            
            curr_node.label = row["term_type"]        
    
    def get_known_terms(self, tokenized_terms):
        search_start_idx = 0
        rock_terms = []
        while search_start_idx < len(tokenized_terms):
            # Perform search from this location
            curr_node = self.root_node
            curr_idx = search_start_idx
            match_idx, match_label = search_start_idx, None

            # Perform DFS on the trie
            while curr_idx < len(tokenized_terms) and curr_node.contains_child(tokenized_terms[curr_idx].lower()):
                curr_node = curr_node.get_child(tokenized_terms[curr_idx].lower())
                if curr_node.label is not None:
                    match_idx = curr_idx
                    match_label = curr_node.label
                
                curr_idx += 1
            
            # Perform the update
            if match_label is not None:
                start_idx, end_index = search_start_idx, match_idx + 1
                rock_terms.append(RockTerm(
                    term_type = match_label, 
                    term_tokens = tokenized_terms[start_idx : end_index], 
                    span = (start_idx, end_index)
                ))
                search_start_idx = end_index
            else:
                search_start_idx += 1
        
        return rock_terms

    def get_matching_term(self, term_idx, sentence_terms, known_terms):        
        # See if any of the existing terms already match
        for known_idx in range(len(known_terms)):
            if known_terms[known_idx].idx_in_range(term_idx):
                return known_terms[known_idx]
        
        # Determine the range of the next term
        term_type = sentence_terms[term_idx].pos_
        start_idx = term_idx
        while start_idx >= 0 and sentence_terms[start_idx].pos_ == term_type:
            start_idx -= 1
        start_idx += 1

        end_idx = term_idx
        while end_idx < len(sentence_terms) and sentence_terms[end_idx].pos_ == term_type:
            end_idx += 1

        # Record the new term
        known_terms.append(RockTerm(
            term_type = "lith_" + term_type, 
            term_tokens = [str(sentence_terms[token_idx].text) for token_idx in range(start_idx, end_idx)], 
            span = (start_idx, end_idx)
        ))
        return known_terms[-1]

    def get_unknown_terms(self, sentence_terms, known_terms):
        # Determine the idx already used
        used_terms = set()
        for curr_term in known_terms:
            for start_idx, end_index in curr_term.occurences:
                for idx in range(start_idx, end_index + 1):
                    used_terms.add(idx)
        
        # Determine if any of the unused index is a proper noun
        search_start_idx = 0
        while search_start_idx < len(sentence_terms):
            # Perform search from this index
            curr_idx = search_start_idx
            while curr_idx < len(sentence_terms) and curr_idx not in used_terms and sentence_terms[curr_idx].pos_ == "PROPN":
                curr_idx += 1
            
            if curr_idx > search_start_idx:
                known_terms.append(RockTerm(
                    term_type = "proper_noun", 
                    term_tokens = sentence_terms[search_start_idx : curr_idx], 
                    span = (search_start_idx, curr_idx)
                ))
                search_start_idx = curr_idx
            
            else:
                search_start_idx += 1
    
        # Check if any liths are actually lith attributes
        curr_idx = 0
        while curr_idx < len(known_terms):
            if not known_terms[curr_idx].term_type.startswith("lith"):
                curr_idx += 1
                continue
            
            # See if any terms in span are amod
            parent_idx = -1
            curr_term = known_terms[curr_idx]
            for start_idx, end_index in curr_term.occurences:
                for term_span_idx in range(start_idx, end_index):
                    curr_token = sentence_terms[term_span_idx]
                    if curr_token.dep_ == "amod":
                        parent_idx = curr_token.head.i
                        break
            
            if parent_idx < 0:
                curr_idx += 1
                continue
                
            # Update this term to be a lith attribute
            parent_term = self.get_matching_term(parent_idx, sentence_terms, known_terms)
            curr_term = known_terms.pop(curr_idx)
            curr_term.term_type = "att_amod"
            parent_term.add_child(curr_term)

    def extract_terms(self, sentence):
        sentence = sentence.replace("-", " ").replace("(", "").replace(")", "")
        sentence_terms = [token for token in NERExtractor.tokenizer(sentence)]

        sentence_words = [str(token).strip() for token in sentence_terms]
        rock_terms = self.get_known_terms(sentence_words)
        self.get_unknown_terms(sentence_terms, rock_terms)
        return sentence_words, rock_terms

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help = "The directory containing the model")
    return parser.parse_args()

def main(args):
    # Perform initialization
    ner_extractor = NERExtractor(args.data_dir)
    example_txt = "the mount galen volcanics consists of basalt, andesite, dacite, and rhyolite lavas and dacite and rhyolite tuff and tuff-breccia"

    # Run on terms
    sentence_words, rock_terms = ner_extractor.extract_terms(example_txt)
    print(sentence_words)
    for rock in rock_terms:
        print(rock.get_json())

if __name__ == "__main__":
    main(read_args())