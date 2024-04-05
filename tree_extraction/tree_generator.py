from coref_resolution import *
from ner_extractor import *
from re_extractor import *

class TreeGenerator:

    def __init__(self, data_dir):
        self.ner_extractor = NERExtractor(data_dir)
        self.coref_resolver = CorefResolver()
        self.re_extractor = RE_Extractor(data_dir)
        self.start_prefixes = ["strat", "lith", "att"]
    
    def get_rock_level(self, rock):
        rock_type = rock.term_type
        for idx, prefix in enumerate(self.start_prefixes):
            if rock_type.startswith(prefix):
                return idx

        raise Exception("Invalid term type of " + rock_type)
    
    def perform_level_merge(self, lower_level, terms_by_level, sentence_words, sentence_spans):
        upper_level = lower_level - 1
        while upper_level > 0 and upper_level not in terms_by_level:
            upper_level -= 1
        
        lower_terms, lower_prefix = terms_by_level[lower_level], self.start_prefixes[lower_level]
        upper_terms, upper_prefix = terms_by_level[upper_level], self.start_prefixes[upper_level]

        for lower_term in lower_terms:
            # Determine term to merge with
            upper_merge_idx, upper_merge_probability = -1, -1.0
            for upper_idx, upper_term in enumerate(upper_terms):
                pair_probability = self.re_extractor.get_relationship_probability(sentence_words, lower_term, upper_term, lower_prefix, upper_prefix, sentence_spans)
                if pair_probability > upper_merge_probability:
                    upper_merge_idx = upper_idx
                    upper_merge_probability = pair_probability

            # Perform the merge
            if upper_merge_idx >= 0:
                upper_terms[upper_merge_idx].add_child(lower_term, child_probability = upper_merge_probability)
        
    def run_for_text(self, text):
        # Get the rock terms
        sentence_words, rock_terms, sentence_spans = self.ner_extractor.extract_terms(text)
        self.coref_resolver.record_cooref_occurences(sentence_words, rock_terms)

        # Get rocks by level
        rocks_by_level = {}
        for curr_rock in rock_terms:
            rock_level = self.get_rock_level(curr_rock)
            if rock_level not in rocks_by_level:
                rocks_by_level[rock_level] = []
            rocks_by_level[rock_level].append(curr_rock)        
        
        # Perform the merging
        for lower_level in range(len(self.start_prefixes) - 1, 0, -1):
            if lower_level not in rocks_by_level:
                continue

            self.perform_level_merge(lower_level, rocks_by_level, sentence_words, sentence_spans)
        
        # Return the results
        entity_name = "terms"
        entities_arr = []
        highest_exist = 0
        while highest_exist < len(self.start_prefixes) and highest_exist not in rocks_by_level:
            highest_exist += 1
        
        if highest_exist < len(self.start_prefixes):
            entity_name = self.start_prefixes[highest_exist] + "s"
            entities_arr = [entity.get_json() for entity in rocks_by_level[highest_exist]]

        return {
            "original_txt" : text,
            "words" : sentence_words,
            entity_name : entities_arr
        }

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help = "The directory containing the model")
    return parser.parse_args()

def main():
    args = read_args()
    tree_generator = TreeGenerator(args.data_dir)
    example_text = "the mount galen volcanics consists of basalt, andesite, dacite, and rhyolite lavas and dacite and rhyolite tuff and tuff-breccia. "
    example_text += "The Hayhook formation was named, mapped and discussed by lasky and webber (1949). the formation ranges up to at least 2500 feet in thickness."

    # Get the result for this txt
    result = tree_generator.run_for_text(example_text)
    with open("tree_example.json", "w+") as writer:
        json.dump(result, writer, indent = 4)

if __name__ == "__main__":
    main()