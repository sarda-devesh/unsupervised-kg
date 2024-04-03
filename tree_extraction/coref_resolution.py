from fastcoref import FCoref
import torch
import spacy

class CorefResolver:

    def __init__(self):
        self.model = FCoref(device = "cuda" if torch.cuda.is_available() else "cpu")
    
    def run_coref(self, sentence_words):
        prediction = self.model.predict(texts = [sentence_words], is_split_into_words=True)[0]
        cluster = prediction.get_clusters(as_strings=False)[0]
        print(cluster)

def main():
    txt = "he artillery formation was named, mapped and discussed by lasky and webber (1949). the formation ranges up to at least 2500 feet in thickness. "
    nlp = spacy.load("en_core_web_lg") 
    sentence_words = [str(token.text) for token in nlp(txt)]

    resolver = CorefResolver()
    resolver.run_coref(sentence_words)

if __name__ == "__main__":
    main()