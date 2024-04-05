import math
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

def main():
    # Load the model
    model_path = "/ssd/dsarda/unsupervised-kg/rebel_finetuning/model/archive_tuned"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    # Upload the model
    model.push_to_hub("rebel_macrostrat_finetuned")
    tokenizer.push_to_hub("rebel_macrostrat_finetuned")

if __name__ == "__main__":
    main()