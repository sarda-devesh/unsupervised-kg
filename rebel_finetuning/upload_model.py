import math
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import HfApi
import torch
import os

def main():
    # Load the model
    model_path = "/ssd/dsarda/unsupervised-kg/rebel_finetuning/model/archive_tuned"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    # Upload the model
    repo_name = "dsarda/rebel_macrostrat_finetuned"
    '''
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    print("Uploaded model and tokenizer")
    '''
    
    # Upload the data
    api = HfApi()
    for file_name in os.listdir(model_path):
        file_path = os.path.join(model_path, file_name)
        print("Uploading file", file_name)
        api.upload_file(path_or_fileobj = file_path, path_in_repo = file_name, repo_id = repo_name)

if __name__ == "__main__":
    main()
