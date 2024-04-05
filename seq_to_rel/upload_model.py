import torch
from huggingface_hub import HfApi
import os

def main():
    # Upload the weights
    model_path = "/ssd/dsarda/unsupervised-kg/seq_to_rel/strat_att_results/model.tar.gz"
    api = HfApi()
    api.upload_file(
        repo_id = "seq2rel_macrostrat_finetuned",
        repo_type = "model",
        path_or_fileobj = model_path,
        path_in_repo = os.path.basename(model_path)
    )
    print("Uploaded file", os.path.basename(model_path))

if __name__ == "__main__":
    main()