import requests
import argparse
import pandas as pd
import json
import os
import time

def read_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--paragraphs_file', type= str, required = True, help = "The path to the parquet files containing the entities")
    return parser.parse_args()

request_url = 'http://localhost:9000/?properties={"annotations":"named entities","outputFormat":"json"}'
def get_entities_for_paragraph(para_id, paragraph):
    # Send the request
    response = requests.post(request_url, data = {'data': paragraph})
    if response.status_code != 200:
        print("Failed to get result for para_id", para_id)
        return []

    # Extract the entities
    data = response.json()
    if "sentences" not in data:
        print("Failed to get sentences for para_id", para_id)
    
    sentences = data["sentences"]
    extracted_entities = []
    for sentence in sentences:
        if "entitymentions" not in sentence:
            continue
        
        mentions = sentence["entitymentions"]
        for mention in mentions:
            # Extract the ner values
            text = mention["text"]
            ner_label = mention["ner"]
            conf_val = -1.0
            if "nerConfidences" in mention and ner_label in mention["nerConfidences"]:
                conf_val = float(mention["nerConfidences"][ner_label])  

            # Save the result
            if conf_val > 0.0:
                extracted_entities.append([para_id, text, ner_label, conf_val])

    return extracted_entities

def save_results(save_directory, write_idx, paragraph_rows, ner_rows):
    # Save the results for the paragraphs
    paragraphs_df = pd.DataFrame(paragraph_rows, columns = ["paragraph_id", "paragraph"])
    paragraphs_df.to_csv(os.path.join(save_directory, "paragraphs_" + str(write_idx) + ".csv"), index = False)
    
    # Save the ner results
    ner_df = pd.DataFrame(ner_rows, columns = ["paragraph_id", "text", "ner_label", "ner_confidence"])
    ner_df.to_csv(os.path.join(save_directory, "ner_results_" + str(write_idx) + ".csv"), index = False)

    print("Finished writing file", write_idx)


def get_ner_for_paragraphs(pargraph_df_file, save_directory, num_logs = 100):
    pargraph_df = pd.read_parquet(pargraph_df_file)
    paragraph_rows = []
    ner_rows = []
    write_idx = 0
    os.makedirs(save_directory, exist_ok = True)

    # Get the ner for the paragraphs 
    log_rate = int(len(pargraph_df.index)/num_logs)
    print(len(pargraph_df.index), "paragraphs with log rate of", log_rate)

    for para_id, row in pargraph_df.iterrows():
        paragraph_txt = row["paragraph"]
        ner_rows.extend(get_entities_for_paragraph(para_id, paragraph_txt))
        paragraph_rows.append([para_id, paragraph_txt.replace("\n", " ")])

        # Save the results every so often
        if para_id > 0 and para_id % log_rate == 0:
            save_results(save_directory, write_idx, paragraph_rows, ner_rows)
            paragraph_rows = []
            ner_rows = []
            write_idx += 1
    
    # Ensure that the final results are saved
    save_results(save_directory, write_idx, paragraph_rows, ner_rows)

if __name__ == "__main__":
    args = read_args()
    get_ner_for_paragraphs(args.paragraphs_file, "ner_outputs")