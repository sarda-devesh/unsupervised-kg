import pandas as pd
import os
import requests

def read_df(confidence_level = 0.5):
    dfs_to_merge = []

    output_dir = "ner_outputs"
    for file_name in os.listdir(output_dir):
        if "ner_results" not in file_name or "csv" not in file_name or file_name[0] == '.':
            continue
        
        # Read the df
        file_path = os.path.join(output_dir, file_name)
        df = pd.read_csv(file_path)
        df = df[df["ner_confidence"] >= confidence_level]
        dfs_to_merge.append(df)
    
    return pd.concat(dfs_to_merge)

def present_in_xdd(entity_name):
    try:
        formatted_name = entity_name.replace(" ", "%20")
        request_url = f"https://xdd.wisc.edu/api/snippets?term={formatted_name}&inclusive=true&clean=true"
        response = requests.get(request_url).json()
        is_present = False
        if 'success' in response and 'data' in response['success']: 
            data = response['success']['data']
            is_present = len(data) > 0
        
        return is_present
    except Exception as e:
        print("Failed to get xdd result for", entity_name, "due to error", e)
        return False

def present_in_macrostrat(entity_name):
    try:
        formatted_name = entity_name.split(" ")[0].strip().replace(" ", "%20")
        request_url = f"https://v2.macrostrat.org/api/v2/defs/strat_names?strat_name_like={formatted_name}"
        response = requests.get(request_url).json()
        is_present = False
        if 'success' in response and 'data' in response['success']: 
            data = response['success']['data']
            is_present = len(data) > 0
        
        return is_present
    except Exception as e:
        print("Failed to get xdd result for", entity_name, "due to error", e)
        return False

def main():
    # Get the df
    entities_df = read_df()
    entities_df = entities_df[entities_df["ner_label"].isin(["CHRONOSTRAT", "LEXICON"])]

    # Get the existing results
    existing_df = None
    existing_entities = set()
    save_path = os.path.join("ner_outputs", "entity_is_present.csv")
    if os.path.exists(save_path):
        existing_df = pd.read_csv(save_path)
        existing_entities = set(existing_df["entity_name"].values)
    
    # Process all entities
    all_entities = set(entities_df["text"].values)
    print("Got a total of", len(entities_df.index), "entities")
    print("Found", len(all_entities), "unique entities with", len(existing_entities), "existing results")
    results_row = []

    for entity_name in all_entities:
        # See if we have already processed this
        entity_name = entity_name.replace("\n", " ").replace("+", " ").strip()
        if "Formation" in entity_name:
            parts = entity_name.split(" ")
            if parts[0] == "Formation":
                entity_name = " ".join(parts[1 : ] + ["Formation"])

        if entity_name in existing_entities:
            continue
        
        entity_in_xdd = present_in_xdd(entity_name)
        entity_in_macrostrat = present_in_macrostrat(entity_name)
        results_row.append([entity_name, entity_in_xdd, entity_in_macrostrat])

    # Save the result
    save_df = pd.DataFrame(results_row, columns = ["entity_name", "in_xdd", "in_macrostrat"])
    if existing_df is not None:
        save_df = pd.concat([existing_df, save_df])
    
    if existing_df is None or len(save_df.index) > len(existing_df.index):
        save_df.to_csv(save_path, index = False)
    
    # Get the formations we want to process
    filtered_df = save_df[(save_df["in_macrostrat"] == False) & (save_df["in_xdd"] == True)]
    entities_to_process = set(filtered_df["entity_name"].values)
    txt_save_path = "narrowed_entities.txt"
    with open(txt_save_path, 'w+') as writer:
        for entity_name in entities_to_process:
            if len(entity_name.split(" ")) > 1:
                writer.write(entity_name + "\n")

if __name__ == "__main__":
    main()