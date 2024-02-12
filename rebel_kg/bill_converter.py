import os
import pandas as pd
import json
from generate_kg_per_formation import *

conversion = {
    'lithology has type of' : ("lith_to_lith_type", "lithology", "lithology type"),
    'has structure of' : ("att_structure", "lithology", "lith attribute structure"),
    'stratigraphic unit has lithology' : ("strat_name_to_lith", "strat_name", "lith"),
    'has color of' : ("att_color", "lithology", "lith attribute color"), 
    'has bedform of' : ("att_bedform", "lithology", "lith attribute bedform"),
    'has lithology of' : ("att_lithology", "lithology", "lith attribute lithology"), 
    'has sedimentary structure' : ("att_sed_structure", "lithology", "lith attribute sedementary"), 
    'has grains of' : ("att_grains", "lithology", "lith attribute grain")
}
def create_triplet(row):
    head, human_relationship, tail, paper_id = row["head"], row["relationship"], row["tail"], row["paper_id"]
    relationship_type, src_type, dst_type = conversion[human_relationship]
    return {
        "head" : head,
        "type" : relationship_type,
        "model_used" : "llm_based",
        "tail" : tail,
        "human_type" : human_relationship,
        "src_type" : src_type,
        "dst_type" : dst_type,
        "sources" : [
            {
                "article_id" : paper_id,
                "txt_used" : [
                    ""
                ]
            }
        ]
    }

def perform_conversion():
    dataset_file = "cleaned_output.csv"
    df = pd.read_csv(dataset_file)
    
    save_dir = "bill_results"
    for formation_name, formation_rows in df.groupby(by = "formation_name"):
        kg_relationships = []
        for idx, row in formation_rows.iterrows():
            kg_relationships.append(create_triplet(row))

        save_path = os.path.join(save_dir, formation_name.replace(" ", "_") + ".json")
        json_to_save = {
            "knowledge_graph" : kg_relationships
        }
        with open(save_path, 'w+', encoding='utf-8') as writer:
            json.dump(json_to_save, writer, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    perform_conversion()