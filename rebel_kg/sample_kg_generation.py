import json
import os

from formation_kg_generator import *

def run_for_sample_file(save_dir, paragraphs_dir):
    # Load the formaitons to process
    with open("formation_to_process.txt", "r") as reader:
        formation_names = reader.readlines()
    
    models_to_use = get_models(["seq2rel", "rebel"], ["/ssd/dsarda/unsupervised-kg/seq_to_rel/output/model.tar.gz", "/ssd/dsarda/unsupervised-kg/rebel_finetuning/model/archive_tuned"]) 
    os.makedirs(save_dir, exist_ok = True)

    for name in formation_names:
        # Determine if the file already exists
        formation_name = name.strip()
        file_name = formation_name.replace(" ", "_") + ".json"
        save_path = os.path.join(save_dir, file_name)
        if os.path.exists(save_path):
            continue

        # Load the paragraph for this formation name
        paragraph_path = os.path.join(paragraphs_dir, file_name)
        if not os.path.exists(paragraph_path):
            continue

        matching_paragraphs = None
        try:
            with open(paragraph_path, 'r') as reader:
                matching_paragraphs = json.load(reader)
        except Exception as e:
            print("Failed to parse file", paragraph_path, "as json")
            continue

        # Get the Kg for this formation
        result_json = get_kg_for_paragraphs(models_to_use, matching_paragraphs)
        if "knowledge_graph" not in result_json:
            print("Encountered error of", result_json["reason"])
            continue
        
        print("Saving knowledge graph to", save_path)
        with open(save_path, 'w+', encoding='utf-8') as writer:
            json.dump(result_json["knowledge_graph"], writer, ensure_ascii=False, indent=4)

def get_matching_paragraphs(save_dir):
    with open("formation_to_process.txt", "r") as reader:
        formation_names = reader.readlines()
    
    os.makedirs(save_dir, exist_ok = True)
    overall_count, num_matchings, total_paragraphs = 0, 0, 0
    for name in formation_names:
        formation_name = name.strip()
        save_path = os.path.join(save_dir, formation_name.replace(" ", "_") + ".json")
        if os.path.exists(save_path):
            continue
        
        overall_count += 1
        all_paragraphs = get_larger_text_for_formation(formation_name, topic = POSSIBLE_TOPICS[0]) + get_larger_text_for_formation(formation_name, topic = POSSIBLE_TOPICS[1])
        if len(all_paragraphs) == 0:
            print("Failed to get result for", formation_name)
            continue
        
        num_matchings += 1
        total_paragraphs += len(all_paragraphs)

        data_to_save = {"all_paragraphs" : all_paragraphs}
        print("Saving result to", save_path)
        with open(save_path, 'w+', encoding='utf-8') as writer:
            json.dump(data_to_save, writer, ensure_ascii=False, indent=4)
        
        time.sleep(1)

    print("Total matching", (100.0 * num_matchings)/overall_count)
    print("Average paragraph", total_paragraphs/num_matchings)

def load_json_file(json_file):
    with open(json_file, 'r') as reader:
        data = json.load(reader)
    return data

def extract_for_strat_to_lith(relationship, strat_map, lith_map):
    src, dst = relationship["head"].lower(), relationship["tail"].lower()
    if src in strat_map:
        if dst in lith_map:
            return [src, strat_map[src], "strat_name_to_lith", dst, lith_map[dst]]
    elif dst in strat_map:
        if src in lith_map:
            return [dst, strat_map[dst], "strat_name_to_lith", src, lith_map[src]]
    
    return []

def extract_for_lith(relationship, lith_map):
    src, dst = relationship["head"].lower(), relationship["tail"].lower()
    if src in lith_map:
        return [src, lith_map[src], relationship["type"], dst, lith_map[src]]
    elif dst in lith_map:
        return [dst, lith_map[dst], relationship["type"], src, lith_map[dst]]
    
    return []

def extract_for_lith_att(relationship, lith_id_map, lith_att_map):
    # Get the map for this attribute
    att_type = relationship["type"].replace("att_", "").replace("_", " ").lower()
    att_map = lith_att_map[att_type]

    src, dst = relationship["head"].lower(), relationship["tail"].lower()
    if src in lith_id_map:
        if dst in att_map:
            return [src, lith_id_map[src], relationship["type"], dst, att_map[dst]]
    elif dst in lith_id_map:
        if src in att_map:
            return [dst, lith_id_map[dst], relationship["type"], src, att_map[src]]
    
    return []

def process_json_file(json_file_path, relationship_rows, sources_rows, ids_maps):
    with open(json_file_path, 'r') as reader:
        data = json.load(reader)
    
    # Get the strat id
    strat_map = ids_maps["strat_name_map"]
    strat_name = os.path.basename(json_file_path).replace(".json", "").strip()
    strat_name = strat_name.replace("_", " ").lower()
    if strat_name not in strat_map:
        raise Exception("Couldn't find strat " + str(strat_name) + " in start map")
    
    strat_row_prefix = [strat_map[strat_name], strat_name]
    for relationship in data:
        # Get the result row depending on type of relationship
        relationship_type = relationship["type"]
        result_row = None
        if relationship_type.startswith("strat_name"):
            result_row = extract_for_strat_to_lith(relationship, ids_maps["strat_name_map"], ids_maps["lith_id_map"])
        elif relationship_type.startswith("lith"):
            result_row = extract_for_lith(relationship, ids_maps["lith_id_map"])
        elif relationship_type.startswith("att"):
            result_row = extract_for_lith_att(relationship, ids_maps["lith_id_map"], ids_maps["lith_att_map"])
        else:
            continue
        
        # Ensure we have results for this row
        if result_row is None or len(result_row) == 0:
            continue
        
        # Add this to relationship_row
        relationship_id = len(relationship_rows)
        relationship_rows.append([relationship_id] + strat_row_prefix + result_row)
    
        # Add in the sources
        for source in relationship["sources"]:
            article_id = source["article_id"]
            for snippet in source["snippets_used"]:
                sources_rows.append([relationship_id, article_id, snippet.strip()])

def create_sample_df():
    src_dir = "formation_sample_kgs"
    dst_dir = "formation_sample_dfs"
    os.makedirs(dst_dir, exist_ok = True)

    # Load the ids maps
    id_map_dirs = "id_maps"
    lith_id_map = load_json_file(os.path.join(id_map_dirs, "lith_id_map.json"))
    lith_att_map = load_json_file(os.path.join(id_map_dirs, "lith_att_id_map.json"))
    strat_name_map = load_json_file(os.path.join(id_map_dirs, "strat_names_map.json"))
    ids_maps = {
        "lith_id_map" : lith_id_map,
        "lith_att_map" : lith_att_map,
        "strat_name_map" : strat_name_map
    }

    # Create the dataframes for 
    relationship_rows, sources_rows = [], []
    count = 0
    for kg_file in os.listdir(src_dir):
        if "json" not in kg_file or kg_file[0] == '.':
            continue

        # Process the current json file        
        process_json_file(os.path.join(src_dir, kg_file), relationship_rows, sources_rows, ids_maps)
        count += 1
    
    print("Processed a total of", count, "json files")

    # Save the relationships file
    relationship_cols = ["relationship_id", "kg_strat_id", "kg_strat_name", "src", "src_id", "type", "dst", "dst_id"]
    relationship_df = pd.DataFrame(relationship_rows, columns = relationship_cols)
    relationship_df = relationship_df.drop_duplicates()
    print("Got a total of", len(relationship_df.index), "relationships")
    relationship_save_path = os.path.join(dst_dir, "relationships.csv")
    relationship_df.to_csv(relationship_save_path, index = False)

    # Save the sources file
    sources_cols = ["relationship_id", "article_id", "snippets_txt"]
    sources_df = pd.DataFrame(sources_rows, columns = sources_cols)
    sources_df = sources_df.drop_duplicates()
    print("Got a total of", len(relationship_df.index), "sources")
    sources_save_path = os.path.join(dst_dir, "sources.csv")
    sources_df.to_csv(sources_save_path, index = False)

if __name__ == "__main__":
    # run_for_sample_file("formation_sample_kgs", "formation_sample_paragraphs")
    create_sample_df()