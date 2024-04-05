import pandas as pd

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_file', type= str, required = True, help = "The path to the graph file")
    return parser.parse_args()

def unique_ent_and_rel(graph_file):
    df = pd.read_csv(graph_file)
    r_types = list(df["edge_name"].unique())

    unique_entities = set()
    r_tokens = []
    for curr_r in r_types:
        if "_to_" in curr_r:
            src_type, dst_type = curr_r.split("_to_")
            unique_entities.add(src_type)
            unique_entities.add(dst_type)
        else:
            unique_entities.add("lith")
            unique_entities.add("lith_att")
        
        r_tokens.append("@" + str(curr_r) + "@")
    
    print("Relationship tokens of", r_tokens)

    unique_arr = list(unique_entities)
    ent_tokens = []
    for unique_val in unique_arr:
        ent_tokens.append("@" + str(unique_val) + "@")

    print("Entity tokens of", ent_tokens)

if __name__ == "__main__":
    args = read_args()
    unique_ent_and_rel(args.graph_file)