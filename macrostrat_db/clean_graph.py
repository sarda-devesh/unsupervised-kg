import pandas as pd

def clean_src(row):
    src_name = row["edge_src"]
    src_name = src_name.replace("\"", "")
    if ":" in src_name:
        src_name = src_name.split(":")[-1]

    row["edge_src"] = src_name.strip()
    return row

if __name__ == "__main__":
    df = pd.read_csv("macrostrat_graph.csv")
    print("Original length is", len(df.index))

    df = df.apply(lambda row : clean_src(row), axis = 1)
    df = df.drop_duplicates()

    print("Filtered df of length", len(df.index))
    df.to_csv("cleaned_macrostrat_graph.csv", index = False)