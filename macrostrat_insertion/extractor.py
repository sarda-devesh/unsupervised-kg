from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import sqlalchemy
from sqlalchemy.orm import declarative_base
from sqlalchemy import text
import json
import traceback
from datetime import datetime, timezone

from re_detail_adder import *

def load_engine(config_file):
    # Load the config file
    with open(config_file, 'r') as reader:
        config = json.load(reader)

    # Create the app 
    app = Flask(__name__)
    url_object = sqlalchemy.URL.create(
        "postgresql",
        username = config["username"],
        password = config["password"],  
        host = config["host"],
        port = config["port"],
        database = config["database"],
    )

    # Create the db
    engine = sqlalchemy.create_engine(url_object)
    return engine

# Connect to the database
MAX_TRIES = 5
CONFIG_FILE_PATH = "local_postgres.json"
engine = load_engine(CONFIG_FILE_PATH)

def print_schema():
    inspector = sqlalchemy.inspect(engine)
    schemas = inspector.get_schema_names()
    for schema in schemas:
        schema_name = str(schema)
        if schema_name == "macrostrat_kg":
            for table_name in inspector.get_table_names(schema=schema):
                for column in inspector.get_columns(table_name, schema=schema):
                    print("Table: %s, Column: %s" % (table_name, column))

def get_relationships(connection, run_id, source_id):
    query_to_run = f"""
    SELECT R.src, R.relationship_type, R.dst
    FROM macrostrat_kg.relationships AS R, macrostrat_kg.relationships_extracted AS RE
    WHERE RE.run_id = '{run_id}' AND R.run_id = '{run_id}'
    AND R.relationship_id = RE.relationship_id
    AND RE.source_id = {source_id}
    """

    all_relationships = []
    query_result = connection.execute(text(query_to_run))
    for row in query_result:
        relationship = row._mapping
        all_relationships.append({
            "src" : relationship["src"],
            "relationship_type" : relationship["relationship_type"],
            "dst" : relationship["dst"]
        })
        
    return all_relationships

def get_result_for_run(connection, run_id):
    # Get the unique sources run
    unique_sources_query = f"""
    SELECT weaviate_id, MIN(source_id)
    FROM macrostrat_kg.sources
    WHERE run_id = '{run_id}'  
    GROUP BY weaviate_id
    """
    query_result = connection.execute(text(unique_sources_query)).all()
    valid_source_ids = ",".join(["'" + str(data[1]) + "'" for data in query_result])

    sources_query = f"""
    SELECT *
    FROM macrostrat_kg.sources
    WHERE run_id = '{run_id}'
    AND source_id IN ({valid_source_ids})
    """

    all_results = []
    fields_to_keep = ["preprocessor_id", "paper_id", "hashed_text", "weaviate_id", "paragraph_text"]
    result = connection.execute(text(sources_query))
    for row in result:
        sources_row = row._mapping
        # Get the txt data
        txt_map = {}
        for field_name in fields_to_keep:
            txt_map[field_name] = sources_row[field_name]
        
        all_results.append({
            "text" : txt_map,
            "relationships" : get_relationships(connection, run_id, sources_row["source_id"])
        })
    
    return all_results

def main():
    save_dir = "db_extracted_runs"
    all_runs_query = """
    SELECT *
    FROM macrostrat_kg.metadata
    """

    with engine.connect() as connection:
        result = connection.execute(text(all_runs_query))
        for row in result:
            # Don't process example row
            metadata_row = row._mapping
            is_example = False
            for value in metadata_row.values():
                if "example" in value:
                    is_example = True
                    break
            
            if is_example:
                continue
            
            run_id = metadata_row["run_id"]
            json_to_save = {
                "run_id" : run_id,
                "extraction_pipeline_id" : metadata_row["extraction_pipeline_id"],
                "model_id" : metadata_row["model_id"],
                "results" : get_result_for_run(connection, run_id)
            }

            save_path = os.path.join(save_dir, run_id + ".json")
            with open(save_path, "w+") as writer:
                json.dump(json_to_save, writer, indent = 4)

if __name__ == "__main__":
    main()