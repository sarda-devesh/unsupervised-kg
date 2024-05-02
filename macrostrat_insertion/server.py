from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import sqlalchemy
from sqlalchemy.orm import declarative_base
import json
import traceback
from datetime import datetime, timezone

from re_detail_adder import *

DEFAULT_SCHEMA = "macrostrat_kg"
def load_flask_app(config_file):
    # Load the config file
    with open(config_file, 'r') as reader:
        config = json.load(reader)

    # Create the app 
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = sqlalchemy.URL.create(
        "postgresql",
        username = config["username"],
        password = config["password"],  
        host = config["host"],
        port = config["port"],
        database = config["database"],
    )

    # Create the db
    Base = declarative_base(metadata = sqlalchemy.MetaData(schema = DEFAULT_SCHEMA))
    db = SQLAlchemy(model_class=Base)
    db.init_app(app)
    with app.app_context():
        db.reflect()

    return app, db

# Connect to the database
MAX_TRIES = 5
CONFIG_FILE_PATH = "local_postgres.json"
app, db = load_flask_app(CONFIG_FILE_PATH)

def get_new_record_id(table):
    num_tries = 0
    while num_tries < MAX_TRIES:
        try:
            return db.session.scalar(sqlalchemy.select(sqlalchemy.func.count()).select_from(table)) + 1
        except Exception as e:
            print("Got exception in get of", traceback.format_exc())
        num_tries += 1
    
    raise Exception("Unable to generate new record id")

def insert_run_metadata(request_data):
    # Ensure we have a run id
    if "run_id" not in request_data:
        return "Missing key run_id in data", None
    
    # Check if the run_id already exists
    run_id = request_data["run_id"]
    metadata_table = db.metadata.tables['macrostrat_kg.metadata']
    try:
        query_to_run = sqlalchemy.select(sqlalchemy.func.count()).select_from(metadata_table).filter_by(run_id = run_id)
        existing_count = db.session.scalar(query_to_run)
        if existing_count > 0:
            return "", run_id
    except Exception as e:
        return traceback.format_exc(), None

    # Check if other keys exists
    required_keys = [ "extraction_pipeline_id", "model_id"]  
    record_to_add = {"run_id" : run_id}
    for curr_key in required_keys:
        if curr_key not in request_data:
            return "Missing key " + curr_key + " in data", None
        else:
            record_to_add[curr_key] = request_data[curr_key]

    # Try to add the record
    record_to_add["run_timestamp"] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    try:
        insert_stmt = sqlalchemy.insert(metadata_table).values(**record_to_add)
        db.session.execute(insert_stmt)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return traceback.format_exc(), None

    return "", run_id

relationship_processor = REProcessor("id_maps")
def insert_result(result, run_id):
    # Make sure the result has key
    required_result_key = ["text", "relationships"]
    for key in required_result_key:
        if key not in result:
            return False, "Missing key " + key + " in result"
    
    # Make sure text has required key
    text_data = result["text"]
    required_text_keys = ["preprocessor_id", "paper_id", "hashed_text", "weaviate_id", "paragraph_text"]
    text_record = {"run_id" : run_id}
    for required_key in required_text_keys:
        if required_key not in text_data:
            return False, "Missing key " + required_key + " not in result's text"
        else:
            text_record[required_key] = text_data[required_key]
    
    source_id = -1
    try:
        # Determine the new source id
        sources_table = db.metadata.tables['macrostrat_kg.sources']
        source_id = get_new_record_id(sources_table)
        text_record["source_id"] = source_id
        
        # Insert the source
        insert_stmt = sqlalchemy.insert(sources_table).values(**text_record)
        db.session.execute(insert_stmt)
        db.session.commit()
    except:
        db.session.rollback()
        return False, traceback.format_exc()
    
    # Insert the resulting relationship
    failed_relationship = 0
    for current_relationship in result["relationships"]:
        updated_relationship = relationship_processor.get_relationship_json(current_relationship)
        if updated_relationship is None:
            continue
        
        try:
            # Get the new relationship id
            relationship_table = db.metadata.tables['macrostrat_kg.relationships']
            relationship_id = get_new_record_id(relationship_table)
            updated_relationship["relationship_id"] = relationship_id

            # Insert new relationship
            updated_relationship["run_id"] = run_id
            relationship_insert_stmt = sqlalchemy.insert(relationship_table).values(**updated_relationship)
            db.session.execute(relationship_insert_stmt)
            db.session.commit()

            # Record the relationship <-> source relationship
            relationship_extracted = {
                "run_id" : run_id,
                "relationship_id" : relationship_id,
                "source_id" : source_id
            }
            relationship_extracted_table = db.metadata.tables['macrostrat_kg.relationships_extracted']
            relationship_extracted_insert_stmt = sqlalchemy.insert(relationship_extracted_table).values(**relationship_extracted)
            db.session.execute(relationship_extracted_insert_stmt)
            db.session.commit()

        except:
            print("Failed to insert into relationship due to error", traceback.format_exc())
            db.session.rollback()
            failed_relationship += 1
            continue
    
    if failed_relationship > 0:
        return False, "Failed to insert " + str(failed_relationship) + " relationships"

    return True, ""

@app.route("/record_run", methods=["POST"])
def record_run():
    # Record the run
    request_data = request.get_json()
    run_err_msg, run_id = insert_run_metadata(request_data)
    if len(run_err_msg) != 0:
        return jsonify({"error" : run_err_msg}), 400

    if "results" in request_data:
        for current_result in  request_data["results"]:
            success, result_error_msg = insert_result(current_result, run_id)
            if not success:
                return jsonify({"error" : result_error_msg}), 400

    return jsonify({"sucess" : "Sucessfully processed all relationship"}), 200

if __name__ == "__main__":
   app.run(host = "0.0.0.0", port = 9543, debug = True)