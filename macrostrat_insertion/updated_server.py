from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import sqlalchemy
from sqlalchemy.dialects.postgresql import insert as INSERT_STATEMENT
from sqlalchemy import select as SELECT_STATEMENT
from sqlalchemy.orm import declarative_base
import json
import traceback
from datetime import datetime, timezone

from re_detail_adder import *

DEFAULT_SCHEMA = "macrostrat_kg_new"
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
CONFIG_FILE_PATH = "dev_macrostrat.json"
app, db = load_flask_app(CONFIG_FILE_PATH)
re_processor = REProcessor("id_maps")

ENTITY_TYPE_TO_ID_MAP = {
    "strat_name" : "macrostrat_strat_id",
    "lith" : "macrostrat_lith_id",
    "lith_att" : "macrostrat_lith_att_id"
}
def get_entity_id(run_id, entity_name, entity_type):
    # Create the entity value
    entity_unique_rows = ["run_id", "entity_name", "entity_type"] + list(ENTITY_TYPE_TO_ID_MAP.values())
    entities_table = db.metadata.tables['macrostrat_kg_new.entities']
    entities_values = {
        "run_id" : run_id,
        "entity_name" : entity_name,
        "entity_type" : entity_type
    }

    # Get the entity id
    entity_id = re_processor.get_entity_id(entity_name, entity_type)
    if entity_id != -1:
        key_name = ENTITY_TYPE_TO_ID_MAP[entity_type]
        entities_values[key_name] = entity_id

    # Try to create this entity
    try:
        entities_insert_statement = INSERT_STATEMENT(entities_table).values(**entities_values)
        entities_insert_statement = entities_insert_statement.on_conflict_do_nothing(index_elements = entity_unique_rows)
        db.session.execute(entities_insert_statement)
        db.session.commit()
    except:
        error_msg =  "Failed to insert entity " + entity_name +  " for run " + str(run_id) + " due to error: " + traceback.format_exc()
        return False, error_msg
    
    # Get this entity id
    entity_id = ""
    try:
        # Execute the select query
        entities_select_statement = SELECT_STATEMENT(entities_table)
        entities_select_statement = entities_select_statement.where(entities_table.c.run_id == run_id)
        entities_select_statement = entities_select_statement.where(entities_table.c.entity_name == entity_name)
        entities_select_statement = entities_select_statement.where(entities_table.c.entity_type == entity_type)
        entities_result = db.session.execute(entities_select_statement).all()

        # Ensure we got a result
        if len(entities_result) == 0:
            raise Exception("Got zero rows matching query " + str(entities_select_statement))
        
        # Extract the sources id
        first_row = entities_result[0]._mapping
        entity_id = str(first_row["entity_id"])
    except:
        error_msg =  "Failed to get sources id for entity " + str(entity_name)
        error_msg += " for run " + str(run_id) + " due to error: " + traceback.format_exc()
        return False, error_msg

    return True, entity_id

RELATIONSHIP_DETAILS = {
    "strat" : ("strat_to_lith", "strat_name", "lith"),
    "att" : ("lith_to_attribute", "lith", "lith_att")
}
def record_relationship(run_id, source_id, relationship):
    # Verify the fields
    expected_fields = ["src", "relationship_type", "dst"]
    relationship_values = {}
    for field in expected_fields:
        if field not in relationship:
            return False, "Request relationship missing field " + field
        relationship_values[field] = relationship[field]

    # Extract the types
    provided_relationship_type = relationship["relationship_type"]
    db_relationship_type, src_entity_type, dst_entity_type = "", "", ""
    for key_name in RELATIONSHIP_DETAILS:
        if provided_relationship_type.startswith(key_name):
            db_relationship_type, src_entity_type, dst_entity_type = RELATIONSHIP_DETAILS[key_name]
            break
    
    # Ignore this type
    if len(db_relationship_type) == 0 or len(src_entity_type) == 0 or len(dst_entity_type) == 0:
        return True, ""
    
    # Get the entity ids
    sucessful, src_entity_id = get_entity_id(run_id, relationship_values["src"], src_entity_type)
    if not sucessful:
        return sucessful, src_entity_id
    
    sucessful, dst_entity_id = get_entity_id(run_id, relationship_values["dst"], dst_entity_type)
    if not sucessful:
        return sucessful, dst_entity_id

    # Record the relationship
    db_relationship_insert_values = {
        "run_id" : run_id,
        "src_entity_id" : src_entity_id,
        "dst_entity_id" : dst_entity_id,
        "source_id" : source_id,
        "relationship_type" : db_relationship_type
    }
    unique_columns = ["run_id", "src_entity_id", "dst_entity_id", "relationship_type", "source_id"]
    relationship_tables = db.metadata.tables['macrostrat_kg_new.relationship']
    try:
        relationship_insert_statement = INSERT_STATEMENT(relationship_tables).values(**db_relationship_insert_values)
        relationship_insert_statement = relationship_insert_statement.on_conflict_do_nothing(index_elements = unique_columns)
        db.session.execute(relationship_insert_statement)
        db.session.commit()
    except:
        error_msg = "Failed to insert relationship type " + str(provided_relationship_type) + " for source " + str(source_id)
        error_msg += " for run " + str(run_id) + " due to error: " + traceback.format_exc()
        return False, error_msg
    
    return True, ""

def record_for_result(run_id, request):
    # Ensure txt exists
    if "text" not in request:
        return False, "result is missing text field"
    
    source_fields = ["preprocessor_id", "paper_id", "hashed_text", "weaviate_id", "paragraph_text"]
    source_values = {"run_id" : run_id}
    text_data = request["text"]
    for field_name in source_fields:
        if field_name not in text_data:
            return False, "Request text is missing field " + str(field_name)
        source_values[field_name] = text_data[field_name]
    
    # Remove non ascii data from text
    paragraph_txt = source_values["paragraph_text"]
    source_values["paragraph_text"] = paragraph_txt.encode("ascii", errors="ignore").decode()

    sources_table = db.metadata.tables['macrostrat_kg_new.sources']
    try:
        # Try to insert the sources
        sources_insert_statement = INSERT_STATEMENT(sources_table).values(**source_values)
        sources_insert_statement = sources_insert_statement.on_conflict_do_nothing(index_elements=["run_id", "weaviate_id"])
        db.session.execute(sources_insert_statement)
        db.session.commit()
    except:
        error_msg =  "Failed to insert paragraph " + str(source_values["weaviate_id"])
        error_msg += " for run " + str(source_values["run_id"]) + " due to error: " + traceback.format_exc()
        return False, error_msg
        
    # Deal with case if we have no relationships
    if "relationships" not in request:
        return True, ""
    
    # Get the sources id
    source_id = ""
    try:
        # Execute the select query
        sources_select_statement = SELECT_STATEMENT(sources_table)
        sources_select_statement = sources_select_statement.where(sources_table.c.run_id == run_id)
        sources_select_statement = sources_select_statement.where(sources_table.c.weaviate_id == source_values["weaviate_id"])
        sources_result = db.session.execute(sources_select_statement).all()

        # Ensure we got a result
        if len(sources_result) == 0:
            raise Exception("Got zero rows matching query " + str(sources_select_statement))
        
        # Extract the sources id
        first_row = sources_result[0]._mapping
        source_id = str(first_row["source_id"])
    except:
        error_msg =  "Failed to get sources id for paragraph " + str(source_values["weaviate_id"])
        error_msg += " for run " + str(source_values["run_id"]) + " due to error: " + traceback.format_exc()
        return False, error_msg
    
    # Record the relationships
    if "relationships" in request:
        for relationship in request["relationships"]:
            sucessful, message = record_relationship(run_id, source_id, relationship)
            if not sucessful:
                return sucessful, message

    # Record the entities
    if "just_entities" in request:
        required_entity_keys = ["entity", "entity_type"]
        for entity_data in request["just_entities"]:
            # Ensure that it has all the required keys
            for key in required_entity_keys:
                if key not in entity_data:
                    return False, "Provided just entities missing key " + str(key) 

            # Only record strats
            entity_type = entity_data["entity_type"]
            if not entity_type.startswith("strat"):
                continue
            
            # Record the entity
            sucessful, entity_id = get_entity_id(run_id, entity_data["entity"], "strat_name")
            if not sucessful:
                return sucessful, entity_id

    return True, ""

def process_input_request(request_data):
    # Get the metadata fields
    metadata_fields = ["run_id", "extraction_pipeline_id", "model_id"]
    metadata_values = {}
    for field_name in metadata_fields:
        if field_name not in request_data:
            return False, "Request data is missing field " + str(field_name)
        metadata_values[field_name] = request_data[field_name]

    # Insert this run to the metadata
    try:
        metadata_table = db.metadata.tables['macrostrat_kg_new.metadata']
        metadata_insert_statement = INSERT_STATEMENT(metadata_table).values(**metadata_values)
        metadata_insert_statement = metadata_insert_statement.on_conflict_do_nothing(index_elements=["run_id"])
        db.session.execute(metadata_insert_statement)
        db.session.commit()
    except Exception:
        return False, "Failed to insert run " + str(metadata_values["run_id"]) + " due to error: " + traceback.format_exc()

    # Record the results
    if "results" in request_data:
        for result in request_data["results"]:
            sucessful, error_msg = record_for_result(request_data["run_id"], result)
            if not sucessful:
                return sucessful, error_msg

    return True, ""

@app.route("/record_run", methods=["POST"])
def record_run():
    # Record the run
    sucessful, error_msg = process_input_request(request.get_json())
    if not sucessful:
        return jsonify({"error" : error_msg}), 400
    
    return jsonify({"sucess" : "Sucessfully processed the run"}), 200 

if __name__ == "__main__":
   app.run(host = "0.0.0.0", port = 9543, debug = True)