from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
import sqlalchemy
from sqlalchemy.orm import declarative_base

# Initialize the flask app
Base = declarative_base(metadata = sqlalchemy.MetaData(schema="macrostrat_kg"))
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = sqlalchemy.URL.create(
    "postgresql",
    username="macrostrat_kg_admin",
    password="24d6d24d-19f4-4b85-95ae-f115af96ad6d",  
    host="db.development.svc.macrostrat.org",
    port = 5432,
    database="macrostrat",
)

# Create the database
db = SQLAlchemy(model_class=Base)
db.init_app(app)
with app.app_context():
    db.reflect()

def insert_into_db():
    with app.app_context():
        metadata_table = db.Model.metadata.tables['macrostrat_kg.metadata']
        new_record = {
            'run_id': "a",
            'start_time': 10,
            'end_time': 20,
            'run_description': "test"
        }
        insert_stmt = sqlalchemy.insert(metadata_table).values(**new_record)
        db.session.execute(insert_stmt)
        db.session.commit()
    print("Executed commit")

if __name__ == "__main__":
   insert_into_db() 
   # app.run(host = "0.0.0.0", port = 9543, debug = True)