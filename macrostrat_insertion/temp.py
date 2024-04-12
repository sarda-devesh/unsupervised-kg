import sqlalchemy

def main():
    url_object = sqlalchemy.URL.create(
        "postgresql",
        username="macrostrat_kg_admin",
        password="24d6d24d-19f4-4b85-95ae-f115af96ad6d",  
        host="db.development.svc.macrostrat.org",
        port = 5432,
        database="macrostrat",
    )
    engine = sqlalchemy.create_engine(url_object)

    inspector = sqlalchemy.inspect(engine)
    schemas = inspector.get_schema_names()
    for schema in schemas:
        schema_name = str(schema)
        if schema_name == "macrostrat_kg":
            for table_name in inspector.get_table_names(schema=schema):
                for column in inspector.get_columns(table_name, schema=schema):
                    print("Table: %s, Column: %s" % (table_name, column))

if __name__ == "__main__":
    main()