from flask import Flask
from formation_kg_generator import *

app = Flask(__name__)

arguments = {
    "formation" : "The formation we want to get the knowledge graph for",
    "model_types" : "A comma seperated list of the models we want to use to generate kg. Valid options: seq2rel, rebel",
    "article_limit" : "The number of articles we want to get snippets from. Default: 5",
    "snippets_limit" : "The maximum number of snippets we should get per article: Default: 5"
}

@app.route('/kg')
def kg_getter():
    return {"parameters" : arguments}
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 9000)