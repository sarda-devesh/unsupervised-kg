from flask import Flask, jsonify, request
import multiprocessing
import numpy as np
import json

from formation_kg_generator import *

# Initialize the global variables
DEFAULT_ARTICLE_LIMIT = 5
DEFAULT_SNIPPEETS_LIMIT = 1
DEFAULT_MODEL = "seq2rel"
MAX_PROCESSES = min(DEFAULT_ARTICLE_LIMIT * DEFAULT_SNIPPEETS_LIMIT, 5)
arguments = {
    "formation" : "The formation we want to get the knowledge graph for",
    "model_types" : f"A comma seperated list of the models we want to use to generate kg. Valid options: seq2rel, rebel. Default: {DEFAULT_MODEL}",
    "article_limit" : f"The number of articles we want to get snippets from. Default: {DEFAULT_ARTICLE_LIMIT}",
    "snippets_limit" : f"The maximum number of snippets we should get per article: Default: {DEFAULT_SNIPPEETS_LIMIT}"
}

model_paths = {
    "seq2rel" : "/app/seq_to_rel/output/model.tar.gz",
    "rebel" : "/app/rebel_finetuning/model/archive_tuned"
}

cache_dir = "formation_cache"

# Create the processing pool
model_wrappers = dict()
def pool_init(workers_loaded):
    for model_name in model_paths: 
        curr_model_path = model_paths[model_name]
        if model_name == "seq2rel":
            model_wrappers[model_name] = Seq2RelWrapper(curr_model_path)
        elif model_name == "rebel":
            model_wrappers[model_name] = RebelWrapper(curr_model_path)
        else:
            raise Exception(f"Invalid model name of {model_name}")
    
    with workers_loaded.get_lock():
        workers_loaded.value += 1

def pool_worker(snippets, models_to_use):
    # Create the worker kg
    worker_kg = KG()
    if len(snippets) == 0:
        return worker_kg

    # Load the models to use
    curr_worker_models = []
    for model_name in models_to_use:
        curr_worker_models.append(model_wrappers[model_name])
    
    # Get the kg for the provided snippets 
    for article_id, curr_line in snippets:
        curr_line = curr_line.strip()
        for model in curr_worker_models:
            line_kg = get_kg_for_line(model, curr_line, article_id)
            worker_kg.merge_with_kb(line_kg)

    return worker_kg

workers_loaded = multiprocessing.Value('i', 0)
kg_worker_pool = multiprocessing.Pool(processes = MAX_PROCESSES, initializer=pool_init, initargs=(workers_loaded, ))

# Create the flask app
app = Flask(__name__)

@app.route('/kg')
def kg_getter():
    # Read the args
    param_values = {}
    args_exist = False
    for arg in arguments:
        value = request.args.get(arg)
        param_values[arg] = value
        if value is not None:
            args_exist = True
    
    # Return the arguments
    if not args_exist:
        return jsonify({"parameters" : arguments})
    
    # Check if formation is present
    formation_name = param_values["formation"]
    if formation_name is None:
        return jsonify({
            "result" : "failure",
            "reason" : "No formation is specified"
        })
    
    # Determine the models to use
    models_argument = param_values["model_types"]
    if models_argument is None:
        models_argument = DEFAULT_MODEL
    models_to_use = models_argument.split(" ")
    models_to_use = [model.strip() for model in models_to_use]

    for model in models_to_use:
        if model not in model_paths:
            return jsonify({
                "result" : "failure",
                "reason" : f"Invalid model type of {model}"
            })
    
    # Read in the timings
    article_limit, snippets_limit = DEFAULT_ARTICLE_LIMIT, DEFAULT_SNIPPEETS_LIMIT
    if param_values["article_limit"] is not None:
        article_limit = int(param_values["article_limit"])
    
    if param_values["snippets_limit"] is not None:
        snippets_limit = int(param_values["snippets_limit"])
    
    # Check the cache
    save_name = formation_name.replace(" ", "_") + "_" + ",".join(models_to_use) + "_" + str(article_limit) + "_" + str(snippets_limit) + ".json"
    save_path = os.path.join(cache_dir, save_name)
    if os.path.exists(save_path):
        with open(save_path, 'r') as reader:
            data = json.load(reader)
        return jsonify(data)
    
    # Load the snippets
    snippets = get_snippets_for_formation(formation_name, article_limit, snippets_limit)
    if len(snippets) == 0:
        return {
            "result" : "failure",
            "reason" : "Failed to get snippets for formation " + str(formation_name)
        }
    
    # Distribute the work among the workers
    expected_results = []
    snippets_per_worker = np.array_split(np.array(snippets), MAX_PROCESSES)
    for curr_snippets in snippets_per_worker:
        result = kg_worker_pool.apply_async(pool_worker, (list(curr_snippets), models_to_use, ))
        expected_results.append(result)
    
    # Get the kg from the workers and combine them
    formation_kg = KG()
    for result in expected_results:
        result_kg = result.get()
        formation_kg.merge_with_kb(result_kg)

    # Save the result to the cache
    json_to_write = {
        "result" : "sucess",
        "knowledge_graph" : formation_kg.get_json_representation()
    }

    with open(save_path, 'w+', encoding='utf-8') as writer:
        json.dump(json_to_write, writer, ensure_ascii=False, indent=4)
    
    # Get kg for this formation
    return jsonify(json_to_write)

if __name__ == "__main__":
    # Ensure the workers have been loaded
    all_models_loaded = False
    while not all_models_loaded:
        with workers_loaded.get_lock():
            all_models_loaded = workers_loaded.value == MAX_PROCESSES

    app.run(host='0.0.0.0', port = 9000)