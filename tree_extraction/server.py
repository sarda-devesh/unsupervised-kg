from tree_generator import *
from flask import Flask, request, jsonify
import time
import argparse

tree_generator = None

app = Flask(__name__)

@app.route('/tree_generator', methods=['GET'])
def generate_tree():
    global tree_generator
    
    try:
        # Get the text
        data = request.get_json()
        if 'txt' not in data:
            return jsonify({"error": "Key 'txt' not found in request body"}), 400

        # Run tree generation
        start_time = time.time()
        text = data["txt"]
        result_json = tree_generator.run_for_text(text)
        result_to_return = {"sucess": result_json}
        result_to_return["time_taken"] = str(time.time() - start_time) + " seconds"

        # Return the result
        return jsonify(result_to_return)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
 
def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help = "The directory containing the model")
    return parser.parse_args()

def main():
    # Create extractor
    global tree_generator
    args = read_args()
    tree_generator = TreeGenerator(args.data_dir)

    # Start the server
    app.run(host = '0.0.0.0', port = 9500, debug = True)

if __name__ == "__main__":
    main()