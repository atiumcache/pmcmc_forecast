import json
from os.path import join

from flask import Flask, jsonify, render_template

from src import paths

app = Flask(__name__)


@app.route("/data")
def get_data():
    # TODO: Un-hardcode the following path.
    json_path = join(paths.PMCMC_RUNS_DIR, "04", "monsoon_test_20240827.json")
    with open(json_path, "r") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
