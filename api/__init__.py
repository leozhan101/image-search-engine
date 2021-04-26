import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from flask import Flask
from flask import request, jsonify
from flask_cors import CORS
from basic.basic_search import search as basic_search
from cnn_classifier.cnn_search import search as cnn_search
from kmedoids_clustering.un_basic_search import search as un_basic_search
from kmedoids_clustering.un_cnn_search import search as un_cnn_search

app = Flask(__name__)
CORS(app)

@app.route('/')
def greeting():
    return 'Welcome to Image Recognizer backend'

@app.route('/basic_search', methods=['GET'])
def basicSearch():
    index = int(request.args.get("index"))

    response = jsonify({"results:": basic_search(index)})

    response.headers.add("Access-Control-Allow-Origin", "*")

    return response

@app.route('/cnn_search', methods=['GET'])
def cnnSearch():
    index = int(request.args.get("index"))

    response = jsonify({"results:":cnn_search(index)})

    response.headers.add("Access-Control-Allow-Origin", "*")

    return response

@app.route('/un_cnn_search', methods=['GET'])
def unCnnSearch():
    index = int(request.args.get("index"))

    response = jsonify({"results:": un_cnn_search(index)})

    response.headers.add("Access-Control-Allow-Origin", "*")

    return response

@app.route('/un_basic_search', methods=['GET'])
def unBasicSearch():
    index = int(request.args.get("index"))

    response = jsonify({"results:": un_basic_search(index)})

    response.headers.add("Access-Control-Allow-Origin", "*")

    return response
