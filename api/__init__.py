from flask import Flask
from api import get_vector

app = Flask(__name__)

@app.route('/')
def greeting():
    return 'Welcome to Image Recognizer backend'

@app.route('/get-vector')
def getVector():
    get_vector.test()
    return 'get-vector called'
