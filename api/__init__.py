from flask import Flask
from api import get_vector
from api.cnndescriptor import CNNDescriptor

app = Flask(__name__)

@app.route('/')
def greeting():
    return 'Welcome to Image Recognizer backend'

@app.route('/search')
def search():
    return 'search called'

# @app.route('/convert/<index>')
# def convert(index):
#     cnnd = CNNDescriptor(index)
#     result = cnnd.describe()
#     return {'vector': result}
