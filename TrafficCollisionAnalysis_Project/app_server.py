from flask import Flask, jsonify, render_template, request
from flask_cors import CORS, cross_origin
import io
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import Flask
import numpy as np
# from AIProject4Api import SupervisedLearning
from ProcessedBestModel import ProcessedBestModel


app = Flask(__name__)


@app.route('/predict/', methods=['GET','POST'])
@cross_origin()
def prediction():
    para = []
    if (request.method=='POST'):
        for k in request.form:
            para.append(request.form[k])
        print(para)
        result = model.predict(para)
    print(result)
    return result

@app.route('/forecast/')
@cross_origin()
def forecast():
    return render_template('forecast.html')
    # result = -1
    # para = []
    # if (request.method=='POST'):
    #     for k in request.form:
    #         para.append(request.form[k])
    #     print(para)
    #     result = model.predict(para)
    # return result
    # return render_template('forecast.html',result=result)




@app.route('/', methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/getscore/<int:a>', methods=['GET'])
@cross_origin()
def predict(a):
    result = model.getresult(a)
    return jsonify({'result': result})


if __name__ == '__main__':
    global model
    model = ProcessedBestModel()
    print('model loaded!')
    app.run(host='0.0.0.0',port=8080,debug=True)
    # app.run(debug=True)
    