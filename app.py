from flask import Flask, render_template, request,jsonify
import time
import threading
from flask_cors import *
import single_chat.dialog as dialog
import os
from see_speak import see_evaluate
import jieba
from wsgiref.simple_server import make_server
import gpt_2.predict_mmi as predictor_mmi
import gpt_2.predict as predictor


def heartbeat():
    print (time.strftime('%Y-%m-%d %H:%M:%S - heartbeat', time.localtime(time.time())))
    timer = threading.Timer(60, heartbeat)
    timer.start()


timer = threading.Timer(60, heartbeat)
timer.start()

app = Flask(__name__, static_url_path="/static")
CORS(app, supports_credentials=True)

history = list()


@app.route('/flush', methods=['post'])
def reply_flush():
    global history
    history = []
    return jsonify({'reply': 'success'})


@app.route('/single', methods=['post'])
def reply():
    global history
    history = []
    msg = request.form['msg']
    msg = " ".join(jieba.cut(msg))
    reply_single = dialog.predict(msg)
    reply_single = reply_single.replace('_UNK', '^_^').replace(" ", "")
    return jsonify({'reply': reply_single})


@app.route('/multi', methods=['post'])
def multi_reply():
    global history
    msg = request.form['msg']
    type = int(request.form['type'])
    if type == 0:
        reply_gpt, history = predictor.predict(msg, history)
    else:
        reply_gpt, history = predictor_mmi.predict(msg, history)
    # if len(history) > 20:
    #     history = history[-12:]
    #     print(history)
    print(history)
    print(len(history))
    return jsonify({'reply': reply_gpt})


@app.route('/img', methods=['post'])
def img_reply():
    global history
    img = request.files['img']
    print(img)
    img_name = request.form['name']
    mode = request.form['mode']
    filepath = 'temp/'
    if img:
        filepath = os.path.join(filepath, img_name)
        img.save(filepath)
        description = see_evaluate.predict_single(filepath)
        description = description.replace(" ", "")
        try:
            os.remove(filepath)
        except Exception as e:
            print(e)
    if int(mode) == 1:
        description, history = predictor.predict(description, history)
    return jsonify({'reply': description})


@app.route("/")
def index():
    return render_template("static/index.html")


if __name__ == '__main__':
    server = make_server('127.0.0.1', 5000, app)
    server.serve_forever()
    app.run(host='0.0.0.0',port=5000)
    print('server start')