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

history = []


@app.route('/single', methods=['post'])
def reply():
    history = []
    msg = request.form['msg']
    msg = " ".join(jieba.cut(msg))
    reply_single = dialog.predict(msg)
    reply_single = reply_single.replace('_UNK', '^_^').replace(" ", "")
    return jsonify({'reply': reply_single})


@app.route('/multi', methods=['post'])
def multi_reply():
    msg = request.form['msg']
    type = int(request.form['type'])
    if type == 0:
        reply_gpt, history_re = predictor.predict(msg, history)
        history.extend(history_re)
    else:
        reply_gpt, history_re = predictor_mmi.predict(msg, history)
        history.extend(history_re)
    return jsonify({'reply': reply_gpt})


@app.route('/img', methods=['post'])
def img_reply():
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
        description, history_re = predictor.predict(description, history)
        history.extend(history_re)
    return jsonify({'reply': description})


@app.route("/")
def index():
    return render_template("static/index.html")

if __name__ == '__main__':
    server = make_server('127.0.0.1', 5000, app)
    server.serve_forever()
    app.run(host='0.0.0.0',port=5000)
    print('server start')