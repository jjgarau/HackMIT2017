from flask import Flask
from flask import request
from flask import render_template
import test

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template("hello.html")

@app.route('/', methods=['POST'])
def my_form_post():
    to = request.form['to']
    # use "to"    
    test.exec(to)
    return render_template("result.html")

if __name__ == '__main__':
    app.run()