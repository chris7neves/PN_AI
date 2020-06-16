from flask import Flask, request, render_template
from Project.heatmap_generator import generate_heatmaps
from Project.SQL.Data_Extract import SQLDatabase

app = Flask(__name__)
app.debug = True


@app.route('/')
def my_form():
    return render_template('my-form.html')


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    generate_heatmaps(text)
    return text


if __name__ == '__main__':
    app.run()
