from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def root():
    data = {
        "title": "Home",
    }
    return render_template("index.html", data=data)


def page_not_found(error):
    return render_template('404.html'), 404


app.register_error_handler(404, page_not_found)
