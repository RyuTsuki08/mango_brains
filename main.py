from flask import Flask, render_template
from src.routes.plots import plots
app = Flask(__name__)
app.register_blueprint(plots)

@app.route("/")
def root():
    data = {
        "title": "Home",
    }
    return render_template("index.html", data=data)


def page_not_found(error):
    return render_template('404.html'), 404


app.register_error_handler(404, page_not_found)

# if __name__ == "__main__":
#     app.run(debug=True)
#     print(app.url_map)
