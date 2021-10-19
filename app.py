from flask import Flask

app = Flask(__name__)

app.secret_key = "secret_key"
app.config["UPLOAD_FOLDER"] = "static/uploads/"
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024