from flask import Flask
from flask_cors import CORS

from config import Config
from extensions import db, jwt, bcrypt, migrate

from modules.auth.routes import auth_bp
from modules.assessments.routes import assess_bp
from modules.sentences.routes import sentences_bp


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Extensions
    db.init_app(app)
    jwt.init_app(app)
    bcrypt.init_app(app)
    migrate.init_app(app, db)
    CORS(app)

    # Blueprints
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(assess_bp, url_prefix="/assessment")
    app.register_blueprint(sentences_bp, url_prefix="/sentences")

    @app.route("/")
    def home():
        return "This is the home page."

    return app


if __name__ == "__main__":
    app = create_app()

    app.run(debug=True)
