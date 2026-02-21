import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask

from .camera import init_camera_store
from .db import db
from .routes import main_bp

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")
    database_path = (ROOT / "app.db").as_posix()
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", f"sqlite:///{database_path}")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)

    app.register_blueprint(main_bp)

    with app.app_context():
        db.create_all()
        init_camera_store()
    return app
