import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS
from sqlalchemy import inspect
from sqlalchemy.exc import OperationalError

from .camera import init_camera_store
from .db import db
from .extensions import migrate, socketio
from .routes import main_bp

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")
    database_path = (ROOT / "app.db").as_posix()
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", f"sqlite:///{database_path}")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # CORS configuration for camera streaming
    CORS(app, resources={
        r"/video_feed/*": {
            "origins": os.getenv("CORS_ORIGINS", "*"),
            "methods": ["GET"],
            "allow_headers": ["Content-Type", "Authorization"],
            "expose_headers": ["Content-Type"],
            "supports_credentials": True
        },
        r"/stats": {"origins": "*"},
        r"/cameras*": {"origins": "*"},
        r"/config": {"origins": "*"},
        r"/inference": {"origins": "*"},
        r"/upload": {"origins": "*"}
    })

    db.init_app(app)
    migrate.init_app(app, db)
    socketio.init_app(app)

    app.register_blueprint(main_bp)

    with app.app_context():
        try:
            db.create_all()
            inspector = inspect(db.engine)
            if inspector.has_table("cameras"):
                init_camera_store()
        except OperationalError:
            # Database not initialized yet (e.g., before first migration).
            pass
        # Pre-warm the ONNX model so the first inference isn't slow
        try:
            from .inference import _load_model
            _load_model()
        except Exception:
            pass
    return app
