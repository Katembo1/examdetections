import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS

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

    app.register_blueprint(main_bp)

    with app.app_context():
        db.create_all()
        init_camera_store()
    return app
