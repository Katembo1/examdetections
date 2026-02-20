from pathlib import Path

from dotenv import load_dotenv
from flask import Flask

from .routes import main_bp

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")
    app.register_blueprint(main_bp)
    return app
