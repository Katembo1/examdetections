"""Shared Flask extensions."""
from flask_migrate import Migrate
from flask_socketio import SocketIO

migrate = Migrate()
socketio = SocketIO(cors_allowed_origins="*")
