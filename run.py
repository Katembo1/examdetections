import os

from app import create_app

app = create_app()


def main() -> None:
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").strip().lower() == "true"

    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    main()
