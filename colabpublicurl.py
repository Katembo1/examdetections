import os
from app import create_app
from app.extensions import socketio

# Check if running in Google Colab
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

app = create_app()

def main() -> None:
    host = "0.0.0.0"
    port = 5000
    
    if IN_COLAB:
        # Install and start ngrok to expose the port
        os.system("pip install pyngrok")
        from pyngrok import ngrok
        
        # Replace 'YOUR_AUTHTOKEN' with your actual ngrok token if you have one
        # ngrok.set_auth_token("YOUR_AUTHTOKEN")
        
        public_url = ngrok.connect(port)
        print(f" * Public URL: {public_url}")
        print(f" * Use this URL in your Jinja2 templates or API calls")

    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)

if __name__ == "__main__":
    main()
