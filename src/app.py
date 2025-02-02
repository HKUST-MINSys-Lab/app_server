from flask import Flask
from src.routes.upload_routes import upload_bp
from src.routes.gpt_routes import gpt_bp
from dotenv import load_dotenv
import os

load_dotenv()

def create_app():
    app = Flask(__name__)
    
    # Register blueprints
    app.register_blueprint(upload_bp)
    app.register_blueprint(gpt_bp)
    
    # Ensure uploads directory exists
    os.makedirs('uploads', exist_ok=True)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)