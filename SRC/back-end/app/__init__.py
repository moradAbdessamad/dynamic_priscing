from flask import Flask #type: ignore
from app.utils.extensions import init_extensions
from .blueprints import register_blueprints
from flask_cors import CORS #type: ignore

def create_app():
    app = Flask(__name__)
    
    CORS(app)

    init_extensions(app)
    register_blueprints(app)
        
    return app