from .api import api_bp

def register_blueprints(app):
    app.register_blueprint(api_bp, url_prefix='/api')
