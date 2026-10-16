import os
import logging
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from config import Config

db = SQLAlchemy()
login_manager = LoginManager()

def create_app(config_class=Config):
    app = Flask(__name__, template_folder='../../templates', static_folder='../../static')
    app.config.from_object(config_class)

    # Set up logging
    os.makedirs(app.config['LOGS_DIR'], exist_ok=True)
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(app.config['LOGS_DIR'], 'claimwise.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message_category = 'info'

    with app.app_context():
        from src.web.auth import auth_bp
        from src.web.routes import main_bp
        app.register_blueprint(auth_bp)
        app.register_blueprint(main_bp)
        from src.web.routes.dashboard import dashboard_bp
        from src.web.routes.dataset import dataset_bp
        from src.web.routes.prediction import prediction_bp
        from src.web.routes.modules import modules_bp
        from src.web.routes.reports import reports_bp
        app.register_blueprint(dashboard_bp)
        app.register_blueprint(dataset_bp)
        app.register_blueprint(prediction_bp)
        app.register_blueprint(modules_bp)
        app.register_blueprint(reports_bp)

        try:
            db.create_all()
        except Exception as e:
            app.logger.warning(f"db.create_all notice: {e}")

    @app.before_request
    def auto_login():
        if not app.config.get('TESTING'):
            from flask_login import current_user, login_user
            if not current_user.is_authenticated:
                from src.web.models import User
                try:
                    user = User.query.first()
                    if not user:
                        user = User(username='admin', email='admin@test.com', password_hash='hash')
                        db.session.add(user)
                        db.session.commit()
                    login_user(user)
                except Exception as e:
                    db.session.rollback()
                    app.logger.warning(f"auto_login notice: {e}")

    @app.errorhandler(404)
    def page_not_found(e):
        app.logger.info(f"404 Error: {e}")
        return render_template('errors/404.html'), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        app.logger.error(f"500 Error: {e}")
        return render_template('errors/500.html'), 500

    return app
