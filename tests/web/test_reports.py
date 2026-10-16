import pytest
from unittest.mock import patch
from src.web import create_app, db
from src.web.models import User
from config import Config

class TestConfig:
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    SECRET_KEY = 'test_secret'
    LOGS_DIR = 'logs'
    OUTPUT_DIR = Config.OUTPUT_DIR
    WTF_CSRF_ENABLED = False

@pytest.fixture
def client():
    app = create_app(TestConfig)
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            user = User(username='test', email='test@test.com', password_hash='hash')
            db.session.add(user)
            db.session.commit()
        yield client

def test_unauthenticated_access_reports(client):
    rv = client.get('/reports', follow_redirects=True)
    assert b'Login' in rv.data

def test_authenticated_access_reports(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    rv = client.get('/reports')
    assert rv.status_code == 200
    assert b'Reports & Analytics' in rv.data
    assert b'Data Quality Report' in rv.data
    assert b'Model Evaluation Results' in rv.data
