import pytest
from src.web import create_app, db
from src.web.models import User

class TestConfig:
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    SECRET_KEY = 'test_secret'
    LOGS_DIR = 'logs'

@pytest.fixture
def client():
    app = create_app(TestConfig)
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
        yield client

def test_dashboard_requires_auth(client):
    rv = client.get('/dashboard', follow_redirects=True)
    # Should redirect to login page
    assert b'Login' in rv.data
    assert b'Welcome to ClaimWise.' not in rv.data

def test_404_page(client):
    rv = client.get('/nonexistent_route')
    assert rv.status_code == 404
    assert b'404 - Page Not Found' in rv.data
