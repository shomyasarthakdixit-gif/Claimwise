import pytest
from src.web import create_app, db
from src.web.models import User

class TestConfig:
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    SECRET_KEY = 'test_secret'
    WTF_CSRF_ENABLED = False
    LOGS_DIR = 'logs'

@pytest.fixture
def client():
    app = create_app(TestConfig)
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
        yield client

def test_landing_page(client):
    rv = client.get('/')
    assert b'ClaimWise' in rv.data
    assert b'Insurance Claims Severity Analytics' in rv.data

def test_registration(client):
    rv = client.post('/register', data=dict(
        username='testuser',
        email='test@example.com',
        password='testpassword',
        confirm='testpassword'
    ), follow_redirects=True)
    assert b'Congratulations, you are now a registered user!' in rv.data

def test_login_logout(client):
    client.post('/register', data=dict(
        username='testuser',
        email='test@example.com',
        password='testpassword',
        confirm='testpassword'
    ))
    
    rv = client.post('/login', data=dict(
        username='testuser',
        password='testpassword'
    ), follow_redirects=True)
    assert b'ClaimWise Dashboard Overview' in rv.data
    
    rv = client.get('/logout', follow_redirects=True)
    assert b'Login to Dashboard' in rv.data

def test_duplicate_username(client):
    client.post('/register', data=dict(
        username='testuser',
        email='test1@example.com',
        password='testpassword',
        confirm='testpassword'
    ))
    rv = client.post('/register', data=dict(
        username='testuser',
        email='test2@example.com',
        password='testpassword',
        confirm='testpassword'
    ), follow_redirects=True)
    assert b'Username already exists.' in rv.data
