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
            user = User(username='test', email='test@test.com', password_hash='hash')
            db.session.add(user)
            db.session.commit()
        yield client

def test_dashboard_unauthenticated(client):
    rv = client.get('/dashboard', follow_redirects=True)
    assert b'Login' in rv.data

def test_dashboard_authenticated(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    rv = client.get('/dashboard')
    assert rv.status_code == 200
    assert b'ClaimWise Dashboard Overview' in rv.data
    assert b'M1 \xe2\x80\x94 ML Lifecycle' in rv.data  # \xe2\x80\x94 is the em dash '—'
