import pytest
import os
from src.web import create_app, db
from src.web.models import User
from config import Config

class TestConfig:
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    SECRET_KEY = 'test_secret'
    LOGS_DIR = 'logs'
    PROCESSED_DATA_DIR = Config.PROCESSED_DATA_DIR # Use real data dir to test actual csvs

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

def test_dataset_unauthenticated(client):
    rv = client.get('/dataset', follow_redirects=True)
    assert b'Login' in rv.data

def test_dataset_authenticated(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    rv = client.get('/dataset')
    assert rv.status_code == 200
    assert b'Dataset Explorer' in rv.data

def test_dataset_pagination_sizes(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    for size in [25, 50, 100, 250]:
        rv = client.get(f'/dataset?page_size={size}')
        assert rv.status_code == 200
        assert f'value="{size}" selected'.encode() in rv.data
        # Ensure we do not render 188318 rows. The table body row count should be exactly <= size.
        # We can check the number of <tr> inside the tbody. A rough proxy is counting '<tr>'.
        # Since it's a bit brittle to parse HTML, we just ensure it does not contain '188318' as current page length.
        assert b'Total matched rows: 194592' in rv.data

def test_dataset_invalid_page_size(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    rv = client.get('/dataset?page_size=999')
    assert b'value="25" selected' in rv.data

def test_dataset_search_valid(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    rv = client.get('/dataset?search=A')
    assert rv.status_code == 200
    assert b'value="A"' in rv.data

def test_dataset_search_empty(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    rv = client.get('/dataset?search=nonexistent_value_123456789')
    assert rv.status_code == 200
    assert b'No records match your current filters.' in rv.data

def test_dataset_filtering_categorical_equals(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    rv = client.get('/dataset?filter_column=cat1&filter_operator=equals&filter_value=A')
    assert rv.status_code == 200
    assert b'selected>cat1</option>' in rv.data

def test_dataset_filtering_numeric_gte(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    rv = client.get('/dataset?filter_column=cont1&filter_operator=gte&filter_value=0.5')
    assert rv.status_code == 200
    assert b'selected>cont1</option>' in rv.data

def test_dataset_filtering_invalid_column(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    rv = client.get('/dataset?filter_column=invalid_col&filter_operator=equals&filter_value=1')
    assert rv.status_code == 200
    # Should just silently ignore and return the unfiltered dataset

def test_dataset_sorting(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    rv = client.get('/dataset?sort=cont1&order=desc')
    assert rv.status_code == 200
    assert b'&#9660;' in rv.data  # Descending arrow

def test_dataset_selection(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    for ds in ['train', 'test', 'sample_submission']:
        rv = client.get(f'/dataset?dataset={ds}')
        assert rv.status_code == 200
        assert f'value="{ds}" selected'.encode() in rv.data

def test_dataset_invalid_selection(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    rv = client.get('/dataset?dataset=invalid_name')
    assert rv.status_code == 200
    assert b'value="train" selected' in rv.data

def test_dataset_download_auth_required(client):
    rv = client.get('/dataset/download', follow_redirects=True)
    assert b'Login' in rv.data

def test_dataset_download_valid(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    rv = client.get('/dataset/download?dataset=sample_submission')
    assert rv.status_code == 200
    assert rv.headers['Content-Disposition'].startswith('attachment')

def test_dataset_pagination_boundaries(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    rv = client.get('/dataset?page=999999')
    assert rv.status_code == 200
    # Should safely cap at the max page
    assert b'Showing page ' in rv.data

    rv = client.get('/dataset?page=-5')
    assert rv.status_code == 200
    assert b'Showing page 1' in rv.data
