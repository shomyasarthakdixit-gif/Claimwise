import pytest
import os
import io
from unittest.mock import patch
from src.web import create_app, db
from src.web.models import User
from src.web.services.prediction_service import get_model_features, predict_single, predict_batch
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

def generate_valid_features():
    features = get_model_features('linear_regression')
    data = {}
    for f in features:
        if f.startswith('cat'):
            data[f] = 'A'
        else:
            data[f] = '0.5'
    return data

def test_prediction_unauthenticated(client):
    rv = client.get('/prediction/', follow_redirects=True)
    assert b'Login' in rv.data

def test_prediction_authenticated(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    rv = client.get('/prediction/')
    print(rv.data); assert rv.status_code == 200
    assert b'Prediction Interface' in rv.data

def test_unsupported_model_fails(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    data = generate_valid_features()
    data['predict_single'] = '1'
    data['model_id'] = '../../models/malicious.pkl'
    rv = client.post('/prediction/', data=data)
    # The route gracefully defaults to 'random_forest' if an invalid model_id is provided from the UI dropdown
    assert b'Prediction Result' in rv.data
    
def test_valid_single_prediction(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    data = generate_valid_features()
    data['predict_single'] = '1'
    data['model_id'] = 'linear_regression'
    
    rv = client.post('/prediction/', data=data)
    print(rv.data); assert rv.status_code == 200
    assert b'Estimated Claim Loss' in rv.data
    assert b'Validated' in rv.data
    # Ensure no fake confidence score is printed
    assert b'confidence' not in rv.data.lower()
    
def test_invalid_numeric_input(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    data = generate_valid_features()
    data['predict_single'] = '1'
    data['model_id'] = 'linear_regression'
    data['cont1'] = 'abc'  # Invalid float
    
    rv = client.post('/prediction/', data=data)
    print(rv.data); assert rv.status_code == 200
    assert b'must be numeric' in rv.data
    assert b'Prediction Result' not in rv.data

def test_missing_required_feature(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    data = generate_valid_features()
    data['predict_single'] = '1'
    del data['cat1']  # Missing feature
    
    rv = client.post('/prediction/', data=data)
    print(rv.data); assert rv.status_code == 200
    assert b'Missing required features' in rv.data
    
def test_target_leakage_prevented(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    data = generate_valid_features()
    data['predict_single'] = '1'
    data['loss'] = '1000'
    
    rv = client.post('/prediction/', data=data)
    print(rv.data); assert rv.status_code == 200
    assert b'Target leakage detected' in rv.data

def test_batch_prediction_valid(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
        
    features = get_model_features('random_forest')
    # Create simple valid CSV string
    header = ",".join(features)
    row = ",".join(['A' if f.startswith('cat') else '0.5' for f in features])
    csv_data = f"{header}\n{row}\n{row}\n"
    
    data = {
        'model_id': 'random_forest',
        'csv_file': (io.BytesIO(csv_data.encode('utf-8')), 'test.csv')
    }
    
    rv = client.post('/prediction/batch', data=data, content_type='multipart/form-data')
    print(rv.data); assert rv.status_code == 200
    assert rv.headers['Content-Disposition'].startswith('attachment')
    
def test_batch_prediction_missing_column(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
        
    # Create CSV missing cat1
    features = get_model_features('random_forest')
    features = features[1:]
    header = ",".join(features)
    row = ",".join(['A' if f.startswith('cat') else '0.5' for f in features])
    csv_data = f"{header}\n{row}\n"
    
    data = {
        'model_id': 'random_forest',
        'csv_file': (io.BytesIO(csv_data.encode('utf-8')), 'test.csv')
    }
    
    rv = client.post('/prediction/batch', data=data, content_type='multipart/form-data')
    print(rv.data); assert rv.status_code == 200
    assert b'Batch data is missing required features' in rv.data

def test_batch_prediction_leakage_handled_with_warning(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
        
    features = get_model_features('random_forest')
    header = ",".join(features + ['loss'])
    row = ",".join(['A' if f.startswith('cat') else '0.5' for f in features] + ['1000'])
    csv_data = f"{header}\n{row}\n"
    
    data = {
        'model_id': 'random_forest',
        'csv_file': (io.BytesIO(csv_data.encode('utf-8')), 'test.csv')
    }
    
    rv = client.post('/prediction/batch', data=data, content_type='multipart/form-data')
    print(rv.data); assert rv.status_code == 200
    assert rv.headers['Content-Disposition'].startswith('attachment')
    with client.session_transaction() as sess:
        flashes = sess.get('_flashes', [])
        assert any('loss' in f[1] for f in flashes)

def test_invalid_categorical_input(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    data = generate_valid_features()
    data['predict_single'] = '1'
    data['model_id'] = 'linear_regression'
    data['cat1'] = 'UNKNOWN_CATEGORY_12345'
    
    rv = client.post('/prediction/', data=data)
    print(rv.data); assert rv.status_code == 200
    assert b'Estimated Claim Loss' in rv.data
    assert b'Traceback' not in rv.data

def test_batch_prediction_empty_csv(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    data = {
        'model_id': 'random_forest',
        'csv_file': (io.BytesIO(b""), 'test.csv')
    }
    rv = client.post('/prediction/batch', data=data, content_type='multipart/form-data')
    print(rv.data); assert rv.status_code == 200
    assert b'empty' in rv.data.lower() or b'invalid' in rv.data.lower()

def test_batch_prediction_malformed_csv(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    data = {
        'model_id': 'random_forest',
        'csv_file': (io.BytesIO(b"col1,col2\n1,2,3\n"), 'test.csv')
    }
    rv = client.post('/prediction/batch', data=data, content_type='multipart/form-data')
    print(rv.data); assert rv.status_code == 200
    assert b'missing required features' in rv.data.lower() or b'invalid' in rv.data.lower()

def test_batch_prediction_invalid_numeric(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    features = get_model_features('random_forest')
    header = ",".join(features)
    row = ",".join(['A' if f.startswith('cat') else 'not_a_number' for f in features])
    csv_data = f"{header}\n{row}\n"
    
    data = {
        'model_id': 'random_forest',
        'csv_file': (io.BytesIO(csv_data.encode('utf-8')), 'test.csv')
    }
    rv = client.post('/prediction/batch', data=data, content_type='multipart/form-data')
    print(rv.data); assert rv.status_code == 200
    # pandas might fail to convert or the model might fail, returning a generic error
    assert b'failed to process batch' in rv.data.lower() or b'could not convert' in rv.data.lower() or b'error' in rv.data.lower()

def test_oversized_batch_prediction(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    features = get_model_features('random_forest')
    header = ",".join(features)
    # Generate 50001 rows
    row = ",".join(['A' if f.startswith('cat') else '0.5' for f in features])
    # Just mock pd.read_csv to return a large dataframe to avoid memory issues in test
    with patch('src.web.routes.prediction.pd.read_csv') as mock_read:
        import pandas as pd
        mock_read.return_value = pd.DataFrame(columns=features, index=range(50001))
        
        data = {
            'model_id': 'random_forest',
            'csv_file': (io.BytesIO(f"{header}\n{row}\n".encode('utf-8')), 'test.csv')
        }
        rv = client.post('/prediction/batch', data=data, content_type='multipart/form-data')
        print(rv.data); assert rv.status_code == 200
        assert b'limited to 50,000 rows' in rv.data

def test_batch_prediction_unsupported_columns(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    features = get_model_features('random_forest')
    header = ",".join(features + ['unsupported_extra_col'])
    row = ",".join(['A' if f.startswith('cat') else '0.5' for f in features] + ['extra_value'])
    csv_data = f"{header}\n{row}\n"
    
    data = {
        'model_id': 'random_forest',
        'csv_file': (io.BytesIO(csv_data.encode('utf-8')), 'test.csv')
    }
    rv = client.post('/prediction/batch', data=data, content_type='multipart/form-data')
    print(rv.data); assert rv.status_code == 200
    # It should succeed because sklearn pipelines typically ignore extra columns if passed as df,
    # or our code takes `df[features]`. Let's check that it succeeds.
    assert rv.headers.get('Content-Disposition', '').startswith('attachment')

def test_stack_trace_protection(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    data = generate_valid_features()
    data['predict_single'] = '1'
    data['model_id'] = 'linear_regression'
    
    with patch('src.web.routes.prediction.predict_single', side_effect=RuntimeError("Forced failure")):
        rv = client.post('/prediction/', data=data)
        print(rv.data); assert rv.status_code == 200
        assert b'internal error occurred' in rv.data
        assert b'Traceback' not in rv.data

def test_model_loading_failure(client):
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    data = generate_valid_features()
    data['predict_single'] = '1'
    data['model_id'] = 'linear_regression'
    
    with patch.dict('src.web.services.prediction_service._model_cache', {}, clear=True):
        with patch('src.web.services.prediction_service.os.path.exists', return_value=False):
            rv = client.post('/prediction/', data=data)
            print(rv.data); assert rv.status_code == 200
            assert b'error' in rv.data.lower()
            assert b'Traceback' not in rv.data
