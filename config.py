import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'claimwise-dev-key')
    SQLALCHEMY_DATABASE_URI = 'sqlite:///claimwise.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Path configuration
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    INTERIM_DATA_DIR = os.path.join(DATA_DIR, 'interim')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    
    OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')
    MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
    PLOT_DIR = os.path.join(OUTPUT_DIR, 'plot')
    REPORT_DIR = os.path.join(OUTPUT_DIR, 'report')
    PREDICTION_DIR = os.path.join(OUTPUT_DIR, 'predictions')
    EVALUATION_DIR = os.path.join(OUTPUT_DIR, 'evaluation')
    CLUSTERING_DIR = os.path.join(OUTPUT_DIR, 'clustering')
    
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
