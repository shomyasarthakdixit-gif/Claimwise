import os
from flask import Blueprint, render_template, current_app, abort, send_file
from flask_login import login_required
from src.web.services import artifact_service

modules_bp = Blueprint('modules', __name__)

@modules_bp.route('/lifecycle')
@login_required
def lifecycle():
    report = artifact_service.get_json_report('data_quality_report.json')
    return render_template('modules/lifecycle.html', report=report)

@modules_bp.route('/linear-models')
@login_required
def linear_models():
    baseline = {
        'model': 'Ridge Regression',
        'r2': '0.4928',
        'rmse': '2023.31',
        'mae': '1352.19',
        'sample_size': '15,000',
        'random_seed': '42'
    }
    return render_template('modules/linear_models.html', baseline=baseline)

@modules_bp.route('/tree-models')
@login_required
def tree_models():
    return render_template('modules/tree_models.html')

@modules_bp.route('/clustering')
@login_required
def clustering():
    return render_template('modules/clustering.html')

@modules_bp.route('/evaluation')
@modules_bp.route('/model-evaluation')
@login_required
def evaluation():
    metrics = artifact_service.get_json_report('m5_evaluation_metrics.json')
    return render_template('modules/evaluation.html', metrics=metrics)

@modules_bp.route('/artifact/plot/<path:filename>')
@login_required
def serve_plot(filename):
    if '..' in filename or filename.startswith('/') or filename.startswith('\\'):
        abort(404)
    filepath = artifact_service.get_plot_path(filename)
    if not filepath or not os.path.exists(filepath):
        abort(404)
    return send_file(filepath)
