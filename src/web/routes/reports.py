from flask import Blueprint, render_template
from flask_login import login_required
from src.web.services.artifact_service import get_json_report

reports_bp = Blueprint('reports', __name__)

@reports_bp.route('/reports')
@login_required
def reports():
    dq_report = get_json_report('data_quality_report.json')
    m5_metrics = get_json_report('m5_evaluation_metrics.json')
    final_selection = get_json_report('final_model_selection.json')
    return render_template(
        'reports.html',
        dq_report=dq_report,
        m5_metrics=m5_metrics,
        final_selection=final_selection
    )
