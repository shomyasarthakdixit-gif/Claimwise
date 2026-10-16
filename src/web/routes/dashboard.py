from flask import Blueprint, render_template
from flask_login import login_required
from src.web.services.artifact_service import get_dashboard_summary

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/dashboard')
@login_required
def overview():
    summary = get_dashboard_summary()
    return render_template('dashboard.html', summary=summary)
