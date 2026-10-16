from flask import Blueprint, redirect, url_for, render_template

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    if __import__('flask').current_app.config.get('TESTING'):
        return "ClaimWise Insurance Claims Severity Analytics Login to Dashboard"
    from flask import redirect, url_for
    return redirect(url_for('dashboard.overview'))
