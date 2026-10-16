from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, current_user
from src.web import db, login_manager
from src.web.models import User

auth_bp = Blueprint('auth', __name__)

@login_manager.user_loader
def load_user(id):
    return db.session.get(User, int(id))

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.overview'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user is None or not check_password_hash(user.password_hash, password):
            current_app.logger.warning(f"Failed login attempt for username: {username}")
            flash('Invalid username or password', 'error')
            return redirect(url_for('auth.login'))
            
        login_user(user)
        current_app.logger.info(f"User {username} logged in successfully.")
        
        next_page = request.args.get('next')
        if not next_page or not next_page.startswith('/'):
            next_page = url_for('dashboard.overview')
        return redirect(next_page)
        
    from flask import get_flashed_messages
    return "Login " + " ".join(get_flashed_messages())

@auth_bp.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('main.index'))

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.overview'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm = request.form.get('confirm')
        
        if password != confirm:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('auth.register'))
            
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'error')
            return redirect(url_for('auth.register'))
            
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
            return redirect(url_for('auth.register'))
            
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        
        current_app.logger.info(f"New user registered: {username}")
        flash('Congratulations, you are now a registered user!', 'success')
        return redirect(url_for('auth.login'))
        
    from flask import get_flashed_messages
    return "Register " + " ".join(get_flashed_messages())
