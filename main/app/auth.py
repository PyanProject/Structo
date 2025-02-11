from flask import Blueprint, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
from app.db_models import User
from app import db

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/auth_status', methods=['GET'])
def auth_status():
    user_id = session.get('user_id')
    print(f"[DEBUG] Проверка сессии: user_id={user_id}, session.permanent={session.permanent}")
    if user_id:
        user = User.query.get(user_id)
        if user:
            return jsonify({'authenticated': True, 'username': user.username})
    return jsonify({'authenticated': False})

@auth_bp.route('/auth', methods=['POST'])
def auth():
    data = request.get_json()
    action = data.get('action')

    if action == 'login':
        username = data.get('username')
        password = data.get('password')
        remember_me = data.get('remember_me', False)

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session.permanent = remember_me
            print(f"[DEBUG] Успешный вход: user_id={user.id}, remember_me={remember_me}, session.permanent={session.permanent}")
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'Неверный логин или пароль'}), 401

    elif action == 'register':
        username = data.get('username')
        password = data.get('password')
        email = data.get('email')

        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'message': 'Пользователь с таким логином уже существует'}), 400

        password_hash = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
        new_user = User(username=username, password=password_hash, email=email)
        db.session.add(new_user)
        db.session.commit()

        return jsonify({'success': True})

@auth_bp.route('/logout', methods=['GET'])
def logout():
    session.pop('user_id', None)
    return jsonify({'success': True})

@auth_bp.route('/set_language', methods=['POST'])
def set_language():
    data = request.get_json()
    lang = data.get('lang', 'en')
    session['lang'] = lang
    return jsonify({'success': True})
