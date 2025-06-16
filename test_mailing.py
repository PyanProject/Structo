import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(recipient_email, username):
    """
    Отправляет приветственное письмо пользователю через Яндекс почту
    """
    # Настройки SMTP для Яндекс почты
    smtp_server = "smtp.yandex.ru"
    smtp_port = 465
    sender_email = "email here"  # Ваш Яндекс email
    password = "password here"  # Пароль приложения из настроек Яндекса

    try:
        # Создаем сообщение
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = "Добро пожаловать!"

        body = f"""
        Здравствуйте, {username}!
        
        Спасибо за регистрацию на нашем сайте.
        
        С уважением,
        Команда поддержки
        """
        msg.attach(MIMEText(body, 'plain', 'utf-8'))  # Добавляем кодировку utf-8

        # Создаем SSL соединение
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, password)
            server.send_message(msg)
        
        print(f"[DEBUG] Письмо успешно отправлено на {recipient_email}")
        return True

    except Exception as e:
        print(f"[ERROR] Ошибка при отправке письма: {str(e)}")
        return False
    

send_email("qwertymaxqazwsx@gmail.com", 'test')