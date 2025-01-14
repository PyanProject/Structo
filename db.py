import sqlite3

conn = sqlite3.connect('instance/users.db')
cursor = conn.cursor()

# Проверяем все таблицы
cursor.execute("PRAGMA table_info('users');")
columns = cursor.fetchall()
print(columns)


conn.close()
