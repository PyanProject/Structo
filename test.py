import open3d as o3d

# Загрузка файла .ply
file_path = "D:\GitHub\modelit\models\model_ac2dcbce.ply"  # Замените на путь к вашему файлу
mesh = o3d.io.read_triangle_mesh(file_path)

# Проверка успешной загрузки
if mesh.is_empty():
    print("Не удалось загрузить файл .ply")
else:
    print("Файл .ply успешно загружен")

# Визуализация файла
o3d.visualization.draw_geometries([mesh])
