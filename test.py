import open3d as o3d

# Загрузка файла .ply
file_path = "D:\Downloads\model_cf642d76 (2).ply"  # Замените на путь к вашему файлу
mesh = o3d.io.read_triangle_mesh(file_path)

# Проверка успешной загрузки
if mesh.is_empty():
    print("Не удалось загрузить файл .ply")
else:
    print("Файл .ply успешно загружен")

# Визуализация файла
o3d.visualization.draw_geometries([mesh])
