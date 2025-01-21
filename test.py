import trimesh

# Загрузка OFF-файла
mesh = trimesh.load_mesh("airplane_0627.off")

mesh.visual.vertex_colors = [255, 0, 0, 255]  

# Отображение 3D-модели
mesh.show()
