import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Загрузка .ply файла
ply_file = "models/model_8277e091.ply"
mesh = trimesh.load(ply_file)

# Извлечение данных о вершинах и гранях
vertices = mesh.vertices
faces = mesh.faces

# Отображение с помощью matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Добавление полигонов в matplotlib
ax.add_collection3d(Poly3DCollection(vertices[faces], alpha=0.5, edgecolor='k'))

# Настройка осей
ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

plt.show()
