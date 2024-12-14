# открывашка йоу
'''
import open3d as o3d

file_path = "D:\Downloads\model_cf642d76 (2).ply"
mesh = o3d.io.read_triangle_mesh(file_path)

if mesh.is_empty():
    print("Не удалось загрузить файл .ply")
else:
    print("Файл .ply успешно загружен")

o3d.visualization.draw_geometries([mesh])

'''

import dlltracer
import sys

with dlltracer.Trace(out=sys.stdout):
    import open3d as o3d
    print(o3d)
