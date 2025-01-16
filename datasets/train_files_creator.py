import os

root_dir = "datasets\ModelNet40"  # Укажи путь к ModelNet40
split = "train"  # Или "test"

file_list = []
for class_name in os.listdir(root_dir):
    class_path = os.path.join(root_dir, class_name, split)
    if os.path.isdir(class_path):
        for file_name in os.listdir(class_path):
            if file_name.endswith(".off"):  # Формат ModelNet40
                file_list.append(f"{class_name}/{split}/{file_name}")

with open(os.path.join(root_dir, f"{split}_files.txt"), "w") as f:
    f.write("\n".join(file_list))

print(f"Файл {split}_files.txt успешно создан!")
