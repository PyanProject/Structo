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



    ''' def generate_embedding(self, text: str, additional_info: str = "", shape_info: dict = None, lang: str = "en") -> torch.Tensor:
        try:
            preprocessed_text = self.preprocess_text(text, lang)
            combined_text = self.combine_text(preprocessed_text, additional_info, shape_info)
            
            print(f"[EMBED] Генерация эмбеддинга для текста: '{combined_text}'")
            text_input = clip.tokenize([combined_text]).to(self.device)

            with torch.no_grad():
                text_features = self.model.encode_text(text_input)
            
            print(f"[EMBED] Эмбеддинг CLIP сгенерирован. Размерность: {text_features.shape}")

            if hasattr(self, 'reduce_dim_layer'):
                text_features = self.reduce_dim_layer(text_features)
                print(f"[EMBED] Размерность эмбеддинга уменьшена до {text_features.shape[1]}.")

            embedding_filepath = self.save_embedding(text_features)
            if embedding_filepath:
                print(f"[EMBED] Эмбеддинг сохранён: {embedding_filepath}")
            else:
                print("[EMBED] Ошибка при сохранении эмбеддинга.")
            
            return text_features

        except ValueError as ve:
            print(f"[EMBED] Ошибка валидации текста: {ve}")
            raise ve

        except Exception as e:
            print(f"[EMBED] Общая ошибка при обработке текста: {e}")
            raise RuntimeError(f"Ошибка обработки текста: {e}") '''