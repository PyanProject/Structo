Text-to-3D Model Generator
Overview
This system generates 3D models from textual descriptions. It uses BERT to create embeddings from input text, which are then transformed into random 3D scenes using Open3D.

Features
Text Embedding Generation: Uses BERT to convert text into embeddings.
3D Scene Generation: Uses Open3D to visualize and save randomly generated 3D scenes.
Saving Embeddings: Saves generated embeddings as .npy files in the temp_emb directory.

INSTALLATION
Clone the repository:

git clone https://github.com/your-repository-url
cd your-project-directory

Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Usage

Run the main script to start the text-to-3D process:

python main.py

Provide a textual description when prompted.

The system will:

Generate an embedding for the text.
Create a random 3D scene based on the embedding.
Save the 3D model in generated_mesh.ply and the embedding in the temp_emb folder.

Recommendations for Developers

When setting up the environment:

Ensure all dependencies are installed by running pip install -r requirements.txt.
If you’re deploying on a new system, use a virtual environment to avoid conflicts with global packages.

License

MIT License

This README explains the system’s purpose, setup instructions, and usage steps, while also offering recommendations for installing dependencies.


!!! FOR DEVS !!!:

lib docs:
open3d - https://www.open3d.org/docs/release/index.html
transformers - https://github.com/huggingface/transformers/blob/main/i18n/README_ru.md
numpy - https://numpy.org/doc/stable/

load large files:

https://drive.google.com/file/d/1ivyHzADX2YOCwJgi37EmtCS-X-N8eJqZ/view?usp=sharing

place to:
venv/Lib/site-packages/open3d/cpupybind.cp311-win_amd64.pyd
venv/Lib/site-packages/torch/lib/torch_cpu.dll
venv/Lib/site-packages/torch/lib/dnnl.lib





