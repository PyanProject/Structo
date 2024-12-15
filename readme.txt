# Text-to-3D Model Generator

## Overview
This system generates 3D models from textual descriptions. It uses BERT to create embeddings from input text, which are then transformed into random 3D scenes using Open3D.

## Features
- **Text Embedding Generation**: Uses CLIP to convert text into embeddings.
- **3D Scene Generation**: Uses Open3D to visualize and save randomly generated 3D scenes.
- **Saving Embeddings**: Saves generated embeddings as `.npy` files in the `temp_emb` directory.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repository-url
   cd your-project-directory
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the main script to start the text-to-3D process:
   ```bash
   python main.py
   ```

2. Provide a textual description when prompted.

3. The system will:
   - Generate an embedding for the text.
   - Create a random 3D scene based on the embedding.
   - Save the 3D model in `generated_mesh.ply` and the embedding in the `temp_emb` folder.

## Recommendations for Developers

When setting up the environment:
- Ensure all dependencies are installed by running `pip install -r requirements.txt`.
- If you’re deploying on a new system, use a virtual environment to avoid conflicts with global packages.

## License
MIT License
