import torch
import torch.nn as nn
import os
import logging
import sys

# Add the src package path to sys.path
current_path = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_path, "../.."))
if src_path not in sys.path:
    sys.path.append(src_path)

from src.models.text_encoder import CLIPWithProjection, CLIPEncoder
from src.models.shape_generator import ImprovedVoxelTransformer, VoxelTransformer

logger = logging.getLogger(__name__)

class TextTo3DModel(nn.Module):
    """
    Model for generating 3D objects based on text descriptions.
    """
    
    def __init__(self, config):
        super(TextTo3DModel, self).__init__()
        
        logger.info("Initializing TextTo3DModel with configuration:")
        
        # Device configuration
        self.device_str = getattr(config, 'device', 'cpu')
        
        # Check CUDA availability
        if self.device_str == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA is not available. Using CPU instead of GPU.")
            self.device_str = 'cpu'
            
        self.device = torch.device(self.device_str)
        logger.info(f"Using device: {self.device}")
        
        # Initialize text encoder
        self._init_text_encoder(config.model.text_encoder)
        
        # Initialize shape generator
        self._init_shape_generator(config.model.shape_generator)
        
        # Move model to device
        self.to(self.device)
        
        logger.info("TextTo3DModel successfully initialized")
    
    def _init_text_encoder(self, config):
        """
        Initializes the text encoder based on configuration.
        
        Args:
            config: Configuration for the text encoder.
        """
        encoder_type = getattr(config, 'type', 'CLIPWithProjection')
        logger.info(f"Initializing text encoder of type {encoder_type}")
        
        if encoder_type == 'CLIPWithProjection':
            self.text_encoder = CLIPWithProjection(config)
        elif encoder_type == 'CLIPEncoder':
            self.text_encoder = CLIPEncoder(config)
        else:
            raise ValueError(f"Unknown text encoder type: {encoder_type}")
        
        logger.info(f"Text encoder initialized. Embedding dimension: {config.embedding_dim}")
    
    def _init_shape_generator(self, config):
        """
        Initializes the shape generator based on configuration.
        
        Args:
            config: Configuration for the shape generator.
        """
        generator_type = getattr(config, 'type', 'ImprovedVoxelTransformer')
        logger.info(f"Initializing shape generator of type {generator_type}")
        
        if generator_type == 'ImprovedVoxelTransformer':
            self.shape_generator = ImprovedVoxelTransformer(config)
        elif generator_type == 'VoxelTransformer':
            self.shape_generator = VoxelTransformer(config)
        else:
            raise ValueError(f"Unknown shape generator type: {generator_type}")
        
        logger.info(f"Shape generator initialized. Voxel grid dimension: {config.voxel_dim}")
    
    def forward(self, text):
        """
        Forward pass: generating a 3D object from a text description.
        
        Args:
            text (str or List[str]): Text description or list of descriptions.
            
        Returns:
            torch.Tensor: Voxel representation of the 3D object.
        """
        # Check input type
        if isinstance(text, str):
            text = [text]
            
        # Get text embeddings
        text_embeddings = self.text_encoder(text)
        
        # Generate 3D object
        voxel_grid = self.shape_generator(text_embeddings)
        
        return voxel_grid
    
    def save(self, path):
        """
        Saves the model to a file.
        
        Args:
            path (str): Path to save the model.
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'text_encoder_type': type(self.text_encoder).__name__,
            'shape_generator_type': type(self.shape_generator).__name__
        }, path)
        logger.info(f"Model saved to: {path}")
    
    @classmethod
    def load(cls, path, config, device='cpu'):
        """
        Loads the model from a file.
        
        Args:
            path (str): Path to load the model from.
            config: Model configuration.
            device (str): Device to load the model on.
            
        Returns:
            TextTo3DModel: Loaded model.
        """
        checkpoint = torch.load(path, map_location=device)
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        logger.info(f"Model loaded from: {path}")
        return model

    def generate(self, text_prompt, threshold=0.5):
        """
        Generates a 3D object from a text description.
        
        Args:
            text_prompt (str): Text description of the object.
            threshold (float): Threshold for binarizing the voxel grid.
            
        Returns:
            torch.Tensor: Binarized voxel representation of the 3D object.
        """
        # Switch model to evaluation mode
        self.eval()
        
        with torch.no_grad():
            # Generate voxel grid
            voxel_grid = self.forward([text_prompt])
            
            # Binarize using threshold
            binary_voxel_grid = (torch.sigmoid(voxel_grid) > threshold).float()
        
        return binary_voxel_grid
    
    def enable_gradient_checkpointing(self):
        """
        Enables gradient checkpointing to save memory.
        This slows down training but significantly reduces memory usage.
        """
        # Enable gradient checkpointing in shape generator
        if hasattr(self.shape_generator, 'use_gradient_checkpointing'):
            self.shape_generator.use_gradient_checkpointing = True
            print("Gradient checkpointing enabled for shape generator")
        
        # Enable gradient checkpointing in text encoder if not frozen
        if not self.config.model.text_encoder.freeze and hasattr(self.text_encoder, 'model'):
            if hasattr(self.text_encoder.model, 'gradient_checkpointing_enable'):
                self.text_encoder.model.gradient_checkpointing_enable()
                print("Gradient checkpointing enabled for text encoder")
        
        print("Gradient checkpointing successfully configured for memory saving")
    
    def get_memory_usage(self):
        """
        Returns the current memory usage of the model.
        
        Returns:
            dict: Dictionary with memory usage information.
        """
        memory_stats = {}
        
        if torch.cuda.is_available():
            memory_stats['allocated'] = torch.cuda.memory_allocated() / 1e9  # In GB
            memory_stats['cached'] = torch.cuda.memory_reserved() / 1e9  # In GB
            memory_stats['max_allocated'] = torch.cuda.max_memory_allocated() / 1e9  # In GB
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        memory_stats['total_params'] = total_params
        memory_stats['trainable_params'] = trainable_params
        memory_stats['frozen_params'] = total_params - trainable_params
        
        return memory_stats 