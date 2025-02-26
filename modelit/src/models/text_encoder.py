import torch
import torch.nn as nn
import clip
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast
import logging

logger = logging.getLogger(__name__)

class CLIPWithProjection(nn.Module):
    """
    Enhanced text encoder based on CLIP with additional projection layer
    for adapting text embeddings to the 3D model generation task.
    """
    
    def __init__(self, config):
        super(CLIPWithProjection, self).__init__()
        
        self.embedding_dim = config.embedding_dim
        self.pretrained = config.pretrained
        self.freeze = config.freeze
        self.model_name = getattr(config, 'model_name', 'openai/clip-vit-large-patch14')
        
        # Loading CLIP model from Hugging Face
        try:
            logger.info(f"Loading CLIP model from {self.model_name}")
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.tokenizer = CLIPTokenizerFast.from_pretrained(self.model_name)
            self.clip_model = CLIPModel.from_pretrained(self.model_name)
            
            # Get output embedding dimension from CLIP
            self.clip_embedding_dim = self.clip_model.text_model.config.hidden_size
            logger.info(f"CLIP model loaded. Embedding dimension: {self.clip_embedding_dim}")
            
        except Exception as e:
            logger.error(f"Error loading CLIP model: {str(e)}")
            logger.info("Falling back to OpenAI CLIP")
            
            # Fallback to original OpenAI CLIP
            self.clip_model, self.preprocess = clip.load("ViT-L/14", device='cpu')
            self.clip_embedding_dim = self.clip_model.ln_final.weight.shape[0]
            logger.info(f"OpenAI CLIP model loaded. Embedding dimension: {self.clip_embedding_dim}")
        
        # Freeze CLIP parameters if required
        if self.freeze:
            logger.info("Freezing CLIP parameters")
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Projection layer to adapt CLIP embeddings to target dimension
        if self.clip_embedding_dim != self.embedding_dim:
            logger.info(f"Creating projection layer: {self.clip_embedding_dim} -> {self.embedding_dim}")
            self.projection = nn.Sequential(
                nn.Linear(self.clip_embedding_dim, self.clip_embedding_dim // 2),
                nn.LayerNorm(self.clip_embedding_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.clip_embedding_dim // 2, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim)
            )
        else:
            logger.info("Projection layer not required")
            self.projection = nn.Identity()
    
    def _encode_with_transformers(self, text):
        # Tokenize and encode text using Hugging Face Transformers
        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=77, 
            return_tensors="pt"
        ).to(self.clip_model.device)
        
        # Get text embeddings
        with torch.no_grad() if self.freeze else torch.enable_grad():
            outputs = self.clip_model.text_model(**inputs)
            text_embeds = outputs.last_hidden_state.mean(dim=1)  # average over tokens
            
        return text_embeds
    
    def _encode_with_openai_clip(self, text):
        # Encode text using OpenAI CLIP
        with torch.no_grad() if self.freeze else torch.enable_grad():
            tokens = clip.tokenize(text).to(self.clip_model.device)
            text_features = self.clip_model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        return text_features
    
    def forward(self, text):
        """
        Encodes text into fixed-dimension embeddings.
        
        Args:
            text (list): List of text descriptions
            
        Returns:
            torch.Tensor: Tensor with text embeddings
        """
        # Check which type of CLIP model is used
        try:
            if hasattr(self, 'processor'):
                text_embeds = self._encode_with_transformers(text)
            else:
                text_embeds = self._encode_with_openai_clip(text)
                
            # Apply projection layer
            projected_embeds = self.projection(text_embeds)
            
            return projected_embeds
            
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            # Return random embedding in case of error
            batch_size = len(text) if isinstance(text, list) else 1
            return torch.randn(batch_size, self.embedding_dim).to(text_embeds.device)
    
    @property
    def device(self):
        """Returns the device on which the model is placed"""
        return next(self.parameters()).device


class CLIPEncoder(nn.Module):
    """
    Base text encoder using CLIP (for compatibility).
    """
    
    def __init__(self, config):
        super(CLIPEncoder, self).__init__()
        
        self.embedding_dim = config.embedding_dim
        self.pretrained = config.pretrained
        self.freeze = config.freeze
        
        # Load CLIP model
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device='cpu')
        
        # Extract CLIP embedding dimension
        self.clip_embedding_dim = self.clip_model.ln_final.weight.shape[0]
        
        # Freeze CLIP parameters if required
        if self.freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Linear layer to adapt CLIP embeddings
        if self.clip_embedding_dim != self.embedding_dim:
            self.linear = nn.Linear(self.clip_embedding_dim, self.embedding_dim)
        else:
            self.linear = nn.Identity()
    
    def forward(self, text):
        """
        Encodes text into embeddings.
        
        Args:
            text (list): List of text descriptions
            
        Returns:
            torch.Tensor: Tensor with text embeddings
        """
        # Tokenize and encode text
        tokens = clip.tokenize(text).to(self.clip_model.device)
        
        with torch.no_grad() if self.freeze else torch.enable_grad():
            text_features = self.clip_model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Adapt to target dimension
        embeddings = self.linear(text_features)
        
        return embeddings
    
    @property
    def device(self):
        """Returns the device on which the model is placed"""
        return next(self.parameters()).device 