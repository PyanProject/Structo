import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

class ImprovedVoxelTransformer(nn.Module):
    """
    Improved 3D model generator using transformer architecture
    with conditional attention and multi-level decoder for creating voxel grids
    based on text embeddings.
    """
    
    def __init__(self, config):
        super(ImprovedVoxelTransformer, self).__init__()
        
        # Extract parameters from configuration
        self.latent_dim = config.latent_dim
        self.hidden_dims = config.hidden_dims
        self.dropout_rate = config.dropout
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.voxel_dim = config.voxel_dim
        
        # Additional parameters
        self.use_conditional_attention = getattr(config, 'use_conditional_attention', True)
        self.residual_connections = getattr(config, 'residual_connections', True)
        self.normalization = getattr(config, 'normalization', 'layer_norm')
        self.activation_fn = getattr(config, 'activation', 'gelu')
        
        logger.info(f"Initializing ImprovedVoxelTransformer with parameters:")
        logger.info(f"  latent_dim: {self.latent_dim}")
        logger.info(f"  hidden_dims: {self.hidden_dims}")
        logger.info(f"  num_heads: {self.num_heads}")
        logger.info(f"  num_layers: {self.num_layers}")
        logger.info(f"  voxel_dim: {self.voxel_dim}")
        logger.info(f"  use_conditional_attention: {self.use_conditional_attention}")
        logger.info(f"  residual_connections: {self.residual_connections}")
        logger.info(f"  normalization: {self.normalization}")
        logger.info(f"  activation: {self.activation_fn}")
        
        # Choose activation function
        if self.activation_fn == 'gelu':
            self.activation = nn.GELU()
        elif self.activation_fn == 'relu':
            self.activation = nn.ReLU()
        elif self.activation_fn == 'swish':
            self.activation = nn.SiLU()
        else:
            logger.warning(f"Unknown activation function: {self.activation_fn}, using GELU instead")
            self.activation = nn.GELU()
        
        # Normalization function
        if self.normalization == 'layer_norm':
            self.norm_fn = lambda dim: nn.LayerNorm(dim)
        elif self.normalization == 'batch_norm':
            self.norm_fn = lambda dim: nn.BatchNorm1d(dim)
        else:
            logger.warning(f"Unknown normalization: {self.normalization}, using LayerNorm instead")
            self.norm_fn = lambda dim: nn.LayerNorm(dim)
        
        # Projection layer for text embedding
        self.text_projection = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            self.norm_fn(self.latent_dim),
            self.activation,
            nn.Dropout(self.dropout_rate)
        )
        
        # Transformer encoders
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoderBlock(
                dim=self.latent_dim,
                num_heads=self.num_heads,
                dropout=self.dropout_rate,
                use_conditional_attention=self.use_conditional_attention,
                residual=self.residual_connections,
                norm_fn=self.norm_fn,
                activation=self.activation
            ) for _ in range(self.num_layers)
        ])
        
        # Create a list of modules for hierarchical 3D decoder
        self.decoders = nn.ModuleList()
        
        # First decoder layer, transforming embedding into initial 3D representation
        first_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dims[0] * 2 * 2 * 2),
            self.norm_fn(self.hidden_dims[0] * 2 * 2 * 2),
            self.activation,
            nn.Dropout(self.dropout_rate)
        )
        self.decoders.append(first_decoder)
        
        # Building a multi-stage decoder
        # Each level increases resolution and decreases number of channels
        current_res = 2
        for i in range(len(self.hidden_dims) - 1):
            # Check if we've reached the target resolution
            if current_res * 2 > self.voxel_dim:
                break
                
            input_channels = self.hidden_dims[i]
            output_channels = self.hidden_dims[i + 1]
            
            # Create resolution upscale block
            upscale_block = UpsamplingBlock(
                in_channels=input_channels,
                out_channels=output_channels,
                scale_factor=2,
                dropout=self.dropout_rate,
                use_residual=self.residual_connections,
                activation=self.activation
            )
            
            self.decoders.append(upscale_block)
            current_res *= 2
            
        # If target resolution not reached, add additional layers
        while current_res < self.voxel_dim:
            last_channels = self.hidden_dims[-1]
            
            # Gradually reduce number of channels
            if last_channels > 16:
                out_channels = last_channels // 2
            else:
                out_channels = last_channels
                
            # Add another upscaling block
            upscale_block = UpsamplingBlock(
                in_channels=last_channels,
                out_channels=out_channels,
                scale_factor=2,
                dropout=self.dropout_rate,
                use_residual=self.residual_connections,
                activation=self.activation
            )
            
            self.decoders.append(upscale_block)
            current_res *= 2
            self.hidden_dims.append(out_channels)
        
        # Final convolutional layer for creating single-channel output
        self.final_conv = nn.Conv3d(
            in_channels=self.hidden_dims[-1],
            out_channels=1,  # One channel for binary voxel grid
            kernel_size=3,
            padding=1
        )
        
        logger.info(f"Decoder built with {len(self.decoders)} levels")
        logger.info(f"Final resolution: {current_res}x{current_res}x{current_res}")
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm3d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, text_embedding):
        """
        Generates a voxel grid based on text embedding.
        
        Args:
            text_embedding (torch.Tensor): Text embedding [batch_size, latent_dim]
            
        Returns:
            torch.Tensor: Voxel grid [batch_size, 1, voxel_dim, voxel_dim, voxel_dim]
        """
        batch_size = text_embedding.shape[0]
        
        # Project text embedding
        x = self.text_projection(text_embedding)
        
        # Apply transformer layers
        for transformer in self.transformer_encoders:
            x = transformer(x, context=text_embedding)
        
        # Prepare input for decoder
        x = self.decoders[0](x)
        
        # Transform to 3D format
        x = x.view(batch_size, self.hidden_dims[0], 2, 2, 2)
        
        # Pass through all decoder levels
        for i in range(1, len(self.decoders)):
            x = self.decoders[i](x)
        
        # Final convolution to get voxel grid
        voxel_grid = self.final_conv(x)
        
        return voxel_grid


class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block with conditional attention support.
    """
    
    def __init__(self, dim, num_heads, dropout=0.1, use_conditional_attention=True, 
                 residual=True, norm_fn=nn.LayerNorm, activation=None):
        super(TransformerEncoderBlock, self).__init__()
        
        # If no activation is provided, create GELU by default
        if activation is None:
            activation = nn.GELU()
        
        self.use_conditional_attention = use_conditional_attention
        self.residual = residual
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Conditional attention (if enabled)
        if use_conditional_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.norm2 = norm_fn(dim)
        
        # Normalization and feedforward
        self.norm1 = norm_fn(dim)
        self.norm3 = norm_fn(dim)
        
        # Fully connected layer
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            activation,  # Using instance directly
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, context=None):
        """
        Forward pass through transformer encoder block.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, dim]
            context (torch.Tensor, optional): Context tensor for conditional attention
                
        Returns:
            torch.Tensor: Output tensor after processing
        """
        # Self-attention
        residual = x
        x = self.norm1(x)
        
        # For 1D inputs (only batch_size, dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            if self.residual:
                residual = residual.unsqueeze(1)
            
        # Apply self-attention
        x_attn, _ = self.attention(x, x, x)
        
        # Apply residual connection
        if self.residual:
            x = residual + x_attn
        else:
            x = x_attn
        
        # Conditional attention (if enabled)
        if self.use_conditional_attention and context is not None:
            residual = x
            x = self.norm2(x)
            
            # Prepare context
            if len(context.shape) == 2:
                context = context.unsqueeze(1)
                
            # Apply cross-attention
            x_cross, _ = self.cross_attention(x, context, context)
            
            # Apply residual connection
            if self.residual:
                x = residual + x_cross
            else:
                x = x_cross
        
        # MLP block
        residual = x
        x = self.norm3(x)
        x = self.mlp(x)
        
        # Apply residual connection
        if self.residual:
            x = residual + x
            
        # For 1D outputs
        if len(x.shape) == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
            
        return x


class UpsamplingBlock(nn.Module):
    """
    Block for increasing the resolution of 3D object representation.
    """
    
    def __init__(self, in_channels, out_channels, scale_factor=2, dropout=0.1, 
                 use_residual=True, activation=None):
        super(UpsamplingBlock, self).__init__()
        
        # If no activation is provided, create GELU by default
        if activation is None:
            activation = nn.GELU()
        
        self.use_residual = use_residual
        
        # Increase resolution using interpolation
        self.upsample = nn.Upsample(
            scale_factor=scale_factor, 
            mode='trilinear', 
            align_corners=False
        )
        
        # First convolution
        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1
        )
        
        # Normalization and activation after first convolution
        self.norm1 = nn.BatchNorm3d(in_channels)
        self.activation1 = activation  # Using instance directly
        self.dropout1 = nn.Dropout3d(dropout)
        
        # Second convolution to change number of channels
        self.conv2 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        
        # Normalization and activation after second convolution
        self.norm2 = nn.BatchNorm3d(out_channels)
        self.activation2 = activation  # Using instance directly
        self.dropout2 = nn.Dropout3d(dropout)
        
        # Residual connection (if dimensions differ)
        if in_channels != out_channels and use_residual:
            self.shortcut = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1
            )
        else:
            self.shortcut = None
            
    def forward(self, x):
        """
        Forward pass through the upsampling block.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, in_channels, D, H, W]
                
        Returns:
            torch.Tensor: Output tensor with increased resolution 
                         [batch_size, out_channels, D*2, H*2, W*2]
        """
        # Save input for residual connection
        residual = x
        
        # Increase resolution
        x = self.upsample(x)
        
        # First convolution with normalization and activation
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        
        # Second convolution with normalization and activation
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        
        # Apply residual connection (if enabled)
        if self.use_residual:
            if self.shortcut is not None:
                residual = self.shortcut(self.upsample(residual))
            else:
                residual = self.upsample(residual)
                
            # Check dimension compatibility
            if residual.shape == x.shape:
                x = x + residual
                
        return x
        

class VoxelTransformer(nn.Module):
    """
    Base 3D model generator using transformers (for compatibility).
    """
    
    def __init__(self, config):
        super(VoxelTransformer, self).__init__()
        
        # Extract parameters from configuration
        self.latent_dim = config.latent_dim
        self.hidden_dims = config.hidden_dims
        self.dropout = config.dropout
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.voxel_dim = config.voxel_dim
        
        # Projection layer for text embedding
        self.text_projection = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.latent_dim * 4,
                dropout=self.dropout,
                batch_first=True
            ) for _ in range(self.num_layers)
        ])
        
        # Transform embedding into 3D volume
        self.fc = nn.Linear(self.latent_dim, self.hidden_dims[0] * 4 * 4 * 4)
        
        # Create decoder for generating 3D volume
        self.decoders = nn.ModuleList()
        
        # Sequentially increase resolution
        input_dim = 4
        for i in range(len(self.hidden_dims) - 1):
            # If target resolution reached, stop
            if input_dim >= self.voxel_dim:
                break
                
            # Upsampling layer
            self.decoders.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        self.hidden_dims[i],
                        self.hidden_dims[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1
                    ),
                    nn.BatchNorm3d(self.hidden_dims[i + 1]),
                    nn.GELU()
                )
            )
            
            input_dim *= 2
        
        # Final layer for single-channel output
        self.final_conv = nn.Conv3d(
            in_channels=self.hidden_dims[-1],
            out_channels=1,
            kernel_size=3,
            padding=1
        )
        
    def forward(self, text_embedding):
        """
        Generates a voxel grid based on text embedding.
        
        Args:
            text_embedding (torch.Tensor): Text embedding [batch_size, latent_dim]
            
        Returns:
            torch.Tensor: Voxel grid [batch_size, 1, voxel_dim, voxel_dim, voxel_dim]
        """
        batch_size = text_embedding.shape[0]
        
        # Project text embedding
        x = self.text_projection(text_embedding)
        
        # Add dummy sequence dimension for transformers
        x = x.unsqueeze(1)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Remove sequence dimension
        x = x.squeeze(1)
        
        # Transform to 3D format
        x = self.fc(x)
        x = x.view(batch_size, self.hidden_dims[0], 4, 4, 4)
        
        # Pass through decoder
        for decoder in self.decoders:
            x = decoder(x)
        
        # Check dimensions and resize if needed
        if x.shape[2] != self.voxel_dim:
            x = F.interpolate(
                x,
                size=(self.voxel_dim, self.voxel_dim, self.voxel_dim),
                mode='trilinear',
                align_corners=False
            )
        
        # Final convolution
        voxel_grid = self.final_conv(x)
        
        return voxel_grid 