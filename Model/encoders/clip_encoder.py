import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, CLIPTextModel, CLIPVisionModel
import numpy as np
from typing import Dict, Optional
from gym import spaces

from src.common.param import args


class CLIPVisionEncoder(nn.Module):
    """CLIP Vision Encoder for encoding RGB images with spatial features for CMA"""
    
    def __init__(self, observation_space: spaces.Dict, device: torch.device = torch.device("cpu"),
                 model_name: str = "openai/clip-vit-base-patch32", freeze_backbone: bool = False,
                 spatial_output: bool = True, output_size: int = 256):
        super().__init__()
        
        self.device = device
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.spatial_output = spatial_output
        self._output_size = output_size
        
        # Initialize CLIP vision model
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        
        if freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        
        # Get the feature dimension from CLIP vision model
        self.clip_feature_dim = self.vision_model.config.hidden_size  # 768 for base, 1024 for large
        
        # Calculate expected input shape
        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
            self.rgb_image_shape = observation_space.spaces["rgb"].shape
        else:
            self._n_input_rgb = 3
            self.rgb_image_shape = (224, 224, 3)  # Default CLIP input size
        
        # Preprocessing layers to match CLIP expected input format
        self.preprocess = nn.Sequential(
            nn.AdaptiveAvgPool2d((224, 224)),  # Resize to CLIP expected size
        )
        
        if self.spatial_output:
            # For CMA policy, we need spatial features
            # We'll use the patch embeddings from CLIP and reshape them to spatial format
            # CLIP ViT-Base has 14x14 = 196 patches (excluding CLS token)
            self.spatial_height = 4  # Match ResNet spatial output
            self.spatial_width = 4   
            
            # Project CLIP features to spatial format
            self.spatial_projection = nn.Sequential(
                nn.Linear(self.clip_feature_dim, self._output_size * self.spatial_height * self.spatial_width),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
            # Add spatial embeddings like in ResNet encoder
            self.spatial_embeddings = nn.Embedding(self.spatial_height * self.spatial_width, 64)
            
            # Output shape for CMA compatibility
            self._output_shape = (
                self._output_size + self.spatial_embeddings.embedding_dim,
                self.spatial_height,
                self.spatial_width,
            )
        else:
            # For simple vector output
            self.fc_projection = nn.Sequential(
                nn.Linear(self.clip_feature_dim, self._output_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self._output_shape = (self._output_size,)
        
        # Add layer_extract compatibility for collect_data hooks
        # Create an identity layer that can be hooked to extract CLIP features
        self.layer_extract = nn.Identity()

    @property
    def is_blind(self):
        return self._n_input_rgb == 0

    @property
    def output_size(self):
        return self._output_size

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            observations: Dict containing 'rgb' key with tensor of shape [batch_size, H, W, C]
        
        Returns:
            features: Tensor of shape matching output_shape
        """
        if self.is_blind:
            if self.spatial_output:
                return torch.zeros(
                    observations["rgb"].shape[0], *self._output_shape,
                    device=self.device, dtype=torch.float32
                )
            else:
                return torch.zeros(
                    observations["rgb"].shape[0], self._output_size,
                    device=self.device, dtype=torch.float32
                )
        
        rgb_observations = observations["rgb"]
        batch_size = rgb_observations.shape[0]
        
        # Convert from [batch_size, H, W, C] to [batch_size, C, H, W]
        if rgb_observations.dim() == 4:
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
        
        # Ensure values are in [0, 1] range and convert to float
        if rgb_observations.dtype == torch.uint8:
            rgb_observations = rgb_observations.float() / 255.0
        
        # Preprocess to CLIP expected input size
        rgb_observations = self.preprocess(rgb_observations)
        
        # CLIP vision model expects [batch_size, 3, 224, 224] and values in [0, 1]
        with torch.set_grad_enabled(not self.freeze_backbone):
            vision_outputs = self.vision_model(pixel_values=rgb_observations)
        
        # Use pooled output (CLS token representation)
        clip_features = vision_outputs.pooler_output  # [batch_size, clip_feature_dim]
        
        # 打印CLIP RGB特征信息到命令行
        print(f"[CLIP RGB] Features shape: {clip_features.shape}, mean: {clip_features.mean():.4f}, std: {clip_features.std():.4f}, range: [{clip_features.min():.4f}, {clip_features.max():.4f}]")
        
        # Apply layer_extract hook point (for collect_data compatibility)
        hooked_features = self.layer_extract(clip_features)
        
        if self.spatial_output:
            # Project to spatial features
            spatial_features = self.spatial_projection(hooked_features)  # [batch_size, output_size * H * W]
            spatial_features = spatial_features.view(
                batch_size, self._output_size, self.spatial_height, self.spatial_width
            )
            
            # Add spatial embeddings
            spatial_positions = torch.arange(
                self.spatial_height * self.spatial_width, 
                device=self.device
            ).unsqueeze(0).expand(batch_size, -1)
            
            spatial_embed = self.spatial_embeddings(spatial_positions)  # [batch_size, H*W, embed_dim]
            spatial_embed = spatial_embed.view(
                batch_size, self.spatial_height, self.spatial_width, self.spatial_embeddings.embedding_dim
            ).permute(0, 3, 1, 2)  # [batch_size, embed_dim, H, W]
            
            # Concatenate along channel dimension
            features = torch.cat([spatial_features, spatial_embed], dim=1)
            
            return features
        else:
            # Simple vector output
            features = self.fc_projection(hooked_features)
            return features


class CLIPDepthEncoder(nn.Module):
    """CLIP Vision Encoder for encoding depth images with spatial features"""
    
    def __init__(self, observation_space: spaces.Dict, device: torch.device = torch.device("cpu"),
                 model_name: str = "openai/clip-vit-base-patch32", freeze_backbone: bool = False,
                 spatial_output: bool = True, output_size: int = 128):
        super().__init__()
        
        self.device = device
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.spatial_output = spatial_output
        self._output_size = output_size
        
        # Initialize CLIP vision model
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        
        if freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        
        # Get the feature dimension from CLIP vision model
        self.clip_feature_dim = self.vision_model.config.hidden_size  # 768 for base, 1024 for large
        
        # Calculate expected input shape
        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
            self.depth_image_shape = observation_space.spaces["depth"].shape
        else:
            self._n_input_depth = 1
            self.depth_image_shape = (256, 256, 1)  # Default depth input size
        
        # Preprocessing layers to convert depth to RGB-like format for CLIP
        self.depth_preprocess = nn.Sequential(
            nn.AdaptiveAvgPool2d((224, 224)),  # Resize to CLIP expected size
        )
        
        # Convert single channel depth to 3-channel RGB-like input
        self.depth_to_rgb = nn.Conv2d(1, 3, kernel_size=1, padding=0)
        
        if self.spatial_output:
            # For CMA policy, we need spatial features
            # Match the original depth encoder spatial dimensions
            self.spatial_height = 8  # Original depth encoder spatial output
            self.spatial_width = 8   
            
            # Project CLIP features to spatial format
            self.spatial_projection = nn.Sequential(
                nn.Linear(self.clip_feature_dim, self._output_size * self.spatial_height * self.spatial_width),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
            # Add spatial embeddings like in original depth encoder
            self.spatial_embeddings = nn.Embedding(self.spatial_height * self.spatial_width, 64)
            
            # Output shape for CMA compatibility
            self._output_shape = (
                self._output_size + self.spatial_embeddings.embedding_dim,
                self.spatial_height,
                self.spatial_width,
            )
        else:
            # For simple vector output
            self.fc_projection = nn.Sequential(
                nn.Linear(self.clip_feature_dim, self._output_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self._output_shape = (self._output_size,)
        
        # Add layer_extract compatibility for collect_data hooks
        # Create an identity layer that can be hooked to extract CLIP features
        self.layer_extract = nn.Identity()

    @property
    def is_blind(self):
        return self._n_input_depth == 0

    @property
    def output_size(self):
        return self._output_size

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            observations: Dict containing 'depth' key with tensor of shape [batch_size, H, W, C]
        
        Returns:
            features: Tensor of shape matching output_shape
        """
        if self.is_blind:
            if self.spatial_output:
                return torch.zeros(
                    observations["depth"].shape[0], *self._output_shape,
                    device=self.device, dtype=torch.float32
                )
            else:
                return torch.zeros(
                    observations["depth"].shape[0], self._output_size,
                    device=self.device, dtype=torch.float32
                )
        
        depth_observations = observations["depth"]
        batch_size = depth_observations.shape[0]
        
        # Convert from [batch_size, H, W, C] to [batch_size, C, H, W]
        if depth_observations.dim() == 4:
            depth_observations = depth_observations.permute(0, 3, 1, 2)
        
        # Ensure depth values are normalized [0, 1]
        if depth_observations.max() > 1.0:
            depth_observations = depth_observations / depth_observations.max()
        
        # Convert single channel depth to 3-channel for CLIP
        depth_rgb = self.depth_to_rgb(depth_observations)
        
        # Preprocess to CLIP expected input size
        depth_rgb = self.depth_preprocess(depth_rgb)
        
        # Ensure values are in [0, 1] range
        depth_rgb = torch.clamp(depth_rgb, 0, 1)
        
        # CLIP vision model expects [batch_size, 3, 224, 224] and values in [0, 1]
        with torch.set_grad_enabled(not self.freeze_backbone):
            vision_outputs = self.vision_model(pixel_values=depth_rgb)
        
        # Use pooled output (CLS token representation)
        clip_features = vision_outputs.pooler_output  # [batch_size, clip_feature_dim]
        
        # 打印CLIP Depth特征信息到命令行
        print(f"[CLIP DEPTH] Features shape: {clip_features.shape}, mean: {clip_features.mean():.4f}, std: {clip_features.std():.4f}, range: [{clip_features.min():.4f}, {clip_features.max():.4f}]")
        
        # Apply layer_extract hook point (for collect_data compatibility)
        hooked_features = self.layer_extract(clip_features)
        
        if self.spatial_output:
            # Project to spatial features
            spatial_features = self.spatial_projection(hooked_features)  # [batch_size, output_size * H * W]
            spatial_features = spatial_features.view(
                batch_size, self._output_size, self.spatial_height, self.spatial_width
            )
            
            # Add spatial embeddings
            spatial_positions = torch.arange(
                self.spatial_height * self.spatial_width, 
                device=self.device
            ).unsqueeze(0).expand(batch_size, -1)
            
            spatial_embed = self.spatial_embeddings(spatial_positions)  # [batch_size, H*W, embed_dim]
            spatial_embed = spatial_embed.view(
                batch_size, self.spatial_height, self.spatial_width, self.spatial_embeddings.embedding_dim
            ).permute(0, 3, 1, 2)  # [batch_size, embed_dim, H, W]
            
            # Concatenate along channel dimension
            features = torch.cat([spatial_features, spatial_embed], dim=1)
            
            return features
        else:
            # Simple vector output
            features = self.fc_projection(hooked_features)
            return features


class CLIPTextEncoder(nn.Module):
    """CLIP Text Encoder for encoding instruction text"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", 
                 freeze_backbone: bool = False, max_length: int = 77):
        super().__init__()
        
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.max_length = max_length
        
        # Initialize CLIP text model and processor
        self.text_model = CLIPTextModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        if freeze_backbone:
            for param in self.text_model.parameters():
                param.requires_grad = False
        
        # Get the output dimension from CLIP text model
        self._output_size = self.text_model.config.hidden_size  # 512 for base, 768 for large
        
        # For compatibility with existing CMA code that expects sequence output
        self.final_state_only = False  # Return sequence for attention

    @property
    def output_size(self):
        return self._output_size

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            observations: Dict containing 'instruction' key with token indices
        
        Returns:
            features: Tensor of shape [batch_size, seq_len, output_size] for attention
                     or [batch_size, output_size] if final_state_only is True
        """
        # Get instruction tokens
        instruction_tokens = observations["instruction"]
        batch_size = instruction_tokens.shape[0]
        
        # CLIP models have a maximum sequence length (usually 77)
        max_seq_length = 77  # Standard CLIP max length
        
        # Truncate or pad to CLIP's expected length
        if instruction_tokens.shape[1] > max_seq_length:
            # Truncate to max length
            instruction_tokens = instruction_tokens[:, :max_seq_length]
            print(f"[CLIP TEXT] Truncated instruction from {observations['instruction'].shape[1]} to {max_seq_length} tokens")
        elif instruction_tokens.shape[1] < max_seq_length:
            # Pad with zeros to max length
            padded_tokens = torch.zeros(batch_size, max_seq_length, dtype=instruction_tokens.dtype, device=instruction_tokens.device)
            padded_tokens[:, :instruction_tokens.shape[1]] = instruction_tokens
            instruction_tokens = padded_tokens
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (instruction_tokens != 0).long()
        
        # Process through CLIP text model
        with torch.set_grad_enabled(not self.freeze_backbone):
            text_outputs = self.text_model(
                input_ids=instruction_tokens,
                attention_mask=attention_mask
            )
        
        if self.final_state_only:
            # Return pooled output for final state only
            return text_outputs.pooler_output
        else:
            # Return sequence outputs for attention mechanism
            # Shape: [batch_size, seq_len, hidden_size]
            sequence_output = text_outputs.last_hidden_state
            
            # Apply padding mask to sequence output (set padded positions to 0)
            # This is important for compatibility with CMA attention mechanism
            expanded_attention_mask = attention_mask.unsqueeze(-1).expand_as(sequence_output)
            sequence_output = sequence_output * expanded_attention_mask.float()
            
            # Transpose to match expected format [batch_size, hidden_size, seq_len]
            return sequence_output.transpose(1, 2)


class CLIPInstructionEncoder(nn.Module):
    """
    CLIP-based instruction encoder that can work as a drop-in replacement
    for the original InstructionEncoder
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 freeze_backbone: bool = False, final_state_only: bool = False):
        super().__init__()
        
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        
        # Initialize CLIP text model and processor
        self.text_model = CLIPTextModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        if freeze_backbone:
            for param in self.text_model.parameters():
                param.requires_grad = False
        
        # Configuration for compatibility
        self.config = type('Config', (), {})()
        self.config.final_state_only = final_state_only
        
        # Get the CLIP output dimension
        self.clip_output_size = self.text_model.config.hidden_size
        
        # For CMA compatibility, we need to match the original instruction encoder output size
        # Original: hidden_size * (1 + bidirectional) = 128 * 2 = 256
        self.target_output_size = 256  # Match original instruction encoder
        
        # Projection layer to match original encoder dimensions
        self.projection = nn.Linear(self.clip_output_size, self.target_output_size)
        
        # Set output size for compatibility
        self._output_size = self.target_output_size

    @property
    def output_size(self):
        return self._output_size

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            observations: Dict containing 'instruction' key with token indices
        
        Returns:
            features: Tensor shape depends on final_state_only:
                     - If True: [batch_size, output_size] 
                     - If False: [batch_size, output_size, seq_len] for attention
        """
        # Get instruction tokens
        instruction_tokens = observations["instruction"]
        
        # CLIP models have a maximum sequence length (usually 77)
        max_seq_length = 77  # Standard CLIP max length
        
        # Truncate or pad to CLIP's expected length
        if instruction_tokens.shape[1] > max_seq_length:
            # Truncate to max length
            instruction_tokens = instruction_tokens[:, :max_seq_length]
            print(f"[CLIP TEXT] Truncated instruction from {observations['instruction'].shape[1]} to {max_seq_length} tokens")
        elif instruction_tokens.shape[1] < max_seq_length:
            # Pad with zeros to max length
            batch_size = instruction_tokens.shape[0]
            padded_tokens = torch.zeros(batch_size, max_seq_length, dtype=instruction_tokens.dtype, device=instruction_tokens.device)
            padded_tokens[:, :instruction_tokens.shape[1]] = instruction_tokens
            instruction_tokens = padded_tokens
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (instruction_tokens != 0).long()
        
        # Process through CLIP text model
        with torch.set_grad_enabled(not self.freeze_backbone):
            text_outputs = self.text_model(
                input_ids=instruction_tokens,
                attention_mask=attention_mask
            )
        
        if self.config.final_state_only:
            # Return pooled output [batch_size, output_size]
            clip_text_features = text_outputs.pooler_output
            # Apply projection to match original encoder dimensions
            clip_text_features = self.projection(clip_text_features)
            # 打印CLIP指令特征信息到命令行
            print(f"[CLIP TEXT] Features shape: {clip_text_features.shape}, mean: {clip_text_features.mean():.4f}, std: {clip_text_features.std():.4f}, range: [{clip_text_features.min():.4f}, {clip_text_features.max():.4f}]")
            return clip_text_features
        else:
            # Return sequence outputs for attention mechanism
            # Shape: [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size, seq_len]
            sequence_output = text_outputs.last_hidden_state
            
            # Apply projection to match original encoder dimensions
            # sequence_output shape: [batch_size, seq_len, clip_hidden_size]
            sequence_output = self.projection(sequence_output)  # -> [batch_size, seq_len, target_hidden_size]
            
            # Apply padding mask to sequence output (set padded positions to 0)
            # This is important for compatibility with CMA attention mechanism
            expanded_attention_mask = attention_mask.unsqueeze(-1).expand_as(sequence_output)
            sequence_output = sequence_output * expanded_attention_mask.float()
            
            # 打印CLIP指令序列特征信息到命令行
            print(f"[CLIP TEXT SEQ] Features shape: {sequence_output.shape}, mean: {sequence_output.mean():.4f}, std: {sequence_output.std():.4f}, range: [{sequence_output.min():.4f}, {sequence_output.max():.4f}]")
            return sequence_output.transpose(1, 2)