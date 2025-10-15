"""
Refactored BLIP-2 Encoders for Vision-Language Navigation

This module provides unified BLIP-2 based encoders for:
- RGB images (224×224×3) 
- Depth images (256×256×1)
- Text instructions (max 512 tokens)

Key improvements:
- Shared model instance for memory efficiency
- Proper data type and device management  
- Clean separation of concerns
- Unified interface for all modalities
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from PIL import Image
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    AutoTokenizer,
    AutoImageProcessor,
)

try:
    from transformers import Blip2ImageProcessor
except ImportError:
    Blip2ImageProcessor = None

from src.common.param import args

logger = logging.getLogger(__name__)


class BLIP2SharedComponents:
    """Shared BLIP-2 model components to avoid memory duplication."""
    
    _instance = None
    _model = None
    _processor = None
    _device = None
    
    def __new__(cls, model_name: str = "Salesforce/blip2-opt-2.7b", device: torch.device = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", device: torch.device = None):
        if hasattr(self, '_initialized'):
            return
            
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load shared components
        self._load_model()
        self._load_processor()
        
        self._initialized = True
        logger.info(f"BLIP-2 shared components initialized on {self.device}")
    
    def _resolve_pretrained_kwargs(self) -> Dict:
        """Resolve kwargs for from_pretrained calls."""
        kwargs = {}
        is_local_path = os.path.isdir(self.model_name) or os.path.isfile(self.model_name)
        if is_local_path or os.environ.get("TRANSFORMERS_OFFLINE") == "1":
            kwargs["local_files_only"] = True
        return kwargs
    
    def _load_model(self):
        """Load BLIP-2 model once for all encoders."""
        if self._model is None:
            pretrained_kwargs = self._resolve_pretrained_kwargs()
            self._model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16,
                **pretrained_kwargs
            )
            self._model.to(self.device)
            logger.info(f"Loaded BLIP-2 model: {self.model_name}")
    
    def _load_processor(self):
        """Load BLIP-2 processor with fallback handling."""
        if self._processor is None:
            pretrained_kwargs = self._resolve_pretrained_kwargs()
            try:
                self._processor = Blip2Processor.from_pretrained(self.model_name, **pretrained_kwargs)
            except Exception as e:
                logger.warning(f"Failed to load Blip2Processor directly: {e}")
                # Manual construction fallback
                image_processor = self._load_image_processor(pretrained_kwargs)
                tokenizer = self._load_tokenizer(pretrained_kwargs)
                self._processor = Blip2Processor(tokenizer=tokenizer, image_processor=image_processor)
            logger.info("Loaded BLIP-2 processor")
    
    def _load_image_processor(self, pretrained_kwargs):
        """Load image processor with fallback."""
        if Blip2ImageProcessor is not None:
            try:
                return Blip2ImageProcessor.from_pretrained(self.model_name, **pretrained_kwargs)
            except Exception:
                pass
        return AutoImageProcessor.from_pretrained(self.model_name, **pretrained_kwargs)
    
    def _load_tokenizer(self, pretrained_kwargs):
        """Load tokenizer with proper configuration."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=False, **pretrained_kwargs
        )
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer
    
    @property 
    def model(self):
        return self._model
        
    @property
    def processor(self):
        return self._processor


class BaseBLIP2Encoder(nn.Module):
    """Base class for BLIP-2 encoders with shared components."""
    
    def __init__(self, 
                 model_name: str = "Salesforce/blip2-opt-2.7b",
                 freeze_backbone: bool = False,
                 output_size: int = 256,
                 device: torch.device = None):
        super().__init__()
        
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self._output_size = output_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get shared components
        self.shared = BLIP2SharedComponents(model_name, self.device)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.shared.model.parameters():
                param.requires_grad = False
        
        logger.info(f"Initialized BLIP-2 encoder with output_size={output_size}")
    
    @property
    def output_size(self):
        return self._output_size


class BLIP2VisionEncoder(BaseBLIP2Encoder):
    """BLIP-2 Vision Encoder for RGB images (224×224×3)."""
    
    def __init__(self, 
                 observation_space: spaces.Dict,
                 device: torch.device = torch.device("cpu"),
                 model_name: str = "Salesforce/blip2-opt-2.7b",
                 freeze_backbone: bool = False,
                 spatial_output: bool = True,
                 output_size: int = 256):
        
        super().__init__(model_name, freeze_backbone, output_size, device)
        
        self.spatial_output = spatial_output
        
        # Validate RGB observation space
        rgb_shape = observation_space.spaces["rgb"].shape
        if rgb_shape != (224, 224, 3):
            logger.warning(f"RGB shape {rgb_shape} != expected (224, 224, 3)")
        
        # Vision feature extraction setup
        self.vision_hidden_size = 1408  # BLIP-2 vision encoder hidden size
        self.patch_size = 14  # BLIP-2 patch size  
        self.num_patches = (224 // self.patch_size) ** 2  # 16x16 = 256 patches
        
        # Projection layer
        self.projection = nn.Linear(self.vision_hidden_size, output_size)
        self.projection.to(device=self.device, dtype=torch.float16)
        
        # Hook for feature extraction
        self.hook_output = None
        self.hook = self._setup_vision_hook()
    
    def _setup_vision_hook(self):
        """Setup hook to extract vision features from last layer."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.hook_output = output[0]  # hidden_states
            elif hasattr(output, 'last_hidden_state'):
                self.hook_output = output.last_hidden_state  
            else:
                self.hook_output = output
        
        # Hook to the last vision encoder layer
        target_layer = self.shared.model.vision_model.encoder.layers[-1]
        return target_layer.register_forward_hook(hook_fn)
    
    @property
    def is_blind(self):
        return False
    
    @property  
    def output_shape(self):
        if self.spatial_output:
            spatial_dim = int(self.num_patches ** 0.5)  # 16
            return (self.output_size, spatial_dim, spatial_dim)
        else:
            return (self.output_size,)
    
    def _preprocess_rgb_batch(self, rgb_batch: torch.Tensor) -> torch.Tensor:
        """Preprocess RGB batch for BLIP-2."""
        batch_size = rgb_batch.shape[0]
        
        # Ensure correct dtype and range
        if rgb_batch.dtype != torch.float32:
            rgb_batch = rgb_batch.float()
        
        # Normalize to [0, 1] and convert to uint8 for PIL compatibility
        rgb_normalized = torch.clamp(rgb_batch, 0, 1)
        rgb_uint8 = (rgb_normalized * 255).byte().cpu().numpy()
        
        # Process each image in batch  
        pixel_values = []
        for i in range(batch_size):
            # Convert from [H, W, C] to PIL Image
            img_np = rgb_uint8[i]  # Shape: [224, 224, 3]
            pil_image = Image.fromarray(img_np, 'RGB')
            
            # Process with BLIP-2 processor
            processed = self.shared.processor(images=pil_image, return_tensors="pt")
            pixel_values.append(processed.pixel_values)
        
        return torch.cat(pixel_values, dim=0).to(self.device)
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for RGB images."""
        rgb_images = observations["rgb"]  # [B, H, W, C]
        
        # Preprocess images
        pixel_values = self._preprocess_rgb_batch(rgb_images)
        
        # Extract vision features
        with torch.set_grad_enabled(not self.freeze_backbone):
            _ = self.shared.model.vision_model(pixel_values)
        
        # Get features from hook
        vision_features = self.hook_output  # [B, 257, 1408] (256 patches + 1 CLS)
        
        if vision_features.dim() == 3 and vision_features.shape[1] > self.num_patches:
            # Remove CLS token, keep only patch features
            patch_features = vision_features[:, 1:, :]  # [B, 256, 1408]
        else:
            patch_features = vision_features
        
        # Project features
        projected = self.projection(patch_features)  # [B, 256, output_size]
        
        if self.spatial_output:
            # Reshape to spatial format [B, output_size, H, W]
            batch_size, num_patches, feature_dim = projected.shape
            spatial_dim = int(num_patches ** 0.5)  # 16
            
            projected = projected.view(batch_size, spatial_dim, spatial_dim, feature_dim)
            projected = projected.permute(0, 3, 1, 2)  # [B, feature_dim, H, W]
        else:
            # Global average pooling for non-spatial output
            projected = projected.mean(dim=1)  # [B, output_size]
        
        return projected.to(dtype=torch.float32)


class BLIP2DepthEncoder(BaseBLIP2Encoder):
    """BLIP-2 Depth Encoder for depth images (256×256×1)."""
    
    def __init__(self,
                 observation_space: spaces.Dict, 
                 device: torch.device = torch.device("cpu"),
                 model_name: str = "Salesforce/blip2-opt-2.7b",
                 freeze_backbone: bool = False,
                 spatial_output: bool = True,
                 output_size: int = 192):
        
        super().__init__(model_name, freeze_backbone, output_size, device)
        
        self.spatial_output = spatial_output
        
        # Validate depth observation space
        depth_shape = observation_space.spaces["depth"].shape
        if depth_shape != (256, 256, 1):
            logger.warning(f"Depth shape {depth_shape} != expected (256, 256, 1)")
        
        # Vision feature extraction setup (same as RGB)
        self.vision_hidden_size = 1408
        self.patch_size = 14
        self.num_patches = (224 // self.patch_size) ** 2  # Images resized to 224 by processor
        
        # Projection layer
        self.projection = nn.Linear(self.vision_hidden_size, output_size)
        self.projection.to(device=self.device, dtype=torch.float16)
        
        # Hook for feature extraction
        self.hook_output = None
        self.hook = self._setup_vision_hook()
    
    def _setup_vision_hook(self):
        """Setup hook to extract vision features."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.hook_output = output[0]
            elif hasattr(output, 'last_hidden_state'):
                self.hook_output = output.last_hidden_state
            else:
                self.hook_output = output
        
        target_layer = self.shared.model.vision_model.encoder.layers[-1]
        return target_layer.register_forward_hook(hook_fn)
    
    @property
    def is_blind(self):
        return False
    
    @property
    def output_shape(self):
        if self.spatial_output:
            spatial_dim = int(self.num_patches ** 0.5)  # 16
            return (self.output_size, spatial_dim, spatial_dim)
        else:
            return (self.output_size,)
    
    def _preprocess_depth_batch(self, depth_batch: torch.Tensor) -> torch.Tensor:
        """Preprocess depth batch for BLIP-2."""
        batch_size = depth_batch.shape[0]
        
        # Ensure correct shape and dtype
        if depth_batch.shape[-1] == 1:
            depth_batch = depth_batch.squeeze(-1)  # [B, H, W]
        
        if depth_batch.dtype != torch.float32:
            depth_batch = depth_batch.float()
        
        # Normalize to [0, 1] range
        depth_normalized = torch.clamp(depth_batch, 0, 1)
        
        # Convert to 3-channel RGB-like format for BLIP-2
        depth_rgb = depth_normalized.unsqueeze(-1).repeat(1, 1, 1, 3)  # [B, H, W, 3]
        
        # Convert to uint8 for PIL
        depth_uint8 = (depth_rgb * 255).byte().cpu().numpy()
        
        # Process each image in batch
        pixel_values = []
        for i in range(batch_size):
            img_np = depth_uint8[i]  # [256, 256, 3]
            pil_image = Image.fromarray(img_np, 'RGB')
            
            processed = self.shared.processor(images=pil_image, return_tensors="pt")
            pixel_values.append(processed.pixel_values)
        
        return torch.cat(pixel_values, dim=0).to(self.device)
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for depth images."""
        depth_images = observations["depth"]  # [B, H, W, 1] or [B, H, W]
        
        # Preprocess depth images
        pixel_values = self._preprocess_depth_batch(depth_images)
        
        # Extract vision features
        with torch.set_grad_enabled(not self.freeze_backbone):
            _ = self.shared.model.vision_model(pixel_values)
        
        # Get features from hook
        vision_features = self.hook_output  # [B, 257, 1408]
        
        if vision_features.dim() == 3 and vision_features.shape[1] > self.num_patches:
            # Remove CLS token, keep only patch features
            patch_features = vision_features[:, 1:, :]  # [B, 256, 1408]
        else:
            patch_features = vision_features
        
        # Project features
        projected = self.projection(patch_features)  # [B, 256, output_size]
        
        if self.spatial_output:
            # Reshape to spatial format [B, output_size, H, W]
            batch_size, num_patches, feature_dim = projected.shape
            spatial_dim = int(num_patches ** 0.5)  # 16
            
            projected = projected.view(batch_size, spatial_dim, spatial_dim, feature_dim)
            projected = projected.permute(0, 3, 1, 2)  # [B, feature_dim, H, W]
        else:
            # Global average pooling for non-spatial output
            projected = projected.mean(dim=1)  # [B, output_size]
        
        return projected.to(dtype=torch.float32)


class BLIP2InstructionEncoder(BaseBLIP2Encoder):
    """BLIP-2 Instruction Encoder for text instructions (max 512 tokens)."""
    
    def __init__(self,
                 model_name: str = "Salesforce/blip2-opt-2.7b",
                 freeze_backbone: bool = False,
                 output_size: int = 256,
                 final_state_only: bool = True,
                 max_length: int = 512,
                 device: torch.device = None):
        
        super().__init__(model_name, freeze_backbone, output_size, device)
        
        self.final_state_only = final_state_only
        self.max_length = max_length
        
        # Use language model for text encoding
        self.text_model = self.shared.model.language_model
        self.text_model.config.output_hidden_states = True
        
        # Get text hidden size
        self.text_hidden_size = self.text_model.config.hidden_size
        
        # Projection layer
        self.projection = nn.Linear(self.text_hidden_size, output_size)
        self.projection.to(device=self.device, dtype=torch.float16)
    
    @property
    def output_shape(self):
        if self.final_state_only:
            return (self.output_size,)
        else:
            return (self.output_size, self.max_length)
    
    def _preprocess_instructions(self, instructions: Union[List[str], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess instructions for BLIP-2 text model."""
        
        if isinstance(instructions, list) and len(instructions) > 0 and isinstance(instructions[0], str):
            # String instructions - tokenize them
            tokenized = self.shared.processor.tokenizer(
                instructions,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            input_ids = tokenized['input_ids'].to(self.device)
            attention_mask = tokenized['attention_mask'].to(self.device)
            
        else:
            # Pre-tokenized instructions
            instruction_tokens = instructions.to(self.device)
            batch_size = instruction_tokens.shape[0]
            
            # Ensure proper sequence length
            if instruction_tokens.shape[1] > self.max_length:
                instruction_tokens = instruction_tokens[:, :self.max_length]
            elif instruction_tokens.shape[1] < self.max_length:
                padded = torch.zeros(
                    batch_size, self.max_length,
                    dtype=instruction_tokens.dtype,
                    device=self.device
                )
                padded[:, :instruction_tokens.shape[1]] = instruction_tokens
                instruction_tokens = padded
            
            input_ids = instruction_tokens
            attention_mask = (input_ids != 0).float()
        
        return input_ids, attention_mask
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for text instructions."""
        instructions = observations["instruction"]
        
        # Preprocess instructions
        input_ids, attention_mask = self._preprocess_instructions(instructions)
        batch_size = input_ids.shape[0]
        
        # Process through text model
        with torch.set_grad_enabled(not self.freeze_backbone):
            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        
        # Extract hidden states
        if hasattr(text_outputs, "hidden_states") and text_outputs.hidden_states is not None:
            sequence_output = text_outputs.hidden_states[-1]  # [B, seq_len, hidden_size]
        elif hasattr(text_outputs, "last_hidden_state"):
            sequence_output = text_outputs.last_hidden_state
        else:
            raise RuntimeError("BLIP-2 text model did not return hidden states")
        
        # Project features
        projected_sequence = self.projection(sequence_output)  # [B, seq_len, output_size]
        
        if self.final_state_only:
            # Return final valid token representation
            seq_lengths = attention_mask.sum(dim=1) - 1  # Last valid token index
            batch_indices = torch.arange(batch_size, device=self.device)
            final_output = projected_sequence[batch_indices, seq_lengths]  # [B, output_size]
            return final_output.to(dtype=torch.float32)
        else:
            # Return sequence representation with masking
            expanded_mask = attention_mask.unsqueeze(-1).expand_as(projected_sequence)
            masked_sequence = projected_sequence * expanded_mask.float()
            return masked_sequence.transpose(1, 2).to(dtype=torch.float32)  # [B, output_size, seq_len]


# Backward compatibility alias
BLIP2TextEncoder = BLIP2InstructionEncoder


class BLIP2Config:
    """Configuration helper for BLIP-2 encoders."""
    
    def __init__(self, 
                 model_name: str = "Salesforce/blip2-opt-2.7b",
                 freeze_backbone: bool = False,
                 rgb_output_size: int = 256,
                 depth_output_size: int = 192, 
                 instruction_output_size: int = 256,
                 spatial_output: bool = True,
                 final_state_only: bool = True,
                 max_instruction_length: int = 512):
        
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.rgb_output_size = rgb_output_size
        self.depth_output_size = depth_output_size
        self.instruction_output_size = instruction_output_size
        self.spatial_output = spatial_output
        self.final_state_only = final_state_only
        self.max_instruction_length = max_instruction_length
    
    def create_encoders(self, observation_space: spaces.Dict, device: torch.device) -> Tuple[
        BLIP2VisionEncoder, BLIP2DepthEncoder, BLIP2InstructionEncoder
    ]:
        """Create all three encoders with consistent configuration."""
        
        rgb_encoder = BLIP2VisionEncoder(
            observation_space=observation_space,
            device=device,
            model_name=self.model_name,
            freeze_backbone=self.freeze_backbone,
            spatial_output=self.spatial_output,
            output_size=self.rgb_output_size
        )
        
        depth_encoder = BLIP2DepthEncoder(
            observation_space=observation_space,
            device=device,
            model_name=self.model_name,
            freeze_backbone=self.freeze_backbone,
            spatial_output=self.spatial_output,
            output_size=self.depth_output_size
        )
        
        instruction_encoder = BLIP2InstructionEncoder(
            model_name=self.model_name,
            freeze_backbone=self.freeze_backbone,
            output_size=self.instruction_output_size,
            final_state_only=self.final_state_only,
            max_length=self.max_instruction_length,
            device=device
        )
        
        return rgb_encoder, depth_encoder, instruction_encoder


# Helper function for easy encoder creation
def create_blip2_encoders(
    observation_space: spaces.Dict,
    device: torch.device,
    config: Optional[BLIP2Config] = None
) -> Tuple[BLIP2VisionEncoder, BLIP2DepthEncoder, BLIP2InstructionEncoder]:
    """
    Create BLIP-2 encoders with default or custom configuration.
    
    Args:
        observation_space: Gym observation space definition
        device: PyTorch device for the encoders
        config: Optional BLIP2Config instance
    
    Returns:
        Tuple of (rgb_encoder, depth_encoder, instruction_encoder)
    """
    if config is None:
        config = BLIP2Config()
    
    return config.create_encoders(observation_space, device)