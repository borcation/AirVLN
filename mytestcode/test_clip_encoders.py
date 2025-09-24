#!/usr/bin/env python3
"""
Test script for CLIP encoders to verify functionality and dimensions
"""

import torch
import numpy as np
from gym import spaces

# Add the project path to sys.path to import modules
import sys
import os
sys.path.append('/home/work/AirVLN_ws/AirVLN')

from Model.encoders.clip_encoder import CLIPVisionEncoder, CLIPInstructionEncoder
from src.common.param import args

def test_clip_vision_encoder():
    """Test CLIP Vision Encoder"""
    print("Testing CLIP Vision Encoder...")
    
    # Create mock observation space
    observation_space = spaces.Dict({
        'rgb': spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)
    })
    
    device = torch.device('cpu')  # Use CPU for testing
    
    # Initialize encoder
    encoder = CLIPVisionEncoder(
        observation_space=observation_space,
        device=device,
        spatial_output=True,  # CMA needs spatial output
        output_size=256
    )
    
    print(f"Encoder output shape: {encoder.output_shape}")
    print(f"Encoder output size: {encoder.output_size}")
    
    # Create mock observations
    batch_size = 2
    mock_rgb = torch.randint(0, 256, (batch_size, 224, 224, 3), dtype=torch.uint8)
    observations = {'rgb': mock_rgb}
    
    # Forward pass
    try:
        features = encoder(observations)
        print(f"Output features shape: {features.shape}")
        print(f"Expected shape: {(batch_size,) + encoder.output_shape}")
        
        # Check if shapes match
        expected_shape = (batch_size,) + encoder.output_shape
        if features.shape == expected_shape:
            print("✓ Vision encoder test PASSED")
            return True
        else:
            print("✗ Vision encoder test FAILED - shape mismatch")
            return False
            
    except Exception as e:
        print(f"✗ Vision encoder test FAILED with error: {e}")
        return False

def test_clip_instruction_encoder():
    """Test CLIP Instruction Encoder"""
    print("\nTesting CLIP Instruction Encoder...")
    
    # Initialize encoder
    encoder = CLIPInstructionEncoder(
        final_state_only=False  # CMA needs sequence output
    )
    
    print(f"Encoder output size: {encoder.output_size}")
    
    # Create mock instruction tokens
    batch_size = 2
    seq_length = 20
    vocab_size = 1000
    
    # Mock instruction tokens (random token IDs)
    mock_instruction = torch.randint(1, vocab_size, (batch_size, seq_length))
    # Add some padding (zeros)
    mock_instruction[:, -5:] = 0
    
    observations = {'instruction': mock_instruction}
    
    # Forward pass
    try:
        features = encoder(observations)
        print(f"Output features shape: {features.shape}")
        
        # For sequence output, expect [batch_size, hidden_size, seq_length]
        expected_shape = (batch_size, encoder.output_size, seq_length)
        if features.shape == expected_shape:
            print("✓ Instruction encoder test PASSED")
            return True
        else:
            print(f"✗ Instruction encoder test FAILED - shape mismatch")
            print(f"Expected: {expected_shape}, Got: {features.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Instruction encoder test FAILED with error: {e}")
        return False

def test_integration():
    """Test integration with mock CMA forward pass"""
    print("\nTesting Integration...")
    
    try:
        # Mock the args for CLIP usage
        args.use_clip_encoders = True
        args.clip_model_name = "openai/clip-vit-base-patch32"
        args.freeze_clip_backbone = False
        
        # Create mock observation space
        observation_space = spaces.Dict({
            'rgb': spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            'depth': spaces.Box(low=0, high=1, shape=(256, 256, 1), dtype=np.float32),
            'instruction': spaces.Box(low=0, high=1000, shape=(100,), dtype=np.int64)
        })
        
        action_space = spaces.Discrete(4)
        device = torch.device('cpu')
        
        # Try to import and initialize CMA policy
        from Model.cma_policy import CMAPolicy
        
        policy = CMAPolicy(
            observation_space=observation_space,
            action_space=action_space,
            device=device
        )
        
        print("✓ Integration test PASSED - CMAPolicy created successfully with CLIP encoders")
        return True
        
    except Exception as e:
        print(f"✗ Integration test FAILED with error: {e}")
        return False

if __name__ == "__main__":
    print("Running CLIP encoder tests...\n")
    
    # Run tests
    vision_ok = test_clip_vision_encoder()
    instruction_ok = test_clip_instruction_encoder()
    integration_ok = test_integration()
    
    print(f"\n{'='*50}")
    print("Test Results:")
    print(f"Vision Encoder: {'PASS' if vision_ok else 'FAIL'}")
    print(f"Instruction Encoder: {'PASS' if instruction_ok else 'FAIL'}")
    print(f"Integration: {'PASS' if integration_ok else 'FAIL'}")
    
    if all([vision_ok, instruction_ok, integration_ok]):
        print("\n🎉 All tests PASSED!")
    else:
        print("\n❌ Some tests FAILED!")