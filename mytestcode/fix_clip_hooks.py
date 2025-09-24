#!/usr/bin/env python3
"""
为CLIP编码器添加layer_extract兼容性的修复脚本
"""

def fix_clip_encoder_hooks():
    """修复CLIP编码器的hook兼容性"""
    
    clip_encoder_file = "/home/work/AirVLN_ws/AirVLN/Model/encoders/clip_encoder.py"
    
    print("修复CLIP编码器Hook兼容性...")
    
    # 读取文件
    with open(clip_encoder_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经有layer_extract
    if "self.layer_extract = nn.Identity()" in content:
        print("✅ CLIP编码器已经有layer_extract兼容性")
        return True
    
    # 查找并替换CLIPVisionEncoder中的layer_extract定义
    old_pattern_1 = '''        # Add layer_extract compatibility for collect_data hooks
        # Point to the spatial_projection or fc_projection for feature extraction
        if self.spatial_output:
            self.layer_extract = self.spatial_projection
        else:
            self.layer_extract = self.fc_projection'''
    
    new_pattern_1 = '''        # Add layer_extract compatibility for collect_data hooks
        # Create an identity layer that can be hooked to extract CLIP features
        self.layer_extract = nn.Identity()'''
    
    # 替换第一个出现的位置（CLIPVisionEncoder）
    if old_pattern_1 in content:
        content = content.replace(old_pattern_1, new_pattern_1, 1)
        print("✅ 修复了CLIPVisionEncoder的layer_extract")
    
    # 替换第二个出现的位置（CLIPDepthEncoder）
    if old_pattern_1 in content:
        content = content.replace(old_pattern_1, new_pattern_1, 1)
        print("✅ 修复了CLIPDepthEncoder的layer_extract")
    
    # 检查forward方法是否需要修改
    if "hooked_features = self.layer_extract(clip_features)" not in content:
        print("⚠️  需要修改forward方法以使用layer_extract")
        
        # 修改CLIPVisionEncoder的forward方法
        old_forward_1 = '''        # Use pooled output (CLS token representation)
        clip_features = vision_outputs.pooler_output  # [batch_size, clip_feature_dim]
        
        if self.spatial_output:
            # Project to spatial features
            spatial_features = self.spatial_projection(clip_features)'''
        
        new_forward_1 = '''        # Use pooled output (CLS token representation)
        clip_features = vision_outputs.pooler_output  # [batch_size, clip_feature_dim]
        
        # Apply layer_extract hook point (for collect_data compatibility)
        hooked_features = self.layer_extract(clip_features)
        
        if self.spatial_output:
            # Project to spatial features
            spatial_features = self.spatial_projection(hooked_features)'''
        
        if old_forward_1 in content:
            content = content.replace(old_forward_1, new_forward_1)
            print("✅ 修复了CLIPVisionEncoder的forward方法")
        
        # 修改CLIPDepthEncoder的forward方法类似的部分
        old_forward_2 = '''        else:
            # Simple vector output
            features = self.fc_projection(clip_features)
            return features'''
        
        new_forward_2 = '''        else:
            # Simple vector output
            features = self.fc_projection(hooked_features)
            return features'''
        
        if old_forward_2 in content:
            content = content.replace(old_forward_2, new_forward_2)
            print("✅ 修复了输出部分以使用hooked_features")
    
    # 写入文件
    with open(clip_encoder_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("🎉 CLIP编码器Hook兼容性修复完成！")
    return True

if __name__ == "__main__":
    fix_clip_encoder_hooks()