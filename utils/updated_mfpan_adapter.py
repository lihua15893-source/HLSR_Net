"""
修正的MFPAN适配器 - 显式传递dims参数
"""
import torch
import torch.optim as optim
import logging
from typing import Optional, Dict, Any

class MFPANAdapter:
    """MFPAN-UNet模型适配器 - 支持重新设计的消融实验"""
    
    @staticmethod
    def is_mfpan_model(model_type: str) -> bool:
        """判断是否为MFPAN系列模型"""
        return model_type.upper().startswith('MFPAN_UNET')
    
    @staticmethod
    def create_model(model_type: str, config):
        
        """创建MFPAN模型实例（包括新的消融版本）"""
        if not MFPANAdapter.is_mfpan_model(model_type):
            raise ValueError(f"不是MFPAN模型类型: {model_type}")
        
        # 获取参数
        params = getattr(config, 'MFPAN_UNET_PARAMS', {})
        
        # 🎯 统一使用tiny版本的配置
        tiny_dims = [64, 128, 256, 512]  # 与mfpan_unet_tiny一致
        tiny_depths = [2, 2, 6, 2]       # 与mfpan_unet_tiny一致
        
        # 🆕 新的消融实验模型
        if model_type.upper() == 'MFPAN_UNET_ABLATION0':
            from models.mfpan_ablation_complete import MFPAN_Ablation0_StandardBaseline
            model = MFPAN_Ablation0_StandardBaseline(
                in_chans=config.IN_CHANNELS,
                num_classes=config.NUM_CLASSES,
                dims=tiny_dims  # 🎯 显式传递dims
            )
            logging.info("创建消融0: 标准ResNet基线")
            
        elif model_type.upper() == 'MFPAN_UNET_ABLATION1':
            from models.mfpan_ablation_complete import MFPAN_Ablation1_ConvNeXtV2
            model = MFPAN_Ablation1_ConvNeXtV2(
                in_chans=config.IN_CHANNELS,
                num_classes=config.NUM_CLASSES,
                dims=tiny_dims,  # 🎯 显式传递dims
                drop_path_rate=params.get('DROP_PATH_RATE', 0.05)
            )
            logging.info("创建消融1: ConvNeXtV2编码器验证")
            
        elif model_type.upper() == 'MFPAN_UNET_ABLATION2':
            from models.mfpan_ablation_complete import MFPAN_Ablation2_WithHLFD_MFCA
            model = MFPAN_Ablation2_WithHLFD_MFCA(
                in_chans=config.IN_CHANNELS,
                num_classes=config.NUM_CLASSES,
                dims=tiny_dims,  # 🎯 显式传递dims
                drop_path_rate=params.get('DROP_PATH_RATE', 0.05)
            )
            logging.info("创建消融2: ConvNeXtV2 + HLFD-MFCA增强")
            
        elif model_type.upper() == 'MFPAN_UNET_ABLATION3':
            from models.mfpan_ablation_complete import MFPAN_Ablation3_Full
            model = MFPAN_Ablation3_Full(
                in_chans=config.IN_CHANNELS,
                num_classes=config.NUM_CLASSES,
                dims=tiny_dims,  # 🎯 显式传递dims
                drop_path_rate=params.get('DROP_PATH_RATE', 0.05)
            )
            logging.info("创建消融3: 完整MFPAN模型")
        
        # 原有模型保持不变
        elif model_type.upper() == 'MFPAN_UNET_TINY':
            from models.HMP import mfpan_unet_tiny
            model = mfpan_unet_tiny(
                in_chans=config.IN_CHANNELS,
                num_classes=config.NUM_CLASSES,
                drop_path_rate=params.get('DROP_PATH_RATE', 0.05)
            )
            logging.info("创建完整HMP模型: ConvNeXtV2 + HLFD-MFCA + SimplifiedFusion")
            
        elif model_type.upper() == 'MFPAN_UNET_BASE':
            from models.HMP import mfpan_unet_base
            model = mfpan_unet_base(
                in_chans=config.IN_CHANNELS,
                num_classes=config.NUM_CLASSES,
                drop_path_rate=params.get('DROP_PATH_RATE', 0.2)
            )
        else:  # MFPAN_UNET (small)
            from models.HMP import mfpan_unet_small
            model = mfpan_unet_small(
                in_chans=config.IN_CHANNELS,
                num_classes=config.NUM_CLASSES,
                drop_path_rate=params.get('DROP_PATH_RATE', 0.1)
            )
        
        return model
    
    # 🎯 删除这两个不必要的函数，直接使用框架默认的
    # create_loss_function 和 create_optimizer 函数已移除
    # 让训练框架直接使用 get_loss_function 和 get_optimizer