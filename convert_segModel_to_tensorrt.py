#!/usr/bin/env python3
"""
Convert PyTorch model (EDANet/ADSCNet) to ONNX and TensorRT
Author: Simon
Date: 2025/07/24

Usage examples:

# FP32 精度转换ADSCNet ROI模型
python convert_segModel_to_tensorrt.py --model_dir seg_weights/ADSCNet_sperm_ROINAHead_250707 --model_type adscnet

# 使用FP32精度转换EDANet Entire ROI img模型
python convert_segModel_to_tensorrt.py --model_dir seg_weights/EDANet_sperm_ROINAHead_250707 --model_type edanet

========
========
# 使用FP32精度转换EDANet Entire img模型
python convert_segModel_to_tensorrt.py --model_dir seg_weights/EDANet_sperm_NAHeadLatest250610_0610 --model_type edanet

# 使用FP16精度转换EDANet模型
python convert_segModel_to_tensorrt.py --model_dir seg_weights/EDANet_sperm_NAHeadLatest250610_0610 --model_type edanet --precision fp16
"""

import os
import argparse
import yaml
import torch
import numpy as np
import tensorrt as trt
from model_zoo.adscnet import ADSCNet
from model_zoo.edanet import EDANet
from modules import conv1x1, ConvBNAct, DWConvBNAct, DeConvBNAct, Activation  # Needed for model loading

# 设置TensorRT日志级别
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_model(model_path, config_path, model_type='adscnet'):
    """根据配置加载PyTorch模型"""
    print(f"从 {config_path} 加载配置")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建模型实例
    if model_type.lower() == 'edanet':
        print("创建EDANet模型")
        model = EDANet(
            num_classes=config['num_classes'],
            input_channels=config['input_channels'],
            deep_supervision=config.get('deep_supervision', False),
            k=config.get('k', 40),
            num_b1=config.get('num_b1', 5),
            num_b2=config.get('num_b2', 8),
            act_type=config.get('act_type', 'relu')
        )
    else:  # 默认为ADSCNet
        print("创建ADSCNet模型")
        model = ADSCNet(
            num_classes=config['num_classes'],
            input_channels=config['input_channels'],
            deep_supervision=config.get('deep_supervision', False)
        )
    
    # 加载训练好的权重
    print(f"从 {model_path} 加载模型权重")
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, config

def convert_to_onnx(model, config, onnx_path):
    """将PyTorch模型转换为ONNX格式"""
    print(f"转换模型到ONNX格式: {onnx_path}")
    
    # 创建一个输入数据
    batch_size = 1
    channels = config['input_channels']
    height = config['input_h']
    width = config['input_w']
    dummy_input = torch.randn(batch_size, channels, height, width)
    
    # 导出到ONNX
    torch.onnx.export(
        model,                     # PyTorch模型
        dummy_input,               # 输入张量
        onnx_path,                 # 输出文件
        export_params=True,        # 存储训练参数
        opset_version=13,          # ONNX操作集版本
        do_constant_folding=True,  # 优化常量
        input_names=['input'],     # 输入张量名称
        output_names=['output'],   # 输出张量名称
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"模型成功导出为ONNX: {onnx_path}")
    return onnx_path

def build_engine(onnx_path, engine_path, precision="fp32", workspace_size=1<<30, config=None):
    """将ONNX模型转换为TensorRT引擎 (使用TensorRT 10.x API)"""
    print(f"构建TensorRT引擎: {engine_path}")
    print(f"使用精度: {precision}")
    
    # 创建TensorRT组件
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 解析ONNX文件
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("解析ONNX文件失败")
    
    # 配置构建器
    config_builder = builder.create_builder_config()
    
    # 设置工作空间大小
    config_builder.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    
    # 设置精度模式
    if precision.lower() == "fp16" and builder.platform_has_fast_fp16:
        config_builder.set_flag(trt.BuilderFlag.FP16)
        print("已启用FP16精度")
    elif precision.lower() == "int8" and builder.platform_has_fast_int8:
        config_builder.set_flag(trt.BuilderFlag.INT8)
        print("已启用INT8精度 (注意: 需要校准数据来获得最佳精度)")
    
    # 创建优化配置文件来处理动态输入
    profile = builder.create_optimization_profile()
    
    # 获取输入信息
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    input_shape = input_tensor.shape
    channels = input_shape[1]
    
    # 从配置文件获取输入尺寸，如果配置可用
    height = width = 512  # 默认值
    if config and 'input_h' in config and 'input_w' in config:
        height = config['input_h']
        width = config['input_w']
    
    # 设置最小、最佳和最大输入形状
    min_batch = 1
    opt_batch = 1
    max_batch = 8  # 可以根据需要调整最大批次大小
    
    profile.set_shape(
        input_name, 
        (min_batch, channels, height, width),      # 最小形状
        (opt_batch, channels, height, width),      # 最佳形状
        (max_batch, channels, height, width)       # 最大形状
    )
    
    config_builder.add_optimization_profile(profile)
    
    # 构建和保存引擎 - TensorRT 10.x API
    serialized_engine = builder.build_serialized_network(network, config_builder)
    
    if serialized_engine is None:
        raise RuntimeError("构建TensorRT引擎失败")
    
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    
    print(f"TensorRT引擎已构建并保存到: {engine_path}")
    return engine_path

def verify_engine(engine_path):
    """加载并验证TensorRT引擎 (使用TensorRT 10.x API)"""
    print(f"验证TensorRT引擎: {engine_path}")
    
    # 创建TensorRT运行时和反序列化引擎
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    
    engine = runtime.deserialize_cuda_engine(engine_data)
    if engine is None:
        raise RuntimeError("加载TensorRT引擎失败")
    
    # 创建执行上下文
    context = engine.create_execution_context()
    
    # 获取输入和输出名称
    input_names = []
    output_names = []
    
    # TensorRT 10.x API 使用 engine.get_tensor_name() 获取张量名称
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)
    
    # 打印引擎信息
    print(f"引擎验证成功。")
    print(f"张量数量: {engine.num_io_tensors}")
    print(f"输入张量数量: {len(input_names)}")
    print(f"输出张量数量: {len(output_names)}")
    
    # 显示所有张量信息
    print("\n所有张量信息:")
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        tensor_mode = "输入" if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT else "输出"
        tensor_dtype = engine.get_tensor_dtype(tensor_name)
        tensor_shape = engine.get_tensor_shape(tensor_name)
        
        print(f"  张量 {i}: {tensor_name} - 类型: {tensor_mode} - 数据类型: {tensor_dtype} - 形状: {tensor_shape}")
        
        # 检查是否有动态维度
        if any(dim == -1 for dim in tensor_shape) and tensor_mode == "输入":
            print(f"  注意: 张量 {tensor_name} 具有动态维度")
            
            # 尝试获取优化配置文件信息
            try:
                # 获取配置文件数量
                profile_count = engine.num_optimization_profiles
                print(f"  引擎有 {profile_count} 个优化配置文件")
                
                # 在TensorRT 10.x中，使用不同的API获取配置文件形状
                # 尝试访问引擎的配置文件信息
                for p in range(profile_count):
                    # 在TensorRT 10.x中，不再使用context.get_tensor_profile_shape
                    # 而是直接通过引擎访问配置文件形状
                    print(f"  配置文件 {p}:")
                    
                    # 设置活动优化配置文件
                    context.set_optimization_profile_async(p, 0)
                    
                    # 尝试不同的方法获取配置文件形状
                    try:
                        # 方法1: 尝试从引擎获取
                        min_shape = engine.get_profile_shape(p, tensor_name, trt.OptProfileSelector.MIN)
                        opt_shape = engine.get_profile_shape(p, tensor_name, trt.OptProfileSelector.OPT)
                        max_shape = engine.get_profile_shape(p, tensor_name, trt.OptProfileSelector.MAX)
                        print(f"    形状 - 最小: {min_shape}, 最佳: {opt_shape}, 最大: {max_shape}")
                    except (AttributeError, RuntimeError) as e1:
                        print(f"    方法1失败: {e1}")
                        
                        try:
                            # 方法2: 尝试通过设置不同的形状来推断配置
                            min_dims = context.get_min_shape(tensor_name)
                            opt_dims = context.get_opt_shape(tensor_name)
                            max_dims = context.get_max_shape(tensor_name)
                            print(f"    形状 - 最小: {min_dims}, 最佳: {opt_dims}, 最大: {max_dims}")
                        except (AttributeError, RuntimeError) as e2:
                            print(f"    方法2失败: {e2}")
                            print(f"    无法获取配置文件{p}的形状信息")
            except Exception as e:
                print(f"  无法获取配置文件信息: {e}")
                print(f"  在TensorRT 10.x中，配置文件形状获取API已更改")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='将PyTorch模型转换为ONNX和TensorRT')
    parser.add_argument('--model_dir', type=str, required=True, help='包含model.pth和config.yml的目录')
    parser.add_argument('--output_dir', type=str, default=None, help='ONNX和TensorRT模型的输出目录')
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16', 'int8'], 
                        help='TensorRT引擎的精度 (fp32, fp16, int8)')
    parser.add_argument('--model_type', type=str, default='adscnet', choices=['adscnet', 'edanet'],
                        help='模型类型 (adscnet 或 edanet)')
    args = parser.parse_args()
    
    # 设置路径
    model_path = os.path.join(args.model_dir, 'model.pth')
    config_path = os.path.join(args.model_dir, 'config.yml')
    
    if args.output_dir is None:
        args.output_dir = args.model_dir
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    onnx_path = os.path.join(args.output_dir, 'model.onnx')
    engine_path = os.path.join(args.output_dir, f'model_{args.precision}.engine')
    
    # 检查模型和配置是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    
    # 加载PyTorch模型
    model, config = load_model(model_path, config_path, args.model_type)
    
    # 转换为ONNX
    convert_to_onnx(model, config, onnx_path)
    
    # 转换为TensorRT
    build_engine(onnx_path, engine_path, args.precision, config=config)
    
    # 验证引擎
    verify_engine(engine_path)
    
    print("转换成功完成！")
    print(f"- PyTorch模型: {model_path}")
    print(f"- ONNX模型: {onnx_path}")
    print(f"- TensorRT引擎 ({args.precision}): {engine_path}")

if __name__ == '__main__':
    main()