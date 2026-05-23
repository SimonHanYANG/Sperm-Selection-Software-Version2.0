'''
# 基本用法
python onnx2trtengine.py --onnx yolov8n.onnx --engine yolov8n.engine

# 使用FP16精度
python onnx2trtengine.py --onnx yolov8n.onnx --engine yolov8n.engine --fp16

# 自定义输入尺寸
python onnx2trtengine.py --onnx yolov8n.onnx --engine yolov8n.engine --height 1280 --width 1280

'''


import os
import sys
import tensorrt as trt
import numpy as np
import argparse

def onnx_to_tensorrt(onnx_model_path, engine_path, fp16_mode=False, int8_mode=False, 
                     workspace_size=1<<30, input_shape=(1, 3, 640, 640)):
    """
    将ONNX模型转换为TensorRT engine模型并保存
    
    参数:
        onnx_model_path: ONNX模型的路径
        engine_path: 保存TensorRT engine的路径
        fp16_mode: 是否启用FP16精度
        int8_mode: 是否启用INT8精度
        workspace_size: TensorRT工作空间大小(默认1GB)
        input_shape: 输入张量的形状(默认为YOLOv8的默认输入)
    """
    # 创建logger和builder
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    
    # 创建网络定义
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # 创建ONNX解析器
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 读取ONNX模型
    with open(onnx_model_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    print(f"ONNX模型加载成功: {onnx_model_path}")
    
    # 配置Builder
    config = builder.create_builder_config()
    # config.max_workspace_size = workspace_size
    
    if fp16_mode:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("启用FP16模式")
        else:
            print("警告: 平台不支持FP16")
    
    if int8_mode:
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("启用INT8模式")
        else:
            print("警告: 平台不支持INT8")
    
    # 创建优化配置文件
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, min=input_shape, opt=input_shape, max=input_shape)
    config.add_optimization_profile(profile)
    
    # 构建和序列化引擎
    print("开始构建TensorRT引擎，这可能需要几分钟...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("构建引擎失败")
        return False
    
    # 保存引擎
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"TensorRT引擎已保存到: {engine_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='将YOLOv8 ONNX模型转换为TensorRT引擎')
    parser.add_argument('--onnx', type=str, required=True, help='ONNX模型路径')
    parser.add_argument('--engine', type=str, required=True, help='输出TensorRT引擎路径')
    parser.add_argument('--fp16', action='store_true', help='启用FP16精度')
    parser.add_argument('--int8', action='store_true', help='启用INT8精度')
    parser.add_argument('--workspace', type=int, default=1, help='工作空间大小(GB)')
    parser.add_argument('--height', type=int, default=640, help='输入高度')
    parser.add_argument('--width', type=int, default=640, help='输入宽度')
    
    args = parser.parse_args()
    
    workspace_size = args.workspace * (1 << 30)  # 转换为字节
    # workspace_size = args.workspace * (1 << 30)  # 转换为字节
    
    input_shape = (1, 3, args.height, args.width)
    
    print(f"输入形状: {input_shape}")
    
    if not os.path.exists(args.onnx):
        print(f"错误: ONNX模型文件不存在: {args.onnx}")
        return
    
    onnx_to_tensorrt(
        args.onnx, 
        args.engine, 
        fp16_mode=args.fp16, 
        int8_mode=args.int8,
        workspace_size=workspace_size,
        input_shape=input_shape
    )

if __name__ == '__main__':
    main()