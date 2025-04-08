#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import json
import pandas as pd
import numpy as np
import itertools
import time
import datetime

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # src目录
root_dir = os.path.dirname(parent_dir)  # 项目根目录

# 全局参数
model_sizes = [13]  # B，模型大小单位为10亿参数
model_dict = {
    13: {"params": 13000000000}  # 13B模型
}

# 实验参数
node_count = [20]  # 物理网络的节点数量
module_count = [2]  # 模型切割的块数
node_compute_options = [7800, 9100, 10400, 11700, 13000 ]  # 计算能力
node_storage_options = [23400, 27300, 31200, 35100, 39000 ]  # 存储能力
profit_per_users = [0, 25, 50, 75, 100, 125, 150, 175, 200]  # 每用户利润
bandwidth_cost_fixed = [0.1]  # 固定的带宽成本
bandwidths = [100]  # 链路带宽
gpu_costs_options = [0.004, 0.006, 0.008, 0.010, 0.012]  # GPU成本
memory_costs_options = [0.001, 0.002, 0.004, 0.006, 0.008]  # 内存成本
network_degrees = [5]  # 网络拓扑度

# 预先定义的20节点拓扑
ny20_topologies = {
    5: [
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    ]
}

# 固定的距离矩阵和节点成本初始化
FIXED_DISTANCE_MATRIX = None
FIXED_NODE_COSTS = None
FIXED_BANDWIDTH_MATRIX = None

def create_fixed_distance_matrix(n=20):
    """创建固定的距离矩阵，对角线为1，其他位置为2~10的值"""
    matrix = np.ones((n, n), dtype=int)
    
    # 采用确定性方式填充2~10的值
    for i in range(n):
        for j in range(i+1, n):
            # 使用确定性函数生成2~10之间的值
            value = ((i * 3 + j * 5) % 9) + 2  # 生成2~10之间的值
            matrix[i, j] = value
            matrix[j, i] = value  # 对称矩阵
    
    return matrix.tolist()

def create_fixed_node_costs(n=20):
    """创建固定的节点成本矩阵"""
    gpu_costs = []
    memory_costs = []
    node_costs = []
    
    # 对每个节点都分配一个确定的GPU和内存成本
    for i in range(n):
        # 使用确定性函数选择成本
        gpu_cost_idx = i % len(gpu_costs_options)
        memory_cost_idx = (i * 2) % len(memory_costs_options)
        
        gpu_cost = gpu_costs_options[gpu_cost_idx]
        memory_cost = memory_costs_options[memory_cost_idx]
        
        gpu_costs.append(gpu_cost)
        memory_costs.append(memory_cost)
        node_costs.append([gpu_cost, memory_cost])
    
    return node_costs, gpu_costs, memory_costs

def calculate_resource_demands(model_size_b, module_count):
    """计算资源需求
    
    Args:
        model_size_b: 模型大小（单位：B，十亿参数）
        module_count: 模块数量
        
    Returns:
        list: 模块的资源需求列表
    """
    # 定义固定的总资源需求 - 与模型参数量相关，但与模块数量无关
    # 对于13B模型，设置固定的总计算和存储需求
    total_compute = model_dict[model_size_b]["params"]*0.000001  # 总计算需求
    total_storage = model_dict[model_size_b]["params"]*0.000003  # 总存储需求
    
    compute_per_module = []
    storage_per_module = []
    
    # 根据模块数分配资源，确保总和相同
    if module_count == 2:
        # 两块划分：40%-60%分配
        compute_per_module = [int(total_compute * 0.4), int(total_compute * 0.6)]
        storage_per_module = [int(total_storage * 0.4), int(total_storage * 0.6)]
    elif module_count == 3:
        # 三块划分：30%-30%-40%分配
        compute_per_module = [int(total_compute * 0.3), int(total_compute * 0.3), int(total_compute * 0.4)]
        storage_per_module = [int(total_storage * 0.3), int(total_storage * 0.3), int(total_storage * 0.4)]
    elif module_count == 4:
        # 四块划分：20%-25%-25%-30%分配
        compute_per_module = [int(total_compute * 0.2), int(total_compute * 0.25), 
                             int(total_compute * 0.25), int(total_compute * 0.3)]
        storage_per_module = [int(total_storage * 0.2), int(total_storage * 0.25), 
                             int(total_storage * 0.25), int(total_storage * 0.3)]
    elif module_count == 5:
        # 五块划分：15%-20%-20%-20%-25%分配
        compute_per_module = [int(total_compute * 0.15), int(total_compute * 0.2), 
                             int(total_compute * 0.2), int(total_compute * 0.2), int(total_compute * 0.25)]
        storage_per_module = [int(total_storage * 0.15), int(total_storage * 0.2), 
                             int(total_storage * 0.2), int(total_storage * 0.2), int(total_storage * 0.25)]
    elif module_count == 6:
        # 六块划分：10%-15%-15%-20%-20%-20%分配
        compute_per_module = [int(total_compute * 0.1), int(total_compute * 0.15), int(total_compute * 0.15),
                             int(total_compute * 0.2), int(total_compute * 0.2), int(total_compute * 0.2)]
        storage_per_module = [int(total_storage * 0.1), int(total_storage * 0.15), int(total_storage * 0.15),
                             int(total_storage * 0.2), int(total_storage * 0.2), int(total_storage * 0.2)]
    else:
        # 均匀分配作为后备方案
        unit_compute = total_compute // module_count
        unit_storage = total_storage // module_count
        
        # 分配大部分资源均匀，最后一块拿剩余的（确保总和不变）
        for i in range(module_count - 1):
            compute_per_module.append(unit_compute)
            storage_per_module.append(unit_storage)
        
        # 最后一块拿剩余的资源（确保总和等于预定总量）
        compute_per_module.append(total_compute - unit_compute * (module_count - 1))
        storage_per_module.append(total_storage - unit_storage * (module_count - 1))
    
    # 验证总和是否符合预期
    assert sum(compute_per_module) == total_compute, f"计算资源总和不等于{total_compute}，而是{sum(compute_per_module)}"
    assert sum(storage_per_module) == total_storage, f"存储资源总和不等于{total_storage}，而是{sum(storage_per_module)}"
    
    # 返回每个模块的[计算需求,存储需求]列表
    return [[compute_per_module[i], storage_per_module[i]] for i in range(module_count)]

def calculate_data_sizes(model_size_b, module_count):
    """计算模块间数据传输大小
    
    Args:
        model_size_b: 模型大小（单位：B，十亿参数）
        module_count: 模块数量
        
    Returns:
        list: 长度为module_count-1的数据大小列表，表示模块间传输的数据大小
    """
    # 使用固定的较小数据大小，确保带宽足够
    data_sizes = []
    for i in range(module_count - 1):
        # 确定性地计算数据传输大小
        data_size = 10 + i * 2  # 10-20之间的值，足够小
        data_sizes.append(data_size)
    
    return data_sizes

def create_bandwidth_matrix(topology, bandwidth):
    """创建带宽矩阵
    
    Args:
        topology: 网络拓扑（连通性矩阵）
        bandwidth: 带宽值
        
    Returns:
        list: 带宽矩阵
    """
    global FIXED_BANDWIDTH_MATRIX
    
    # 如果已经计算过，直接返回缓存值
    if FIXED_BANDWIDTH_MATRIX is not None:
        return FIXED_BANDWIDTH_MATRIX
    
    # 根据拓扑和带宽创建带宽矩阵
    n = len(topology)
    bandwidth_matrix = []
    
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(0)  # 自身节点之间没有带宽限制
            elif topology[i][j] == 1:
                row.append(bandwidth)  # 相连节点之间有带宽
            else:
                row.append(0)  # 不连接的节点之间带宽为0
        bandwidth_matrix.append(row)
    
    # 缓存结果
    FIXED_BANDWIDTH_MATRIX = bandwidth_matrix
    
    return bandwidth_matrix

def create_test_data():
    """创建测试数据集"""
    global FIXED_DISTANCE_MATRIX, FIXED_NODE_COSTS
    
    # 初始化固定值
    if FIXED_DISTANCE_MATRIX is None:
        FIXED_DISTANCE_MATRIX = create_fixed_distance_matrix(node_count[0])
    
    if FIXED_NODE_COSTS is None:
        FIXED_NODE_COSTS, fixed_gpu_costs, fixed_memory_costs = create_fixed_node_costs(node_count[0])
    
    test_data_list = []
    test_id = 1
    
    # 设置批处理大小
    batch_size = 500  # 每500条保存一次
    batch_count = 0
    
    # 创建所有可能的组合
    combinations = list(itertools.product(
        node_count,              # 节点数量
        module_count,            # 模块数量
        node_compute_options,    # 节点计算能力
        node_storage_options,    # 节点存储能力
        profit_per_users,        # 单用户利润
        model_sizes,             # 模型大小
        network_degrees,         # 拓扑度数
        bandwidths               # 带宽
    ))
    
    total = len(combinations)
    print(f"生成 {total} 个参数组合...")
    
    for combo_idx, (nodes, modules, node_compute, node_storage, profit_per_user, model_size_b, degree, bandwidth) in enumerate(combinations):
        # 固定值
        bandwidth_cost = bandwidth_cost_fixed[0]
        model_size = model_dict[model_size_b]["params"]
        
        # 计算资源需求和数据大小
        resource_demands = calculate_resource_demands(model_size_b, modules)
        data_sizes = calculate_data_sizes(model_size_b, modules)
        
        # 获取固定的距离矩阵
        distance_matrix = FIXED_DISTANCE_MATRIX
        
        # 获取对应度的拓扑
        topology = ny20_topologies[degree]
        
        # 创建带宽矩阵
        bandwidth_matrix = create_bandwidth_matrix(topology, bandwidth)
        
        # 创建节点计算和存储能力的列表 - 所有节点使用同一组固定值
        computation_capacity = [[node_compute, node_storage] for _ in range(nodes)]
        selected_compute_values = [node_compute] * nodes
        selected_storage_values = [node_storage] * nodes
        
        # 获取固定的节点成本
        node_costs = FIXED_NODE_COSTS
        selected_gpu_costs = fixed_gpu_costs
        selected_memory_costs = fixed_memory_costs
        
        test_data = {
            'test_data_id': test_id,
            'node_count': nodes,
            'module_count': modules,
            'computation_ability': node_compute,
            'memory_ability': node_storage,
            'profit_per_user': profit_per_user,
            'model_size': model_size,
            'bandwidth_cost': bandwidth_cost,
            'gpu_cost': selected_gpu_costs[0],  # 只需要一个代表值
            'memory_cost': selected_memory_costs[0],  # 只需要一个代表值
            'computation_capacity': computation_capacity,
            'resource_demands': resource_demands,
            'data_sizes': data_sizes,
            'bandwidth_matrix': bandwidth_matrix,
            'topology_degree': degree,
            'bandwidth': bandwidth,
            'distance_matrix': distance_matrix,
            'node_costs': node_costs,
            'selected_compute_values': selected_compute_values,
            'selected_storage_values': selected_storage_values,
            'selected_gpu_costs': selected_gpu_costs,
            'selected_memory_costs': selected_memory_costs,
        }
        
        test_data_list.append(test_data)
        test_id += 1
        
        # 打印进度
        if (combo_idx + 1) % 100 == 0 or combo_idx == total - 1:
            progress = (combo_idx + 1) / total * 100
            print(f"进度: {progress:.1f}% ({combo_idx + 1}/{total})")
        
        # 批量保存数据
        if len(test_data_list) >= batch_size or combo_idx == total - 1:
            if test_data_list:  # 确保有数据要保存
                batch_count += 1
                print(f"保存批次 {batch_count}...")
                
                # 转换为DataFrame
                df = pd.DataFrame(test_data_list)
                
                # 对于JSON格式的字段，需要转换
                for col in ['computation_capacity', 'resource_demands', 'data_sizes', 'bandwidth_matrix',
                            'distance_matrix', 'node_costs', 'selected_compute_values', 
                            'selected_storage_values', 'selected_gpu_costs', 'selected_memory_costs']:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: json.dumps(x))
                
                # 保存到CSV文件
                output_dir = os.path.join(root_dir, 'data')
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f'experiment1_data_batch_{batch_count}.csv')
                df.to_csv(output_file, index=False)
                
                # 清空列表
                test_data_list = []
                
                print(f"已保存到 {output_file}")
    
    print("所有数据生成完成！")
    return batch_count

def combine_batch_files(batch_count):
    """合并所有批次文件为一个完整的数据集"""
    output_dir = os.path.join(root_dir, 'data')
    batch_files = [os.path.join(output_dir, f'experiment1_data_batch_{i}.csv') for i in range(1, batch_count+1)]
    
    if not batch_files:
        print("没有找到批次文件")
        return
    
    # 合并所有DataFrame
    all_dfs = []
    for batch_file in batch_files:
        if os.path.exists(batch_file):
            df = pd.read_csv(batch_file)
            all_dfs.append(df)
        else:
            print(f"找不到批次文件: {batch_file}")
    
    if not all_dfs:
        print("没有有效的批次文件可以合并")
        return
    
    # 垂直连接所有DataFrame
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # 保存合并后的DataFrame
    output_file = os.path.join(output_dir, 'experiment1_data_module-2.csv')
    combined_df.to_csv(output_file, index=False)
    
    print(f"已将 {len(batch_files)} 个批次文件合并为 {output_file}")
    
    # 复制到src/test目录
    test_dir = os.path.join(current_dir)
    os.makedirs(test_dir, exist_ok=True)
    test_output_file = os.path.join(test_dir, 'experiment1_data_module-2.csv')
    
    combined_df.to_csv(test_output_file, index=False)
    print(f"已复制数据到 {test_output_file}")
    
    # 可选：删除批次文件
    for batch_file in batch_files:
        try:
            os.remove(batch_file)
            print(f"已删除批次文件: {batch_file}")
        except:
            print(f"无法删除批次文件: {batch_file}")

def main():
    print("=== 开始生成实验1数据 ===")
    
    start_time = time.time()
    
    # 生成测试数据
    batch_count = create_test_data()
    
    # 合并批次文件
    if batch_count > 0:
        combine_batch_files(batch_count)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"数据生成完成，用时 {elapsed:.2f} 秒")
    print("=== 实验1数据生成结束 ===")

if __name__ == "__main__":
    main() 

