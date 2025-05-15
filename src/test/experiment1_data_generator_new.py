#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import pandas as pd
import numpy as np
import itertools
import time

# import datetime # 当前未直接使用

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 尝试定位项目根目录，以便 'data' 文件夹在项目根目录下
project_root = current_dir
# 通常脚本在 src/algorithm 或 src/test 或 src，或者直接在项目根目录
if os.path.basename(project_root).lower() in ['algorithm', 'test']:
    project_root = os.path.dirname(project_root)  # 到 src
if os.path.basename(project_root).lower() == 'src':
    project_root = os.path.dirname(project_root)  # 到项目根目录
elif os.path.basename(os.path.dirname(current_dir)) != "":
    parent_dir_of_current = os.path.dirname(current_dir)
    if os.path.exists(os.path.join(parent_dir_of_current, 'src')) or \
            os.path.exists(os.path.join(parent_dir_of_current, 'data')):
        project_root = parent_dir_of_current

# --- 模型单元定义 (15个单元) ---
MODEL_UNITS = [
    # 名称，原始参数量，显存占用(GB)，存储占用(GB)，输出数据量(GB)
    {"name": "Input Embedding", "params": 311296000, "vram_gb": 0.579, "storage_gb": 0.580, "output_data_gb": 0.0186},
    {"name": "Encoder 1", "params": 1135702016, "vram_gb": 2.115, "storage_gb": 2.115, "output_data_gb": 0.0186},
    {"name": "Encoder 2", "params": 1135702016, "vram_gb": 2.115, "storage_gb": 2.115, "output_data_gb": 0.0186},
    {"name": "Encoder 3", "params": 1135702016, "vram_gb": 2.115, "storage_gb": 2.115, "output_data_gb": 0.0186},
    {"name": "Encoder 4", "params": 1135702016, "vram_gb": 2.115, "storage_gb": 2.115, "output_data_gb": 0.0186},
    {"name": "Encoder 5", "params": 1135702016, "vram_gb": 2.115, "storage_gb": 2.115, "output_data_gb": 0.0186},
    {"name": "Encoder 6", "params": 1135702016, "vram_gb": 2.115, "storage_gb": 2.115, "output_data_gb": 0.0186},
    {"name": "Decoder 1", "params": 1514266112, "vram_gb": 2.820, "storage_gb": 2.820, "output_data_gb": 0.0186},
    {"name": "Decoder 2", "params": 1514266112, "vram_gb": 2.820, "storage_gb": 2.820, "output_data_gb": 0.0186},
    {"name": "Decoder 3", "params": 1514266112, "vram_gb": 2.820, "storage_gb": 2.820, "output_data_gb": 0.0186},
    {"name": "Decoder 4", "params": 1514266112, "vram_gb": 2.820, "storage_gb": 2.820, "output_data_gb": 0.0186},
    {"name": "Decoder 5", "params": 1514266112, "vram_gb": 2.820, "storage_gb": 2.820, "output_data_gb": 0.0186},
    {"name": "Decoder 6", "params": 1514266112, "vram_gb": 2.820, "storage_gb": 2.820, "output_data_gb": 0.0186},
    {"name": "Final Layer Norm", "params": 19456, "vram_gb": 3.62e-5, "storage_gb": 3.62e-5, "output_data_gb": 0.0186},
    {"name": "Output Projection", "params": 311296000, "vram_gb": 0.579, "storage_gb": 0.579, "output_data_gb": None},
]
N_UNITS = len(MODEL_UNITS)
TOTAL_MODEL_PARAMS = sum(unit["params"] for unit in MODEL_UNITS)

# --- 更新后的实验参数模板 ---
node_count_options = [20]
# 节点计算能力模板 (抽象单位，与VRAM GB负载能力相当)
node_compute_options_template = [4, 6, 8, 16, 20]  # len=5qqqqqqqqqqqq
# 节点存储能力模板 (VRAM GB)
node_storage_options_template_gb = [8, 16, 24, 32, 40]  # len=5

profit_per_users_options = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350]  # len=15
# 带宽成本模板 ($/GB)
bandwidth_cost_fixed_options = [0.020, 0.035, 0.050, 0.065, 0.080]  # len=1
# 带宽值模板 (Gbps)
bandwidths_options = [100]  # len=1
# GPU成本系数模板 ($/GB-VRAM-hour，占总VRAM成本的~80%)
gpu_costs_options_template = [0.016, 0.028, 0.040, 0.052, 0.064]  # len=5
# 内存成本系数模板 ($/GB-VRAM-hour，占总VRAM成本的~20%)
memory_costs_options_template = [0.004, 0.007, 0.010, 0.013, 0.016]  # len=5
network_degrees_options = [5]  # len=1

# 预先定义的20节点拓扑 (与原始脚本一致)
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

# --- 辅助函数 ---
FIXED_DISTANCE_MATRIX = None
FIXED_NODE_COSTS_LISTS_CACHE = None
FIXED_BANDWIDTH_MATRIX_CACHE = {}


def create_fixed_distance_matrix(n=20):
    global FIXED_DISTANCE_MATRIX
    if FIXED_DISTANCE_MATRIX is not None and len(FIXED_DISTANCE_MATRIX) == n:
        return FIXED_DISTANCE_MATRIX
    matrix = np.ones((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            value = ((i * 3 + j * 5) % 9) + 2
            matrix[i, j] = value
            matrix[j, i] = value
    FIXED_DISTANCE_MATRIX = matrix.tolist()
    return FIXED_DISTANCE_MATRIX


def get_fixed_node_cost_lists(n=20):
    global FIXED_NODE_COSTS_LISTS_CACHE
    if FIXED_NODE_COSTS_LISTS_CACHE is not None and len(FIXED_NODE_COSTS_LISTS_CACHE[0]) == n:
        return FIXED_NODE_COSTS_LISTS_CACHE

    gpu_options = gpu_costs_options_template
    mem_options = memory_costs_options_template

    selected_gpu_costs = []
    selected_memory_costs = []
    for i in range(n):
        gpu_cost_idx = i % len(gpu_options)
        memory_cost_idx = (i * 2) % len(mem_options)
        selected_gpu_costs.append(gpu_options[gpu_cost_idx])
        selected_memory_costs.append(mem_options[memory_cost_idx])

    FIXED_NODE_COSTS_LISTS_CACHE = (selected_gpu_costs, selected_memory_costs)
    return FIXED_NODE_COSTS_LISTS_CACHE


def get_node_costs_matrix_from_fixed_lists(n=20):
    gpu_c_list, mem_c_list = get_fixed_node_cost_lists(n)
    return [[gpu_c_list[i], mem_c_list[i]] for i in range(n)]


def create_bandwidth_matrix(topology, bandwidth_value):
    topology_tuple = tuple(map(tuple, topology))
    cache_key = (topology_tuple, bandwidth_value)
    if cache_key in FIXED_BANDWIDTH_MATRIX_CACHE:
        return FIXED_BANDWIDTH_MATRIX_CACHE[cache_key]
    n = len(topology)
    bw_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                bw_matrix[i][j] = 0
            elif topology[i][j] == 1:
                bw_matrix[i][j] = bandwidth_value
            else:
                bw_matrix[i][j] = 0
    FIXED_BANDWIDTH_MATRIX_CACHE[cache_key] = bw_matrix
    return bw_matrix


def get_partitions(n_total_units, m_blocks):
    if m_blocks <= 0 or m_blocks > n_total_units: return []
    if m_blocks == 1: return [[[i for i in range(n_total_units)]]]
    partitions = []
    possible_cut_locations = range(1, n_total_units)
    for cuts in itertools.combinations(possible_cut_locations, m_blocks - 1):
        current_partition_scheme = []
        start_idx = 0
        for cut_point_value in sorted(list(cuts)):
            current_partition_scheme.append(list(range(start_idx, cut_point_value)))
            start_idx = cut_point_value
        current_partition_scheme.append(list(range(start_idx, n_total_units)))
        partitions.append(current_partition_scheme)
    return partitions


def calculate_demands_for_partition(partition_scheme, model_units_data):
    resource_demands_list = []
    data_sizes_list = []
    m_blocks = len(partition_scheme)
    for i, block_unit_indices in enumerate(partition_scheme):
        if not block_unit_indices:
            print(f"警告: 在分割方案中遇到空块: {partition_scheme}")
            resource_demands_list.append([0.0, 0.0])
            continue

        block_vram_sum_gb = sum(model_units_data[unit_idx]["vram_gb"] for unit_idx in block_unit_indices)

        compute_proxy = round(block_vram_sum_gb, 5)
        memory_demand_gb = round(block_vram_sum_gb, 5)
        resource_demands_list.append([compute_proxy, memory_demand_gb])

        if i < m_blocks - 1:
            last_unit_in_block_idx = block_unit_indices[-1]
            output_data = model_units_data[last_unit_in_block_idx]["output_data_gb"]

            if output_data is None:
                print(
                    f"错误: 中间切割点的输出数据为 None。块索引: {i}, 块单元: {block_unit_indices}, 最后单元: {model_units_data[last_unit_in_block_idx]['name']}")
                data_sizes_list.append(0.0)
            else:
                data_sizes_list.append(round(output_data, 5))
    return resource_demands_list, data_sizes_list


# --- 主数据生成函数 (支持分批写入CSV) ---
def create_test_data_csv_batched(module_counts_to_iterate, batch_size=50000):
    test_data_id_counter = 1
    batch_data_list = []
    first_batch_written = False

    num_nodes_fixed = node_count_options[0]
    fixed_dist_matrix = create_fixed_distance_matrix(num_nodes_fixed)

    fixed_gpu_cost_list, fixed_mem_cost_list = get_fixed_node_cost_lists(num_nodes_fixed)
    fixed_node_costs_matrix = [[fixed_gpu_cost_list[i], fixed_mem_cost_list[i]] for i in range(num_nodes_fixed)]

    representative_scalar_gpu_cost = fixed_gpu_cost_list[0] if fixed_gpu_cost_list else 0.008
    representative_scalar_memory_cost = fixed_mem_cost_list[0] if fixed_mem_cost_list else 0.002

    csv_column_order = [
        'test_data_id', 'node_count', 'module_count', 'computation_ability',
        'memory_ability', 'profit_per_user', 'model_size', 'bandwidth_cost',
        'gpu_cost', 'memory_cost', 'computation_capacity', 'resource_demands',
        'data_sizes', 'bandwidth_matrix', 'topology_degree', 'bandwidth',
        'distance_matrix', 'node_costs', 'selected_compute_values',
        'selected_storage_values', 'selected_gpu_costs', 'selected_memory_costs',
        'partition_details'
    ]

    output_filename_base = f"experiment1_data_N{N_UNITS}_partitions_real_params_v2"  # 更新文件名
    output_dir_path = os.path.join(project_root, 'data')
    os.makedirs(output_dir_path, exist_ok=True)
    output_path_temp = os.path.join(output_dir_path, f"{output_filename_base}_temp.csv")
    if os.path.exists(output_path_temp):
        os.remove(output_path_temp)

    param_combinations = itertools.product(
        node_compute_options_template,
        node_storage_options_template_gb,
        profit_per_users_options,
        bandwidth_cost_fixed_options,
        bandwidths_options,
        network_degrees_options
    )

    num_outer_combinations = 1
    for opt_list in [node_compute_options_template, node_storage_options_template_gb, profit_per_users_options,
                     bandwidth_cost_fixed_options, bandwidths_options, network_degrees_options]:
        num_outer_combinations *= len(opt_list)
    print(f"总共有 {num_outer_combinations} 种外层参数组合。")

    comb_idx = 0
    total_rows_generated_overall = 0

    for node_compute_val, node_storage_val_gb, profit_user_val, bw_cost_val, \
            bw_val, degree_val in param_combinations:

        comb_idx += 1
        if comb_idx % 100 == 0 or comb_idx == 1 or comb_idx == num_outer_combinations:
            print(f"处理外层参数组合 {comb_idx}/{num_outer_combinations}...")

        current_node_count = num_nodes_fixed
        current_computation_capacity_list = [[node_compute_val, node_storage_val_gb] for _ in range(current_node_count)]
        current_topology = ny20_topologies[degree_val]
        current_bandwidth_matrix_list = create_bandwidth_matrix(current_topology, bw_val)
        selected_compute_values_list = [node_compute_val] * current_node_count
        selected_storage_values_list = [node_storage_val_gb] * current_node_count

        for module_count_val in module_counts_to_iterate:
            if not (1 <= module_count_val <= N_UNITS):
                print(f"跳过无效的 module_count: {module_count_val}")
                continue

            partition_schemes = get_partitions(N_UNITS, module_count_val)

            for part_idx, partition_scheme_val in enumerate(partition_schemes):
                current_resource_demands_list, current_data_sizes_list = \
                    calculate_demands_for_partition(partition_scheme_val, MODEL_UNITS)

                row_dict = {
                    "test_data_id": test_data_id_counter,
                    "node_count": current_node_count,
                    "module_count": module_count_val,
                    "computation_ability": node_compute_val,
                    "memory_ability": node_storage_val_gb,
                    "profit_per_user": profit_user_val,
                    "model_size": TOTAL_MODEL_PARAMS,
                    "bandwidth_cost": bw_cost_val,
                    "gpu_cost": representative_scalar_gpu_cost,
                    "memory_cost": representative_scalar_memory_cost,
                    "computation_capacity": json.dumps(current_computation_capacity_list),
                    "resource_demands": json.dumps(current_resource_demands_list),
                    "data_sizes": json.dumps(current_data_sizes_list),
                    "bandwidth_matrix": json.dumps(current_bandwidth_matrix_list),
                    "topology_degree": degree_val,
                    "bandwidth": bw_val,
                    "distance_matrix": json.dumps(fixed_dist_matrix),
                    "node_costs": json.dumps(fixed_node_costs_matrix),
                    "selected_compute_values": json.dumps(selected_compute_values_list),
                    "selected_storage_values": json.dumps(selected_storage_values_list),
                    "selected_gpu_costs": json.dumps(fixed_gpu_cost_list),
                    "selected_memory_costs": json.dumps(fixed_mem_cost_list),
                    "partition_details": json.dumps(partition_scheme_val)
                }
                batch_data_list.append(row_dict)
                test_data_id_counter += 1
                total_rows_generated_overall += 1

                if len(batch_data_list) >= batch_size:
                    df_batch = pd.DataFrame(batch_data_list)
                    if not df_batch.empty:  # 确保DataFrame不为空
                        df_batch = df_batch[csv_column_order]
                        write_header_flag = not first_batch_written
                        if not os.path.exists(output_path_temp):
                            write_header_flag = True
                        df_batch.to_csv(output_path_temp, mode='a', header=write_header_flag, index=False)
                        print(
                            f"写入 {len(batch_data_list)} 行到 {output_path_temp} (总计: {total_rows_generated_overall})")
                    batch_data_list = []
                    first_batch_written = True

        if len(batch_data_list) > 0 and (comb_idx % 500 == 0 or comb_idx == num_outer_combinations):
            df_batch = pd.DataFrame(batch_data_list)
            if not df_batch.empty:
                df_batch = df_batch[csv_column_order]
                write_header_flag = not first_batch_written
                if not os.path.exists(output_path_temp): write_header_flag = True
                df_batch.to_csv(output_path_temp, mode='a', header=write_header_flag, index=False)
                print(
                    f"周期性写入 {len(batch_data_list)} 行到 {output_path_temp} (总计: {total_rows_generated_overall})")
            batch_data_list = []
            first_batch_written = True

    if batch_data_list:
        df_batch = pd.DataFrame(batch_data_list)
        if not df_batch.empty:
            df_batch = df_batch[csv_column_order]
            write_header_flag = not first_batch_written
            if not os.path.exists(output_path_temp): write_header_flag = True
            df_batch.to_csv(output_path_temp, mode='a', header=write_header_flag, index=False)
            print(f"写入最后 {len(batch_data_list)} 行到 {output_path_temp} (总计: {total_rows_generated_overall})")

    final_output_filename = os.path.join(output_dir_path,
                                         f"{output_filename_base}_total_{total_rows_generated_overall}.csv")
    if os.path.exists(
            output_path_temp) and total_rows_generated_overall > 0:  # Only rename if temp file exists and data was written
        try:
            if os.path.exists(final_output_filename):
                os.remove(final_output_filename)
            os.rename(output_path_temp, final_output_filename)
            print(f"文件已重命名为: {final_output_filename}")
        except OSError as e:
            print(f"重命名文件失败: {e}. 数据保留在 {output_path_temp}")
            final_output_filename = output_path_temp
    elif total_rows_generated_overall == 0:
        print("没有数据被生成，因此不创建或重命名文件。")
        if os.path.exists(output_path_temp): os.remove(output_path_temp)  # Clean up empty temp file
        final_output_filename = "No_data_generated.csv"
    else:
        print(f"错误: {output_path_temp} 未找到，但已生成 {total_rows_generated_overall} 行数据。")
        final_output_filename = output_path_temp

    print(f"成功生成 {total_rows_generated_overall} 行测试案例于 {final_output_filename}")


if __name__ == "__main__":
    print("=== 开始生成包含指定 module_count 分割方案的 CSV 实验数据 (支持分批写入, 更新成本/资源范围) ===")
    start_time = time.time()

    user_specified_module_counts = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # user_specified_module_counts = list(range(2, N_UNITS + 1))

    BATCH_WRITE_SIZE = 50000

    valid_user_module_counts = [mc for mc in user_specified_module_counts if 1 <= mc <= N_UNITS]
    if not valid_user_module_counts:
        print("错误：未提供有效的 module_count 值。请检查 user_specified_module_counts。")
    else:
        print(f"将为以下 module_count 值生成数据: {valid_user_module_counts}")
        create_test_data_csv_batched(
            module_counts_to_iterate=valid_user_module_counts,
            batch_size=BATCH_WRITE_SIZE
        )

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"数据生成完成，用时 {elapsed:.2f} 秒")
    print("=== 实验数据生成结束 ===")
