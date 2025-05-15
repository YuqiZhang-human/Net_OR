#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多线程并行版实验1测试 (支持分批读写和断点续传)
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import argparse
import json
import traceback
import concurrent.futures
from tqdm import tqdm
import multiprocessing
import glob

# 添加当前目录和算法目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # src 目录
# 假设 algorithm 目录与 test_experiment1.py 在同一级或上一级的 algorithm 子目录
algorithm_dir_candidate1 = os.path.join(current_dir, 'algorithm')
algorithm_dir_candidate2 = os.path.join(parent_dir, 'algorithm')  # common: src/algorithm

if os.path.isdir(algorithm_dir_candidate1):
    algorithm_dir = algorithm_dir_candidate1
elif os.path.isdir(algorithm_dir_candidate2):
    algorithm_dir = algorithm_dir_candidate2
else:
    # Fallback or error if algorithm directory is not found
    print(f"错误: 未找到 'algorithm' 目录。请确保它位于 {current_dir} 或 {parent_dir} 下。")
    algorithm_dir = os.path.join(parent_dir, 'algorithm')  # Default to common structure

sys.path.append(parent_dir)  # anaconda_env/src
sys.path.append(algorithm_dir)

# 导入优化器类
try:
    from algorithm.multi_function_optimizer import MultiFunctionOptimizer
    from algorithm.single_function_optimizer import SingleFunctionOptimizer
except ImportError as e:
    print(f"导入优化器失败: {e}")
    print(f"请确保 'multi_function_optimizer.py' 和 'single_function_optimizer.py' 文件位于: {algorithm_dir}")
    sys.exit(1)

# 设置输出目录 (与原始脚本一致)
# 确保 parent_dir 指向的是项目根目录的 src 文件夹，如果脚本在 src/test 内
# 如果脚本在 src/algorithm 内，parent_dir 是 src，那么 os.path.dirname(parent_dir) 是项目根目录
project_root_dir = os.path.dirname(parent_dir) if os.path.basename(parent_dir) == 'src' else parent_dir
if not os.path.exists(os.path.join(project_root_dir, 'data')):  # Heuristic for project root
    project_root_dir = parent_dir  # If src is not parent, assume parent_dir is project root

output_base_dir = os.path.join(project_root_dir, 'data', 'analysis', 'table')
os.makedirs(output_base_dir, exist_ok=True)


def process_test_case(test_data_row_dict):
    """
    处理单个测试用例 (输入为字典)

    Args:
        test_data_row_dict: 单行测试数据转换成的字典

    Returns:
        dict: 处理结果
    """
    # 确保所有从DataFrame行转换来的值都是Python原生类型，特别是数值类型
    # Pandas在to_dict时可能会保留numpy类型，json.loads可能需要原生类型
    test_data = {}
    for key, value in test_data_row_dict.items():
        if isinstance(value, np.integer):
            test_data[key] = int(value)
        elif isinstance(value, np.floating):
            test_data[key] = float(value)
        elif isinstance(value, np.bool_):
            test_data[key] = bool(value)
        else:
            test_data[key] = value

    test_id = test_data.get('test_data_id', "N/A_ID")  # 使用更明确的默认ID

    # print(f"\n正在处理测试数据 ID: {test_id}") # 在大量并行时，此打印可能过多

    # 初始化结果字典，包含所有输入列以确保CSV结构一致
    result = test_data.copy()  # 先复制所有输入列

    # 清除可能存在的旧错误信息列，以防输入数据本身包含它们
    error_cols_to_clear = ['multi_func_error', 'single_func_error', 'process_error']
    for col in error_cols_to_clear:
        if col in result:
            del result[col]

    # 优化器需要的JSON字符串列，需要在使用前用json.loads解析
    # 这些列在原始CSV中是JSON字符串，Pandas读取时是字符串
    # 优化器类会处理json.loads()
    json_string_columns = [
        'computation_capacity', 'resource_demands', 'data_sizes',
        'bandwidth_matrix', 'distance_matrix', 'node_costs',
        'selected_compute_values', 'selected_storage_values',
        'selected_gpu_costs', 'selected_memory_costs', 'partition_details'
    ]

    # 确保优化器接收到的这些字段是字符串形式，如果它们已经是解析后的列表/字典，
    # 优化器内部的json.loads会失败。
    # 然而，如果优化器期望的是已经解析好的Python对象，这里就不需要dumps。
    # 假设优化器期望的是字典/列表，而CSV读取进来的是字符串，所以优化器会做loads。
    # 如果 process_test_case 的输入 test_data_row_dict 中这些列已经是Python对象了（例如，如果数据生成脚本直接生成了对象列表），
    # 那么就不需要这一步，或者优化器需要调整。
    # 假设从CSV读取后，这些列是字符串，优化器会处理。

    try:
        # 1. 多功能部署优化
        try:
            multi_optimizer = MultiFunctionOptimizer(test_data)  # 传递整个test_data字典
            multi_func_result = multi_optimizer.optimize_for_profit()

            if multi_func_result:
                min_cost_plan, max_profit_plan, min_profit_plan, max_users_plan = multi_func_result

                result['multi_func_min_cost_cost'] = min_cost_plan[0]
                result['multi_func_min_cost_deploy_cost'] = min_cost_plan[1]
                result['multi_func_min_cost_comm_cost'] = min_cost_plan[2]
                result['multi_func_min_cost_profit'] = min_cost_plan[3]
                result['multi_func_min_cost_users'] = min_cost_plan[4]
                result['multi_func_min_cost_nodes'] = min_cost_plan[5]
                result['multi_func_min_cost_avg_modules'] = min_cost_plan[6]

                result['multi_func_profit_cost'] = max_profit_plan[0]
                result['multi_func_profit_deploy_cost'] = max_profit_plan[1]
                result['multi_func_profit_comm_cost'] = max_profit_plan[2]
                result['multi_func_profit_profit'] = max_profit_plan[3]
                result['multi_func_profit_users'] = max_profit_plan[4]
                result['multi_func_profit_nodes'] = max_profit_plan[5]
                result['multi_func_profit_avg_modules'] = max_profit_plan[6]

                result['multi_func_worst_profit_cost'] = min_profit_plan[0]
                result['multi_func_worst_profit_deploy_cost'] = min_profit_plan[1]
                result['multi_func_worst_profit_comm_cost'] = min_profit_plan[2]
                result['multi_func_worst_profit_profit'] = min_profit_plan[3]
                result['multi_func_worst_profit_users'] = min_profit_plan[4]
                result['multi_func_worst_profit_nodes'] = min_profit_plan[5]
                result['multi_func_worst_profit_avg_modules'] = min_profit_plan[6]

                result['multi_func_max_users_cost'] = max_users_plan[0]
                result['multi_func_max_users_deploy_cost'] = max_users_plan[1]
                result['multi_func_max_users_comm_cost'] = max_users_plan[2]
                result['multi_func_max_users_profit'] = max_users_plan[3]
                result['multi_func_max_users_users'] = max_users_plan[4]
                result['multi_func_max_users_nodes'] = max_users_plan[5]
                result['multi_func_max_users_avg_modules'] = max_users_plan[6]
            else:
                result['multi_func_error'] = "无法找到有效的多功能部署方案"
        except Exception as e:
            result['multi_func_error'] = f"MultiFuncOptError: {str(e)}"
            # traceback.print_exc() # 在并行中打印traceback可能导致输出混乱

        # 2. 单功能部署优化
        try:
            single_optimizer = SingleFunctionOptimizer(test_data)
            single_func_result = single_optimizer.single_func_deployment()

            if single_func_result:
                result['single_func_cost'] = single_func_result[0]
                result['single_func_deploy_cost'] = single_func_result[1]
                result['single_func_comm_cost'] = single_func_result[2]
                result['single_func_profit'] = single_func_result[3]
                result['single_func_users'] = single_func_result[4]
                result['single_func_nodes'] = single_func_result[5]
                result['single_func_avg_modules'] = single_func_result[6]
            else:
                result['single_func_error'] = "无法找到有效的单功能部署方案"
        except Exception as e:
            result['single_func_error'] = f"SingleFuncOptError: {str(e)}"
            # traceback.print_exc()

    except Exception as e:
        result['process_error'] = f"MainProcessError: {str(e)}"
        # traceback.print_exc()

    return result


def append_results_to_csv(results_list, output_csv_file, column_order, write_header):
    """将一批结果追加到CSV文件"""
    if not results_list:
        return
    df_batch = pd.DataFrame(results_list)
    # 确保列顺序与第一次写入时一致，并处理可能新增的列（如错误列）
    # 优先使用 column_order，然后添加 df_batch 中有但 column_order 中没有的列
    final_columns = column_order[:]  # 复制
    for col in df_batch.columns:
        if col not in final_columns:
            final_columns.append(col)

    df_batch = df_batch.reindex(columns=final_columns)  # 确保所有列都存在，顺序正确

    try:
        df_batch.to_csv(output_csv_file, mode='a', header=write_header, index=False)
        # print(f"成功追加 {len(results_list)} 条记录到 {output_csv_file}")
    except Exception as e:
        print(f"写入CSV文件 {output_csv_file} 失败: {e}")


def load_processed_ids(checkpoint_file):
    """加载检查点，返回已处理的测试ID集合"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                # 假设检查点文件每行一个ID
                processed_ids = set(line.strip() for line in f if line.strip())
            print(f"已加载检查点，发现 {len(processed_ids)} 条已处理记录的ID")
            return processed_ids
        except Exception as e:
            print(f"加载检查点 {checkpoint_file} 出错: {e}")
    return set()


def update_checkpoint(newly_processed_ids_batch, checkpoint_file):
    """将新处理的ID追加到检查点文件"""
    try:
        with open(checkpoint_file, 'a') as f:  # 以追加模式打开
            for test_id in newly_processed_ids_batch:
                f.write(str(test_id) + '\n')
        # print(f"已更新检查点 {checkpoint_file}，新增 {len(newly_processed_ids_batch)} 个ID")
    except Exception as e:
        print(f"更新检查点 {checkpoint_file} 失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='多线程并行处理实验1测试数据 (支持分批读写)')
    parser.add_argument('--processes', type=int, default=None, help='并行进程数（默认为CPU核心数-1）')
    parser.add_argument('--limit', type=int, default=None, help='限制总共处理的数据条数（用于调试，会覆盖所有文件）')
    parser.add_argument('--output_pattern', type=str, default='experiment1_results_parallel_{input_name}.csv',
                        help='输出文件名模式')
    parser.add_argument('--chunk_size', type=int, default=10000, help='CSV文件分块读取的大小')  # 新增：分块读取大小
    parser.add_argument('--input', type=str, default='../data/experiment1_data_N15_partitions_real_params_v2_total_30718125.csv',
                        help='输入文件名或文件夹路径 (如果提供文件夹，则处理其中所有CSV)')
    # 移除了 --input_dir，统一由 --input 处理文件或文件夹
    args = parser.parse_args()

    if args.processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    else:
        num_processes = args.processes

    print(f"=== 开始分批并行测试 ===")
    print(f"使用 {num_processes} 个进程并行处理数据")
    print(f"CSV读取块大小: {args.chunk_size}")

    input_files_to_process = []
    if args.input is None:
        # 尝试在 'data' 目录下寻找匹配 'experiment1_data_N15_partitions_total_*.csv' 的文件
        data_dir_path = os.path.join(project_root_dir, 'data')
        if os.path.isdir(data_dir_path):
            pattern = os.path.join(data_dir_path, "experiment1_data_N15_partitions_total_*.csv")
            potential_files = glob.glob(pattern)
            if potential_files:
                input_files_to_process.extend(potential_files)
                print(f"在 {data_dir_path} 中自动找到以下文件进行处理: {input_files_to_process}")
            else:
                print(f"在默认路径 {data_dir_path} 未找到匹配 'experiment1_data_N15_partitions_total_*.csv' 的文件。")
        else:
            print(f"默认数据目录 {data_dir_path} 不存在。")

        if not input_files_to_process:
            print("请使用 --input 参数指定输入CSV文件或包含CSV文件的目录。")
            sys.exit(1)

    elif os.path.isdir(args.input):
        print(f"扫描输入文件夹: {args.input}")
        for filename in os.listdir(args.input):
            if filename.lower().endswith('.csv'):
                input_files_to_process.append(os.path.join(args.input, filename))
        print(f"在 {args.input} 中找到 {len(input_files_to_process)} 个CSV文件")
    elif os.path.isfile(args.input) and args.input.lower().endswith('.csv'):
        input_files_to_process.append(args.input)
    else:
        print(f"指定的输入 '{args.input}' 不是有效的CSV文件或目录。")
        sys.exit(1)

    if not input_files_to_process:
        print("没有找到任何CSV文件进行处理。")
        sys.exit(1)

    overall_processed_records_count = 0
    overall_start_time = time.time()

    for input_file_path in input_files_to_process:
        input_filename_base = os.path.splitext(os.path.basename(input_file_path))[0]

        output_filename = args.output_pattern.replace('{input_name}', input_filename_base)
        output_csv_file = os.path.join(output_base_dir, output_filename)
        checkpoint_file = f"{output_csv_file}.checkpoint"

        print(f"\n--- 开始处理文件: {input_file_path} ---")
        print(f"结果将保存至: {output_csv_file}")
        print(f"检查点文件: {checkpoint_file}")

        processed_ids_set = load_processed_ids(checkpoint_file)
        file_processed_records_this_run = 0
        file_start_time = time.time()

        # 获取CSV的列名，用于后续写入时保持一致
        try:
            temp_df_for_cols = pd.read_csv(input_file_path, nrows=1)
            # 确保所有原始列都在，并添加我们可能生成的新列（如错误列）
            # 这里的列顺序将用于所有批次的写入
            # 假设 process_test_case 会返回所有输入列，并可能添加新的结果列
            # 我们需要一个预期的完整列列表
            expected_output_columns = list(temp_df_for_cols.columns)
            # 添加优化器可能产生的所有结果列名 (基于 process_test_case 返回的字典键)
            # 这是一个示例，实际列名应与 process_test_case 的返回一致
            potential_result_cols = [
                'multi_func_min_cost_cost', 'multi_func_min_cost_deploy_cost', 'multi_func_min_cost_comm_cost',
                'multi_func_min_cost_profit', 'multi_func_min_cost_users', 'multi_func_min_cost_nodes',
                'multi_func_min_cost_avg_modules',
                'multi_func_profit_cost', 'multi_func_profit_deploy_cost', 'multi_func_profit_comm_cost',
                'multi_func_profit_profit', 'multi_func_profit_users', 'multi_func_profit_nodes',
                'multi_func_profit_avg_modules',
                'multi_func_worst_profit_cost', 'multi_func_worst_profit_deploy_cost',
                'multi_func_worst_profit_comm_cost',
                'multi_func_worst_profit_profit', 'multi_func_worst_profit_users', 'multi_func_worst_profit_nodes',
                'multi_func_worst_profit_avg_modules',
                'multi_func_max_users_cost', 'multi_func_max_users_deploy_cost', 'multi_func_max_users_comm_cost',
                'multi_func_max_users_profit', 'multi_func_max_users_users', 'multi_func_max_users_nodes',
                'multi_func_max_users_avg_modules',
                'single_func_cost', 'single_func_deploy_cost', 'single_func_comm_cost',
                'single_func_profit', 'single_func_users', 'single_func_nodes', 'single_func_avg_modules',
                'multi_func_error', 'single_func_error', 'process_error'
            ]
            for col in potential_result_cols:
                if col not in expected_output_columns:
                    expected_output_columns.append(col)

        except Exception as e:
            print(f"读取文件头部失败 {input_file_path}: {e}. 将无法保证列顺序。")
            expected_output_columns = None  # 无法预先确定列顺序

        # 确定第一次写入时是否需要表头
        # 如果输出文件不存在，或者为空，则需要写入表头
        write_header_for_first_batch = not os.path.exists(output_csv_file) or os.path.getsize(output_csv_file) == 0

        total_chunks = 0
        try:
            # 估算总行数用于tqdm (可选，但对用户友好)
            # num_total_rows = sum(1 for row in open(input_file_path, 'r')) -1 # -1 for header
            # print(f"文件总行数 (估算): {num_total_rows}")
            pass
        except:
            # num_total_rows = None
            pass

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            chunk_iterator = pd.read_csv(input_file_path, chunksize=args.chunk_size)

            for chunk_df in chunk_iterator:  # tqdm(chunk_iterator, desc=f"处理 {os.path.basename(input_file_path)} 的数据块", total= (num_total_rows // args.chunk_size if num_total_rows else None)):
                total_chunks += 1
                # print(f"  处理数据块 {total_chunks}...")

                # 确保 test_data_id 是字符串或可以比较的类型
                chunk_df['test_data_id'] = chunk_df['test_data_id'].astype(str)

                records_in_chunk = chunk_df.to_dict('records')

                # 过滤已处理的记录
                records_to_process_in_chunk = [
                    record for record in records_in_chunk
                    if str(record.get('test_data_id')) not in processed_ids_set
                ]

                if not records_to_process_in_chunk:
                    # print(f"    数据块 {total_chunks} 中的所有记录均已处理过，跳过。")
                    continue

                # print(f"    数据块 {total_chunks}: {len(records_in_chunk)} 行, 筛选后待处理: {len(records_to_process_in_chunk)} 行")

                futures = [executor.submit(process_test_case, record) for record in records_to_process_in_chunk]

                batch_results_from_futures = []

                # 使用tqdm显示当前批次（chunk）的处理进度
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                                   desc=f"块 {total_chunks}", leave=False):
                    try:
                        result = future.result()
                        batch_results_from_futures.append(result)
                    except Exception as e_future:
                        print(f"并行任务执行出错: {e_future}")
                        # 可以考虑记录这个错误，或者将原始记录标记为失败
                        # original_record = futures[future] # This mapping is tricky with as_completed
                        # result_placeholder = {"test_data_id": "ERROR_ID", "process_error": str(e_future)}
                        # batch_results_from_futures.append(result_placeholder)

                if batch_results_from_futures:
                    append_results_to_csv(batch_results_from_futures, output_csv_file, expected_output_columns,
                                          write_header_for_first_batch)
                    write_header_for_first_batch = False  # 之后不再写入表头

                    newly_processed_ids_this_batch = {str(res.get('test_data_id')) for res in batch_results_from_futures
                                                      if res.get('test_data_id')}
                    processed_ids_set.update(newly_processed_ids_this_batch)  # 更新全局已处理集合
                    update_checkpoint(newly_processed_ids_this_batch, checkpoint_file)  # 更新检查点文件

                    file_processed_records_this_run += len(batch_results_from_futures)
                    overall_processed_records_count += len(batch_results_from_futures)

                if args.limit is not None and overall_processed_records_count >= args.limit:
                    print(f"已达到处理上限 {args.limit} 条记录。")
                    break  # 跳出 chunk 循环

            if args.limit is not None and overall_processed_records_count >= args.limit:
                break  # 跳出文件循环

        file_end_time = time.time()
        print(f"--- 文件 {input_file_path} 处理完成 ---")
        print(
            f"本次运行处理了 {file_processed_records_this_run} 条新记录, 用时 {file_end_time - file_start_time:.2f} 秒.")
        print(f"总计已处理并保存的记录ID数量: {len(processed_ids_set)}")

    overall_end_time = time.time()
    print("\n=== 所有文件处理完毕 ===")
    print(
        f"所有文件总共新处理了 {overall_processed_records_count} 条记录, 总用时 {overall_end_time - overall_start_time:.2f} 秒.")


if __name__ == "__main__":
    main()
