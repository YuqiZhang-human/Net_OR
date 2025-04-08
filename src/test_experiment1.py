#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多线程并行版实验1测试
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

# 添加当前目录和算法目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
algorithm_dir = os.path.join(current_dir, 'algorithm')
sys.path.append(parent_dir)
sys.path.append(algorithm_dir)

# 导入优化器类
from algorithm.multi_function_optimizer import MultiFunctionOptimizer
from algorithm.single_function_optimizer import SingleFunctionOptimizer

# 设置输出目录
output_dir = os.path.join(parent_dir, 'data', 'analysis', 'table')
os.makedirs(output_dir, exist_ok=True)

def process_test_case(test_data):
    """
    处理单个测试用例
    
    Args:
        test_data: 测试数据字典
        
    Returns:
        dict: 处理结果
    """
    test_id = test_data.get('test_data_id', 0)
    
    # 添加当前处理信息的输出
    print(f"\n正在处理测试数据 ID: {test_id}")
    print(f"参数: profit_per_user={test_data.get('profit_per_user', 0)}, " 
          f"module_count={test_data.get('module_count', 0)}, "
          f"bandwidth={test_data.get('bandwidth', 0)}, "
          f"model_size={test_data.get('model_size', 0)}, "
          f"topology_degree={test_data.get('topology_degree', 0)}, "
          f"gpu_cost={test_data.get('gpu_cost', 0)}")
    
    # 初始化结果字典
    result = {
        'test_data_id': test_id,
        'profit_per_user': test_data.get('profit_per_user', 0),
        'model_size': test_data.get('model_size', 0),
        'module_count': test_data.get('module_count', 0),
        'topology_degree': test_data.get('topology_degree', 0),
        'bandwidth': test_data.get('bandwidth', 0),
        'gpu_cost': test_data.get('gpu_cost', 0),
        'memory_cost': test_data.get('memory_cost', 0),
        'bandwidth_cost': test_data.get('bandwidth_cost', 0),
        'computation_ability': test_data.get('computation_ability', 0),
        'memory_ability': test_data.get('memory_ability', 0),
    }
    
    try:
        print(f"ID {test_id}: 开始处理任务...")
        
        # 1. 多功能部署优化
        try:
            print(f"ID {test_id}: 开始多功能部署优化...")
            multi_optimizer = MultiFunctionOptimizer(test_data)
            print(f"ID {test_id}: 多功能部署优化器初始化完成")
            multi_func_result = multi_optimizer.optimize_for_profit()
            
            # 检查结果是否为None
            if multi_func_result:
                # 解析四种不同优化方案的结果
                min_cost_plan, max_profit_plan, min_profit_plan, max_users_plan = multi_func_result
                
                # 保存最小成本方案结果 - 现在包含部署开销和通信开销
                result['multi_func_min_cost_cost'] = min_cost_plan[0]
                result['multi_func_min_cost_deploy_cost'] = min_cost_plan[1]
                result['multi_func_min_cost_comm_cost'] = min_cost_plan[2]
                result['multi_func_min_cost_profit'] = min_cost_plan[3]
                result['multi_func_min_cost_users'] = min_cost_plan[4]
                result['multi_func_min_cost_nodes'] = min_cost_plan[5]
                result['multi_func_min_cost_avg_modules'] = min_cost_plan[6]
                
                # 保存最大利润方案结果 - 现在包含部署开销和通信开销
                result['multi_func_profit_cost'] = max_profit_plan[0]
                result['multi_func_profit_deploy_cost'] = max_profit_plan[1]
                result['multi_func_profit_comm_cost'] = max_profit_plan[2]
                result['multi_func_profit_profit'] = max_profit_plan[3]
                result['multi_func_profit_users'] = max_profit_plan[4]
                result['multi_func_profit_nodes'] = max_profit_plan[5]
                result['multi_func_profit_avg_modules'] = max_profit_plan[6]
                
                # 保存最小利润方案结果 - 现在包含部署开销和通信开销
                result['multi_func_worst_profit_cost'] = min_profit_plan[0]
                result['multi_func_worst_profit_deploy_cost'] = min_profit_plan[1]
                result['multi_func_worst_profit_comm_cost'] = min_profit_plan[2]
                result['multi_func_worst_profit_profit'] = min_profit_plan[3]
                result['multi_func_worst_profit_users'] = min_profit_plan[4]
                result['multi_func_worst_profit_nodes'] = min_profit_plan[5]
                result['multi_func_worst_profit_avg_modules'] = min_profit_plan[6]
                
                # 保存最大用户量方案结果 - 现在包含部署开销和通信开销
                result['multi_func_max_users_cost'] = max_users_plan[0]
                result['multi_func_max_users_deploy_cost'] = max_users_plan[1]
                result['multi_func_max_users_comm_cost'] = max_users_plan[2]
                result['multi_func_max_users_profit'] = max_users_plan[3]
                result['multi_func_max_users_users'] = max_users_plan[4]
                result['multi_func_max_users_nodes'] = max_users_plan[5]
                result['multi_func_max_users_avg_modules'] = max_users_plan[6]


                print(f"ID {test_id}: 多功能部署优化完成，最大用户数: {max_users_plan[4]}, 利润: {max_users_plan[3]}")
            else:
                result['multi_func_error'] = "无法找到有效的多功能部署方案"
                print(f"ID {test_id}: 多功能部署优化失败，无有效方案")
        except Exception as e:
            # 捕获并记录错误
            result['multi_func_error'] = str(e)
            print(f"ID {test_id}: 多功能部署优化出错: {str(e)}")
            traceback.print_exc()
        
        # 2. 单功能部署优化
        try:
            print(f"ID {test_id}: 开始单功能部署优化...")
            single_optimizer = SingleFunctionOptimizer(test_data)
            print(f"ID {test_id}: 单功能部署优化器初始化完成")
            single_func_result = single_optimizer.single_func_deployment()
            
            # 检查结果是否为None
            if single_func_result:
                result['single_func_cost'] = single_func_result[0]
                result['single_func_deploy_cost'] = single_func_result[1]
                result['single_func_comm_cost'] = single_func_result[2]
                result['single_func_profit'] = single_func_result[3]
                result['single_func_users'] = single_func_result[4]
                result['single_func_nodes'] = single_func_result[5]
                result['single_func_avg_modules'] = single_func_result[6]
                print(f"ID {test_id}: 单功能部署优化完成，最大用户数: {single_func_result[4]}, 利润: {single_func_result[3]}")
            else:
                result['single_func_error'] = "无法找到有效的单功能部署方案"
                print(f"ID {test_id}: 单功能部署优化失败，无有效方案")
        except Exception as e:
            # 捕获并记录错误
            result['single_func_error'] = str(e)
            print(f"ID {test_id}: 单功能部署优化出错: {str(e)}")
            traceback.print_exc()
    
    except Exception as e:
        # 捕获处理过程中的任何其他错误
        result['process_error'] = str(e)
        print(f"ID {test_id}: 处理过程发生错误: {str(e)}")
        traceback.print_exc()
    
    print(f"ID {test_id}: 处理完成\n" + "-"*50)
    return result

def save_checkpoint(results, output_file, checkpoint_file):
    """保存结果和检查点"""
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    # 保存检查点（已处理的测试ID列表）
    processed_ids = df['test_data_id'].tolist()
    with open(checkpoint_file, 'w') as f:
        json.dump(processed_ids, f)
    
    print(f"已保存结果至: {output_file}")
    print(f"已保存检查点至: {checkpoint_file}")

def load_checkpoint(checkpoint_file):
    """加载检查点，返回已处理的测试ID列表"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                processed_ids = json.load(f)
            print(f"已加载检查点，发现 {len(processed_ids)} 条已处理记录")
            return processed_ids
        except Exception as e:
            print(f"加载检查点出错: {e}")
    return []

def main():
    """主函数：读取测试数据，启动多线程处理，生成结果"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='多线程并行处理实验1测试数据')
    parser.add_argument('--processes', type=int, default=None, help='并行进程数（默认为CPU核心数-1）')
    parser.add_argument('--limit', type=int, default=None, help='限制处理的数据条数（用于调试）')
    parser.add_argument('--output', type=str, default='experiment1_results_parallel.csv', help='输出文件名或命名模式，使用{input_name}表示输入文件名')
    parser.add_argument('--batch_size', type=int, default=500, help='批处理大小（多少条记录保存一次结果）')
    parser.add_argument('--input', type=str, default='experiment1_data.csv', help='输入文件名或文件夹路径')
    parser.add_argument('--input_dir', type=str, default=None, help='输入文件夹路径，如果指定则处理该文件夹下所有CSV文件')
    args = parser.parse_args()
    
    # 设置进程数
    if args.processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)  # 默认使用CPU核心数-1
    else:
        num_processes = args.processes
    
    print(f"=== 开始多线程测试 ===")
    print(f"使用 {num_processes} 个进程并行处理数据")
    
    # 处理输入参数，确定需要处理的文件列表
    input_files = []
    
    # 如果指定了input_dir，优先处理文件夹中的所有CSV文件
    if args.input_dir:
        input_dir = args.input_dir
        if not os.path.isabs(input_dir):
            # 如果是相对路径，相对于当前目录解析
            input_dir = os.path.join(current_dir, input_dir)
        
        if os.path.exists(input_dir) and os.path.isdir(input_dir):
            print(f"扫描输入文件夹: {input_dir}")
            for filename in os.listdir(input_dir):
                if filename.lower().endswith('.csv'):
                    input_files.append(os.path.join(input_dir, filename))
            print(f"在 {input_dir} 中找到 {len(input_files)} 个CSV文件")
        else:
            print(f"指定的输入文件夹 {input_dir} 不存在或不是目录")
            sys.exit(1)
    else:
        # 如果未指定input_dir，使用单个输入文件
        # 确定输入文件路径
        test_dir = os.path.join(current_dir, 'test')
        experiment_file = os.path.join(test_dir, args.input)
        
        # 如果src/test目录下没有，再检查项目根目录
        if not os.path.exists(experiment_file):
            experiment_file = os.path.join(parent_dir, args.input)
        
        # 如果根目录下没有，检查src目录
        if not os.path.exists(experiment_file):
            experiment_file = os.path.join(current_dir, args.input)
        
        # 如果仍然没有找到，检查data目录
        if not os.path.exists(experiment_file):
            experiment_file = os.path.join(parent_dir, 'data', args.input)
        
        # 如果文件不存在，退出
        if not os.path.exists(experiment_file):
            print(f"未找到测试数据文件。请确保 {args.input} 位于src/test目录下，或项目根目录。")
            sys.exit(1)
        
        input_files.append(experiment_file)
    
    # 确保output_dir存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 对每个输入文件进行处理
    for input_file in input_files:
        input_filename = os.path.basename(input_file)
        input_name = os.path.splitext(input_filename)[0]
        
        # 根据命名模式生成输出文件名
        output_filename = args.output.replace('{input_name}', input_name)
        output_file = os.path.join(output_dir, output_filename)
        checkpoint_file = f"{output_file}.checkpoint"
        
        print(f"\n=== 处理输入文件: {input_file} ===")
        print(f"输出结果将保存至: {output_file}")
        
        # 读取测试数据文件
        print(f"读取测试数据: {input_file}")
        try:
            test_data = pd.read_csv(input_file)
            print(f"共 {len(test_data)} 条测试记录")
        except Exception as e:
            print(f"读取 {input_file} 失败: {str(e)}")
            continue
        
        # 检查是否有检查点，如果有，跳过已处理的测试
        processed_ids = load_checkpoint(checkpoint_file)
        
        # 上次运行的结果
        previous_results = []
        if processed_ids and os.path.exists(output_file):
            try:
                previous_df = pd.read_csv(output_file)
                previous_results = previous_df.to_dict('records')
                print(f"已加载 {len(previous_results)} 条已处理结果")
            except Exception as e:
                print(f"加载之前的结果出错: {e}")
        
        # 过滤出未处理的测试
        test_data_filtered = test_data[~test_data['test_data_id'].isin(processed_ids)].copy()
        print(f"过滤掉 {len(processed_ids)} 条已处理记录，剩余 {len(test_data_filtered)} 条待处理")
        
        # 如果有限制处理条数
        if args.limit is not None:
            test_data_filtered = test_data_filtered.head(args.limit)
            print(f"限制处理前 {args.limit} 条记录")
        
        # 转换为字典列表，便于处理
        test_data_records = test_data_filtered.to_dict('records')
        print(f"开始处理 {len(test_data_records)} 条测试记录...")
        
        # 所有结果（包括之前处理过的）
        all_results = previous_results.copy()
        
        # 记录开始时间
        start_time = time.time()
        
        # 批次处理，定期保存结果
        if len(test_data_records) > 0:
            # 使用ProcessPoolExecutor进行并行处理
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
                # 提交所有任务
                futures = {executor.submit(process_test_case, record): record for record in test_data_records}
                
                # 实时收集结果
                batch_results = []
                batch_count = 0
                
                print(f"\n总共需要处理 {len(futures)} 条记录")
                
                # 使用tqdm显示进度条
                for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="处理测试用例")):
                    try:
                        result = future.result()
                        batch_results.append(result)
                        all_results.append(result)
                        batch_count += 1
                        
                        # 每处理指定数量的批次保存一次
                        if batch_count >= args.batch_size:
                            save_checkpoint(all_results, output_file, checkpoint_file)
                            print(f">>> 进度: {i+1}/{len(futures)} ({(i+1)/len(futures)*100:.1f}%)")
                            print(f">>> 已完成 {batch_count} 条记录处理，已保存中间结果")
                            batch_count = 0
                    except Exception as e:
                        print(f"处理任务时出错: {str(e)}")
                        traceback.print_exc()
                
                # 保存最终结果
                if batch_results:
                    save_checkpoint(all_results, output_file, checkpoint_file)
                    
        # 计算总处理时间
        end_time = time.time()
        total_time = end_time - start_time
        records_processed = len(test_data_records)
        
        print("\n=== 处理完成 ===")
        print(f"处理了 {records_processed} 条记录，用时 {total_time:.2f} 秒")
        if records_processed > 0:
            print(f"处理速度: {records_processed/total_time:.2f} 记录/秒")
        
        save_checkpoint(all_results, output_file, checkpoint_file)
        
        # 计算统计信息
        print("\n=== 结果统计 ===")
        df_results = pd.DataFrame(all_results)
        print(f"总记录数: {len(df_results)}")
        
        # 计算各种方案的平均利润
        profit_columns = [
            'multi_func_profit_profit',
            'multi_func_max_users_profit',
            'single_func_profit'
        ]
        
        for col in profit_columns:
            if col in df_results.columns:
                avg_profit = df_results[col].mean()
                print(f"{col}: 平均值 = {avg_profit:.2f}")
        
        print(f"=== 测试文件 {input_filename} 处理完成 ===")
    
    print("\n=== 所有文件处理完成 ===")

if __name__ == "__main__":
    main()