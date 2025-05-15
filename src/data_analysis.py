import pandas as pd
import os
import sys
import argparse
import glob
import json  # 新增：用于解析 partition_details
import numpy as np


def analyze_data_file(input_file, output_dir):
    """
    分析单个数据文件

    Args:
        input_file: 输入文件路径
        output_dir: 输出目录路径
    """
    print(f"\n=== 开始分析文件: {input_file} ===")

    os.makedirs(output_dir, exist_ok=True)

    try:
        print(f"读取数据文件: {input_file}")
        data = pd.read_csv(input_file)
        print(f"成功读取数据，共 {len(data)} 条记录")
    except Exception as e:
        print(f"读取 {input_file} 失败: {str(e)}")
        return False

    print("数据文件包含以下列:")
    for col in data.columns:
        print(f"  - {col}")

    # --- 新增：处理 partition_details 列 ---
    if 'partition_details' in data.columns:
        try:
            # 将JSON字符串转换为Python列表，然后转换为可哈希的字符串以便分组
            data['partition_details_str'] = data['partition_details'].apply(
                lambda x: str(json.loads(x)) if pd.notnull(x) and isinstance(x, str) else str(x)
            )
            print("成功处理 'partition_details' 列，并创建 'partition_details_str'")
        except Exception as e:
            print(f"处理 'partition_details' 列失败: {e}. 将尝试直接使用原始列（如果它是字符串）。")
            if not pd.api.types.is_string_dtype(data['partition_details']) and data[
                'partition_details'].notnull().any():
                data['partition_details_str'] = data['partition_details'].astype(str)  # 强制转换为字符串
            else:
                data['partition_details_str'] = data['partition_details']  # 直接使用
    else:
        print("警告: 数据文件中未找到 'partition_details' 列。将无法按分割方案进行详细分析。")
        data['partition_details_str'] = "N/A"  # 添加一个占位列以便后续代码能运行

    # 定义需要计算人均值的列对 (保持不变)
    cost_profit_pairs = [
        ('multi_func_min_cost_cost', 'multi_func_min_cost_users'),
        ('multi_func_min_cost_profit', 'multi_func_min_cost_users'),
        ('multi_func_min_cost_deploy_cost', 'multi_func_min_cost_users'),
        ('multi_func_min_cost_comm_cost', 'multi_func_min_cost_users'),
        ('multi_func_profit_cost', 'multi_func_profit_users'),
        ('multi_func_profit_profit', 'multi_func_profit_users'),
        ('multi_func_profit_deploy_cost', 'multi_func_profit_users'),
        ('multi_func_profit_comm_cost', 'multi_func_profit_users'),
        ('multi_func_worst_profit_cost', 'multi_func_worst_profit_users'),
        ('multi_func_worst_profit_profit', 'multi_func_worst_profit_users'),
        ('multi_func_worst_profit_deploy_cost', 'multi_func_worst_profit_users'),
        ('multi_func_worst_profit_comm_cost', 'multi_func_worst_profit_users'),
        ('multi_func_max_users_cost', 'multi_func_max_users_users'),
        ('multi_func_max_users_profit', 'multi_func_max_users_users'),
        ('multi_func_max_users_deploy_cost', 'multi_func_max_users_users'),
        ('multi_func_max_users_comm_cost', 'multi_func_max_users_users'),
        ('single_func_cost', 'single_func_users'),
        ('single_func_profit', 'single_func_users'),
        ('single_func_deploy_cost', 'single_func_users'),
        ('single_func_comm_cost', 'single_func_users')
    ]

    # 计算人均值 (保持不变)
    for cost_profit_col, users_col in cost_profit_pairs:
        if cost_profit_col in data.columns and users_col in data.columns:
            # 确保 users_col 不为0，避免除零错误
            # 同时，如果 users_col 为 NA，结果也应为 NA
            data[f'{cost_profit_col}_per_user'] = np.where(
                (data[users_col].notna()) & (data[users_col] != 0),
                data[cost_profit_col] / data[users_col],
                np.nan  # 或者可以是0，取决于你希望如何处理无用户或用户数为0的情况
            )
        else:
            print(f"警告: 列 {cost_profit_col} 或 {users_col} 不存在，跳过计算人均值")

    # 定义你关心的利润相关的因变量 (确保这些列名与你的结果文件一致)
    profit_dependent_vars = [
        'multi_func_min_cost_profit',
        'multi_func_profit_profit',  # 多功能最大利润方案的利润
        'multi_func_worst_profit_profit',
        'multi_func_max_users_profit',
        'single_func_profit',
        # 对应的人均利润 (如果已计算)
        'multi_func_min_cost_profit_per_user',
        'multi_func_profit_profit_per_user',
        'multi_func_worst_profit_profit_per_user',
        'multi_func_max_users_profit_per_user',
        'single_func_profit_per_user'
    ]
    # 筛选出实际存在的利润相关列
    actual_profit_vars = [var for var in profit_dependent_vars if var in data.columns]
    if not actual_profit_vars:
        print("错误：结果文件中未找到任何指定的利润相关列。请检查列名。")
        return False
    print(f"将分析以下利润相关列: {actual_profit_vars}")

    # 定义原始的自变量 (用于生成与之前类似的汇总表)
    original_independent_vars = ['profit_per_user', 'module_count', 'computation_ability', 'memory_ability']
    valid_original_vars = [var for var in original_independent_vars if var in data.columns]

    # 1. 按原始自变量分组，分析利润变化 (与原脚本类似，但只关注利润)
    for var in valid_original_vars:
        if var == 'module_count' and 'partition_details_str' in data.columns:
            # 如果是 module_count，我们稍后会结合 partition_details_str 进行更细致的分析
            # 但也可以先生成一个仅按 module_count 汇总的表
            grouped_data = data.groupby(var)[actual_profit_vars].mean().reset_index()
            output_file = os.path.join(output_dir, f'summary_profit_by_{var}.csv')
            grouped_data.to_csv(output_file, index=False)
            print(f"按 {var} 汇总的利润数据已保存至: {output_file}")
        elif var != 'module_count':  # 其他原始自变量
            grouped_data = data.groupby(var)[actual_profit_vars].mean().reset_index()
            output_file = os.path.join(output_dir, f'summary_profit_by_{var}.csv')
            grouped_data.to_csv(output_file, index=False)
            print(f"按 {var} 汇总的利润数据已保存至: {output_file}")

    # 2. 重点分析：不同 module_count 下，不同 partition_details_str 对利润的影响
    if 'module_count' in data.columns and 'partition_details_str' in data.columns:
        group_by_cols_partition = ['module_count', 'partition_details_str']

        # 确保这两列都存在
        if all(col in data.columns for col in group_by_cols_partition):
            summary_by_partition = data.groupby(group_by_cols_partition)[actual_profit_vars].mean().reset_index()
            output_file_partition = os.path.join(output_dir, 'summary_profit_by_module_count_and_partition.csv')
            summary_by_partition.to_csv(output_file_partition, index=False)
            print(f"按 module_count 和 partition_details_str 汇总的利润数据已保存至: {output_file_partition}")

            # 还可以进一步分析：对于每个 module_count，哪个 partition_details_str 的某种利润最高/最低
            # 例如，找出每个 module_count 下，使 'multi_func_profit_profit' 最大的分割方案
            if 'multi_func_profit_profit' in summary_by_partition.columns:
                best_partition_for_profit = summary_by_partition.loc[
                    summary_by_partition.groupby('module_count')['multi_func_profit_profit'].idxmax()
                ]
                output_file_best_profit = os.path.join(output_dir, 'summary_best_partition_for_max_profit_strategy.csv')
                best_partition_for_profit.to_csv(output_file_best_profit, index=False)
                print(
                    f"各 module_count 下最大化 'multi_func_profit_profit' 的分割方案已保存至: {output_file_best_profit}")
        else:
            print("警告: 'module_count' 或 'partition_details_str' 列不存在，无法按分割方案进行详细分析。")

    # 3. 分析其他自变量与 module_count 和 partition_details_str 组合对利润的影响
    #    例如，在特定的 profit_per_user 值下，不同 module_count 和 partition_details_str 的利润表现
    other_vars_for_detailed_analysis = [var for var in valid_original_vars if var not in ['module_count']]

    if 'module_count' in data.columns and 'partition_details_str' in data.columns:
        for ind_var in other_vars_for_detailed_analysis:
            group_by_cols_detailed = [ind_var, 'module_count', 'partition_details_str']
            if all(col in data.columns for col in group_by_cols_detailed):
                summary_detailed = data.groupby(group_by_cols_detailed)[actual_profit_vars].mean().reset_index()
                output_file_detailed = os.path.join(output_dir,
                                                    f'summary_profit_by_{ind_var}_module_count_partition.csv')
                summary_detailed.to_csv(output_file_detailed, index=False)
                print(
                    f"按 {ind_var}, module_count 和 partition_details_str 汇总的利润数据已保存至: {output_file_detailed}")
            else:
                print(f"警告: 无法按 {ind_var}, module_count, partition_details_str 分组，缺少列。")

    # 保存包含 partition_details_str 的完整处理后数据，以备进一步手动分析或绘图
    processed_file_with_partition_str = os.path.join(output_dir, 'processed_data_with_partition_details.csv')
    data.to_csv(processed_file_with_partition_str, index=False)
    print(f"包含 'partition_details_str' 的完整处理后数据已保存至: {processed_file_with_partition_str}")

    print(f"=== 文件 {os.path.basename(input_file)} 分析完成 ===")
    return True


def main():
    """主函数：解析命令行参数，处理输入文件"""
    parser = argparse.ArgumentParser(description='处理Net_OR实验结果数据，重点分析不同分割方案的影响')
    parser.add_argument('--input', type=str, default=None, help='输入数据文件或文件夹路径 (CSV格式)')
    parser.add_argument('--output_dir', type=str, default='./analysis_results_partition_focused',
                        help='输出目录基础路径')  # 修改默认输出目录
    args = parser.parse_args()

    if args.input is None:
        # 尝试从更通用的位置或上一级data目录寻找
        # 注意：这个默认路径需要根据你的项目结构调整
        script_dir = os.path.dirname(os.path.abspath(__file__))  # data_analysis.py 所在的目录
        project_root_candidate1 = os.path.dirname(script_dir)  # src 目录
        project_root_candidate2 = os.path.dirname(project_root_candidate1)  # 项目根目录

        default_paths = [
            os.path.join(project_root_candidate2, 'data', 'analysis', 'table', 'experiment1_results_parallel.csv'),
            # 假设结果文件在此
            os.path.join(project_root_candidate1, 'data', 'analysis', 'table', 'experiment1_results_parallel.csv'),
            # 也可以直接指定你新生成的数据文件名模式
            # 例如，如果你的结果文件名为 experiment1_data_N15_partitions_total_XXXX.csv 且在 data/analysis/table/
            # 你可能需要用 glob 来查找最新的或特定的结果文件
        ]

        # 简化：直接尝试在 data/analysis/table 目录下找最新的 experiment1_results_parallel*.csv
        # 或者，你应该直接提供包含 `partition_details` 列的那个结果文件名
        default_input_dir_for_results = os.path.join(project_root_candidate2, 'data', 'analysis', 'table')
        if os.path.isdir(default_input_dir_for_results):
            # 查找包含 "partitions_total" 并且是CSV的文件
            pattern = os.path.join(default_input_dir_for_results, "*partitions_total_*.csv")
            potential_files = glob.glob(pattern)
            if potential_files:
                # 选择最新的一个（如果文件名包含时间戳或按某种顺序）
                # 这里简单选择第一个找到的，或者你可以修改逻辑选择最新的
                args.input = potential_files[0]
                print(f"使用找到的默认数据文件: {args.input}")

        if args.input is None:  # 如果上面的逻辑没找到，再尝试旧的默认路径
            old_default_paths = [
                '.././data/analysis/table/experiment1_results_parallel.csv',
                '../data/analysis/table/experiment1_results_parallel.csv',
                '../data/experiment1_results_parallel.csv',
                './experiment1_results_parallel.csv',
            ]
            for path in old_default_paths:
                if os.path.exists(path):
                    args.input = path
                    print(f"使用旧的默认数据文件: {path} (可能不含 partition_details)")
                    break

        if args.input is None:
            print("未找到默认数据文件，请通过 --input 参数指定包含 'partition_details' 列的结果CSV文件或其所在目录。")
            return False

    input_files = []
    if os.path.isdir(args.input):
        print(f"输入是目录: {args.input}")
        search_path = os.path.join(args.input, "*.csv")
        input_files = glob.glob(search_path)
        print(f"在目录中找到 {len(input_files)} 个CSV文件")
    elif os.path.isfile(args.input) and args.input.lower().endswith('.csv'):
        input_files = [args.input]
        print(f"输入是单个文件: {args.input}")
    else:
        print(f"输入 {args.input} 不是有效的CSV文件或目录")
        return False

    if not input_files:
        print("没有找到要处理的CSV文件")
        return False

    processed_count = 0
    for input_file_path in input_files:  # 重命名变量以避免与内置的 input 冲突
        basename = os.path.basename(input_file_path)
        filename_no_ext = os.path.splitext(basename)[0]
        file_output_dir = os.path.join(args.output_dir, filename_no_ext + "_analysis")  # 给输出文件夹加后缀

        if analyze_data_file(input_file_path, file_output_dir):
            processed_count += 1

    print(f"\n=== 所有分析完成，共处理 {processed_count}/{len(input_files)} 个文件 ===")
    return True


if __name__ == "__main__":
    main()
