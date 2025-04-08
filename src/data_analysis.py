import pandas as pd
import os
import sys
import argparse
import glob

def analyze_data_file(input_file, output_dir):
    """
    分析单个数据文件
    
    Args:
        input_file: 输入文件路径
        output_dir: 输出目录路径
    """
    print(f"\n=== 开始分析文件: {input_file} ===")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据文件
    try:
        print(f"读取数据文件: {input_file}")
        data = pd.read_csv(input_file)
        print(f"成功读取数据，共 {len(data)} 条记录")
    except Exception as e:
        print(f"读取 {input_file} 失败: {str(e)}")
        return False
    
    # 打印数据文件的列名，以便检查
    print("数据文件包含以下列:")
    for col in data.columns:
        print(f"  - {col}")
    
    # 定义需要计算人均值的列对
    cost_profit_pairs = [
        # 多功能最小成本方案
        ('multi_func_min_cost_cost', 'multi_func_min_cost_users'),
        ('multi_func_min_cost_profit', 'multi_func_min_cost_users'),
        ('multi_func_min_cost_deploy_cost', 'multi_func_min_cost_users'),
        ('multi_func_min_cost_comm_cost', 'multi_func_min_cost_users'),
        
        # 多功能最大利润方案
        ('multi_func_profit_cost', 'multi_func_profit_users'),
        ('multi_func_profit_profit', 'multi_func_profit_users'),
        ('multi_func_profit_deploy_cost', 'multi_func_profit_users'),
        ('multi_func_profit_comm_cost', 'multi_func_profit_users'),
        
        # 多功能最小利润方案
        ('multi_func_worst_profit_cost', 'multi_func_worst_profit_users'),
        ('multi_func_worst_profit_profit', 'multi_func_worst_profit_users'),
        ('multi_func_worst_profit_deploy_cost', 'multi_func_worst_profit_users'),
        ('multi_func_worst_profit_comm_cost', 'multi_func_worst_profit_users'),
        
        # 多功能最大用户方案
        ('multi_func_max_users_cost', 'multi_func_max_users_users'),
        ('multi_func_max_users_profit', 'multi_func_max_users_users'),
        ('multi_func_max_users_deploy_cost', 'multi_func_max_users_users'),
        ('multi_func_max_users_comm_cost', 'multi_func_max_users_users'),
        
        # 单功能方案
        ('single_func_cost', 'single_func_users'),
        ('single_func_profit', 'single_func_users'),
        ('single_func_deploy_cost', 'single_func_users'),
        ('single_func_comm_cost', 'single_func_users')
    ]
    
    # 新增的节点数量和平均模块数列
    node_module_cols = [
        'multi_func_min_cost_nodes', 'multi_func_min_cost_avg_modules',
        'multi_func_profit_nodes', 'multi_func_profit_avg_modules',
        'multi_func_worst_profit_nodes', 'multi_func_worst_profit_avg_modules',
        'multi_func_max_users_nodes', 'multi_func_max_users_avg_modules',
        'single_func_nodes', 'single_func_avg_modules'
    ]
    
    # 计算人均值并添加新列，检查列是否存在
    for cost_profit_col, users_col in cost_profit_pairs:
        if cost_profit_col in data.columns and users_col in data.columns:
            new_col_name = f'{cost_profit_col}_per_user'
            data[new_col_name] = data[cost_profit_col] / data[users_col]
        else:
            print(f"警告: 列 {cost_profit_col} 或 {users_col} 不存在，跳过计算人均值")
    
    # 定义自变量和因变量列表
    independent_vars = ['profit_per_user', 'module_count', 'computation_ability', 'memory_ability']
    
    # 检查自变量是否都存在
    for var in independent_vars[:]:
        if var not in data.columns:
            print(f"警告: 自变量 {var} 不存在，将从分析中移除")
            independent_vars.remove(var)
    
    # 基本因变量 (不包括新增的节点数量和平均模块数列)
    base_dependent_vars = [
        # 多功能最小成本方案
        'multi_func_min_cost_cost', 'multi_func_min_cost_profit', 'multi_func_min_cost_users',
        'multi_func_min_cost_deploy_cost', 'multi_func_min_cost_comm_cost',
        
        # 多功能最大利润方案
        'multi_func_profit_cost', 'multi_func_profit_profit', 'multi_func_profit_users',
        'multi_func_profit_deploy_cost', 'multi_func_profit_comm_cost',
        
        # 多功能最小利润方案
        'multi_func_worst_profit_cost', 'multi_func_worst_profit_profit', 'multi_func_worst_profit_users',
        'multi_func_worst_profit_deploy_cost', 'multi_func_worst_profit_comm_cost',
        
        # 多功能最大用户方案
        'multi_func_max_users_cost', 'multi_func_max_users_profit', 'multi_func_max_users_users',
        'multi_func_max_users_deploy_cost', 'multi_func_max_users_comm_cost',
        
        # 单功能方案
        'single_func_cost', 'single_func_profit', 'single_func_users',
        'single_func_deploy_cost', 'single_func_comm_cost',
    ]
    
    # 人均值列
    per_user_vars = [f'{cost_profit_col}_per_user' for cost_profit_col, _ in cost_profit_pairs 
                    if f'{cost_profit_col}_per_user' in data.columns]
    
    # 合并所有因变量，只包含存在的列
    dependent_vars = []
    for var in base_dependent_vars + node_module_cols + per_user_vars:
        if var in data.columns:
            dependent_vars.append(var)
        else:
            print(f"警告: 因变量 {var} 不存在，将从分析中移除")
    
    # 保存处理后的数据
    processed_file = os.path.join(output_dir, 'processed_data.csv')
    data.to_csv(processed_file, index=False)
    print(f"处理后的数据已保存至: {processed_file}")
    
    # 对每个自变量进行分类汇总并保存结果
    for var in independent_vars:
        # 按当前自变量分组，计算因变量的平均值
        grouped_data = data.groupby(var)[dependent_vars].mean().reset_index()
        
        # 输出到 CSV 文件
        output_file = os.path.join(output_dir, f'summary_{var}.csv')
        grouped_data.to_csv(output_file, index=False)
        print(f"分类汇总结果已保存至: {output_file}")
    
    print(f"=== 文件 {os.path.basename(input_file)} 分析完成 ===")
    return True

def main():
    """主函数：解析命令行参数，处理输入文件"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='处理Net_OR实验结果数据')
    parser.add_argument('--input', type=str, default=None, help='输入数据文件或文件夹路径')
    parser.add_argument('--output_dir', type=str, default='./analysis_results', help='输出目录基础路径')
    args = parser.parse_args()
    
    # 如果未指定输入，尝试默认路径
    if args.input is None:
        default_paths = [
            '.././data/analysis/table/experiment1_results_parallel.csv',
            '../data/analysis/table/experiment1_results_parallel.csv',
            '../data/experiment1_results_parallel.csv',
            './experiment1_results_parallel.csv',
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                args.input = path
                print(f"使用默认数据文件: {path}")
                break
        
        if args.input is None:
            print("未找到默认数据文件，请指定--input参数")
            return False
    
    # 确定输入文件列表
    input_files = []
    
    # 检查输入是文件还是目录
    if os.path.isdir(args.input):
        # 是目录，收集所有CSV文件
        print(f"输入是目录: {args.input}")
        search_path = os.path.join(args.input, "*.csv")
        input_files = glob.glob(search_path)
        print(f"在目录中找到 {len(input_files)} 个CSV文件")
    elif os.path.isfile(args.input) and args.input.lower().endswith('.csv'):
        # 是单个CSV文件
        input_files = [args.input]
        print(f"输入是单个文件: {args.input}")
    else:
        print(f"输入 {args.input} 不是有效的CSV文件或目录")
        return False
    
    # 确保有文件要处理
    if not input_files:
        print("没有找到要处理的CSV文件")
        return False
    
    # 处理每个输入文件
    processed_count = 0
    for input_file in input_files:
        # 为每个输入文件创建单独的输出目录
        basename = os.path.basename(input_file)
        filename_no_ext = os.path.splitext(basename)[0]
        file_output_dir = os.path.join(args.output_dir, filename_no_ext)
        
        # 分析数据文件
        if analyze_data_file(input_file, file_output_dir):
            processed_count += 1
    
    print(f"\n=== 所有分析完成，共处理 {processed_count}/{len(input_files)} 个文件 ===")
    return True

if __name__ == "__main__":
    main()


