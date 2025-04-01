import pandas as pd

# 读取数据文件
# 假设你的文件名为 'your_data.csv'，如果是 Excel 文件，替换为 pd.read_excel('your_file.xlsx')
data = pd.read_csv('your_data.csv')

# 定义需要计算人均值的列对
cost_profit_pairs = [
    ('multi_func_min_cost_cost', 'multi_func_min_cost_users'),
    ('multi_func_min_cost_profit', 'multi_func_min_cost_users'),
    ('multi_func_profit_cost', 'multi_func_profit_users'),
    ('multi_func_profit_profit', 'multi_func_profit_users'),
    ('multi_func_worst_profit_cost', 'multi_func_worst_profit_users'),
    ('multi_func_worst_profit_profit', 'multi_func_worst_profit_users'),
    ('multi_func_max_users_cost', 'multi_func_max_users_users'),
    ('multi_func_max_users_profit', 'multi_func_max_users_users'),
    ('single_func_cost', 'single_func_users'),
    ('single_func_profit', 'single_func_users')
]

# 计算人均值并添加新列
for cost_profit_col, users_col in cost_profit_pairs:
    new_col_name = f'{cost_profit_col}_per_user'
    data[new_col_name] = data[cost_profit_col] / data[users_col]

# 定义自变量和因变量列表
independent_vars = ['profit_per_user', 'module_count', 'computation_ability', 'memory_ability']
dependent_vars = [
    'multi_func_min_cost_cost', 'multi_func_min_cost_profit', 'multi_func_min_cost_users',
    'multi_func_profit_cost', 'multi_func_profit_profit', 'multi_func_profit_users',
    'multi_func_worst_profit_cost', 'multi_func_worst_profit_profit', 'multi_func_worst_profit_users',
    'multi_func_max_users_cost', 'multi_func_max_users_profit', 'multi_func_max_users_users',
    'single_func_cost', 'single_func_profit', 'single_func_users'
] + [f'{cost_profit_col}_per_user' for cost_profit_col, _ in cost_profit_pairs]

# 保存处理后的数据
data.to_csv('processed_data.csv', index=False)
print("处理后的数据已保存至 'processed_data.csv'")

# 对每个自变量进行分类汇总并保存结果
for var in independent_vars:
    # 按当前自变量分组，计算因变量的平均值
    grouped_data = data.groupby(var)[dependent_vars].mean().reset_index()
    
    # 输出到 CSV 文件
    output_file = f'summary_{var}.csv'
    grouped_data.to_csv(output_file, index=False)
    print(f"分类汇总结果已保存至 {output_file}")


