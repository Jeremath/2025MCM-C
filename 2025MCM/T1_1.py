import pandas as pd

# 读取 CSV 文件
file_path = "summerOly_athletes.csv"  # 请将此替换为您的文件路径
data = pd.read_csv(file_path)

# 确保 Medal 列的内容规范
data['Medal'] = data['Medal'].fillna("No Medal")  # 填充缺失值

# 去重：如果 Event、Year、Team 相同，仅保留一次
data = data.drop_duplicates(subset=['Event', 'Year', 'Team', 'Medal'])

# 创建统计结果 DataFrame
# 按 Year 和 Team 分组，统计 Gold, Silver, Bronze
medal_counts = data.pivot_table(
    index=['Year', 'Team'],
    columns='Medal',
    aggfunc='size',
    fill_value=0
).reset_index()

# 确保 Gold、Silver、Bronze 列存在
medal_counts['Gold'] = medal_counts.get('Gold', 0)
medal_counts['Silver'] = medal_counts.get('Silver', 0)
medal_counts['Bronze'] = medal_counts.get('Bronze', 0)

# 计算 Total 奖牌数（Gold + Silver + Bronze）
medal_counts['Total'] = medal_counts['Gold'] + medal_counts['Silver'] + medal_counts['Bronze']

# 保留所需的列
final_result = medal_counts[['Year', 'Team', 'Gold', 'Silver', 'Bronze', 'Total']]

# 保存到新文件
output_file = "real_counts.csv"
final_result.to_csv(output_file, index=False)

print(f"文件已保存为 {output_file}")
