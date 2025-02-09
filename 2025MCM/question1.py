import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12 
# ###找到真实国家排名
# import pandas as pd
# # 读取文件
# real_counts_path = "real_counts.csv"
# countries_without_medals_path = "countries_without_medals.csv"

# real_counts = pd.read_csv(real_counts_path)
# countries_without_medals = pd.read_csv(countries_without_medals_path)

# # 为未获奖国家添加所需的列，并匹配格式
# countries_without_medals['Gold'] = 0
# countries_without_medals['Silver'] = 0
# countries_without_medals['Bronze'] = 0
# countries_without_medals['Total'] = 0

# # 确保未获奖国家数据的列顺序与 real_counts 一致
# countries_without_medals = countries_without_medals[['Year', 'NOC', 'Gold', 'Silver', 'Bronze', 'Total']]

# # 为每个年份动态设置 Rank
# rank_map = real_counts.groupby('Year')['Rank'].max().to_dict()  # 找到每年现有的最大 Rank
# countries_without_medals['Rank'] = countries_without_medals['Year'].map(rank_map).fillna(0).astype(int) + 1

# # 合并数据
# merged_data = pd.concat([real_counts, countries_without_medals[['Rank', 'NOC', 'Gold', 'Silver', 'Bronze', 'Total', 'Year']]])

# # 按 Year 和 Rank 排序
# merged_data = merged_data.sort_values(by=['Year', 'Rank']).reset_index(drop=True)

# # 保存结果到新文件
# output_file = "updated_real_counts.csv"
# merged_data.to_csv(output_file, index=False)

# print(f"文件已保存为 {output_file}")


# import pandas as pd

# # 读取文件
# file_path = "updated_real_counts.csv"  
# data = pd.read_csv(file_path)

# # 按 NOC 分组，汇总 Gold, Silver, Bronze 和 Total 列
# grouped_data = data.groupby('NOC', as_index=False).agg({
#     'Gold': 'sum',
#     'Silver': 'sum',
#     'Bronze': 'sum',
#     'Total': 'sum'
# })

# # 保存结果到文件
# output_file = "medal_totals.csv"
# grouped_data.to_csv(output_file, index=False)

# print(f"文件已保存为 {output_file}")

####---------------Kmeans聚类-------------------------

# # 读取数据
# data = pd.read_csv('medal_totals.csv')
# #新加一列
# # data['Score'] = data['Gold'] * 5 + data['Silver'] * 3 + data['Bronze'] * 2
# # 筛选用于聚类的特征（Gold 和 Total）
# features = data[['Gold', 'Total']]

# # 使用 KMeans 聚类，设置聚类数为 2
# kmeans = KMeans(n_clusters=3, random_state=42)
# data['Cluster'] = kmeans.fit_predict(features)

# # 将带有聚类结果的数据保存回文件
# data.to_csv('medal_totals.csv', index=False)
# print("聚类结果已保存到 medal_totals.csv 文件中！")

# # 可视化聚类结果
# plt.figure(figsize=(10, 7))

# colors = ['#5096DE', '#CBDE3A', '#DE6E66']

# for cluster, color in enumerate(colors):
#     cluster_data = data[data['Cluster'] == cluster]
#     plt.scatter(
#         cluster_data['Gold'], 
#         cluster_data['Total'], 
#         label=f'Cluster {cluster}', 
#         color=color, 
#         s=50
#     )

# plt.title('KMeans Clustering of Countries by Gold and Total', fontsize=14)
# plt.xlabel('Gold', fontsize=18)
# plt.ylabel('Total', fontsize=18)
# plt.legend()


# plt.show()


# # 读取CSV文件
# file_path = "merged_result.csv"  # 请替换为您的文件路径
# data = pd.read_csv(file_path)  # 读取文件时指定第一行作为列名

# # 过滤出2024年，且Team列为指定国家的数据
# teams_of_interest = ['United States', 'China', 'Japan','France','Netherlands','Great Britain','Australia']
# filtered_data = data[(data['Year'] == 2024) & (data['Team'].isin(teams_of_interest))]

# # 找出所有七个国家都有参与的运动项目
# sports_in_all_teams = filtered_data.groupby('Sport')['Team'].nunique() == 4
# sports_in_all_teams = sports_in_all_teams[sports_in_all_teams].index

# # 过滤数据，只保留所有七个国家都有参与的运动项目
# filtered_data = filtered_data[filtered_data['Sport'].isin(sports_in_all_teams)]

# # 按照Team和Sport列进行分组，计算每个国家在每个Sport上的adv值的平均值
# grouped_data = filtered_data.groupby(['Team', 'Sport'])['adv'].mean().unstack()

# # 绘制折线图
# plt.figure(figsize=(12, 6))

# # 绘制每个国家的折线图
# for team in teams_of_interest:
#     if team in grouped_data.index:
#         plt.plot(grouped_data.columns, grouped_data.loc[team], label=team)

# # 添加图例、标题和标签
# plt.legend(title='Teams')
# plt.xlabel('Sports')
# plt.ylabel('Advantage (adv)')

# # 旋转横坐标标签并调整布局
# plt.xticks(rotation=90)  # 使横坐标标签竖直显示
# plt.tight_layout()  # 调整布局，避免标签重叠

# # 显示图形
# plt.show()

# # 读取CSV文件
# file_path = "merged_result.csv"  # 请替换为您的文件路径
# data = pd.read_csv(file_path)

# # 过滤出2024年，且Team列为指定国家的数据
# teams_of_interest = ['United States', 'China', 'Japan', 'France', 'Netherlands', 'Great Britain', 'Australia']
# filtered_data = data[(data['Year'] == 2024) & (data['Team'].isin(teams_of_interest))]

# # 筛选出每个国家adv值最大的那一行
# max_adv_data = filtered_data.loc[filtered_data.groupby('Team')['adv'].idxmax()]

# # 提取Sport和adv列
# result = max_adv_data[['Team', 'Sport', 'adv']]

# # 打印结果
# print(result)


