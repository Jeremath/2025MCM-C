import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# 定义奖牌转换字典
trans = {
    'Gold': 10,
    'Silver': 15,
    'Bronze': 5,
    'No medal': 0.6
}

# 读取文件
athletes = pd.read_csv("summerOly_athletes.csv")
programs = pd.read_csv("summerOly_program.csv")

# 筛选条件
selections = [
    ('Swimming', 'USA', 1.12),
    ('Gymnastics', 'CHN', 3.01),
    ('Volleyball', 'JPN', 7.08)
]

plt.figure(figsize=(12, 8))
plt.rcParams['font.family'] = 'Times New Roman'

for sport, country, coach_effect in selections:
    # 筛选项目数据
    Events = programs[['Year', sport]]
    athletes_filtered = athletes[(athletes["Sport"] == sport) & 
                                 (athletes['NOC'] == country)]
    # 计算得分
    athletes_filtered['Medal'] = athletes_filtered['Medal'].map(trans)
    athletes_filtered = athletes_filtered.sort_values(by='Year')
    athletes_filtered = athletes_filtered.groupby('Year')['Medal'].sum().reset_index()
    Scores = pd.merge(athletes_filtered, Events, on='Year', how='left')
    Scores['Scores'] = Scores['Medal'] / Scores[sport]
    Scores = Scores[Scores['Year'] >= 2008]

    # 画出历史数据
    plt.plot(Scores['Year'], Scores['Medal'], marker='o', linestyle='-', label=f'{sport} ({country})(Actual)')
    
    # 预测加入教练后的数据
    if 2016 in Scores['Year'].values:
        last_medal_score = Scores.loc[Scores['Year'] == 2016, 'Medal'].values[0]
        rand = random.uniform(0.9, 1.1)
        predicted_2020 = last_medal_score * coach_effect * rand
        predicted_2024 = last_medal_score * coach_effect * (2 - rand)
        
        # 添加预测数据到图中
        pred_years = [2016, 2020, 2024]
        pred_scores = [last_medal_score, predicted_2020, predicted_2024]
        plt.plot(pred_years, pred_scores, marker='o', linestyle='--', color=plt.gca().lines[-1].get_color(), label=f'{sport} ({country}) (Predicted)')

# 调整坐标轴和图例
plt.xticks([2008, 2012, 2016, 2020, 2024], fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('Medal Scores', fontsize=18)
plt.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=3, frameon=False, fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
