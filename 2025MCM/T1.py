import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# 主办国字典
hosts = {
    1896: "Greece",
    1900: "France",
    1904: "United States",
    1908: "Gteat Britain",
    1912: "Sweden",
    1920: "Belgium",
    1924: "France",
    1928: "Netherlands",
    1932: "United States",
    1936: "Germany",
    1948: "Gteat Britain",
    1952: "Finland",
    1956: "Australia",
    1960: "Italy",
    1964: "Japan",
    1968: "Mexico",
    1972: "Germany",
    1976: "Canada",
    1980: "Soviet Union",
    1984: "United States",
    1988: "South Korea",
    1992: "Spain",
    1996: "United States",
    2000: "Australia",
    2004: "Greece",
    2008: "China",
    2012: "Gteat Britain",
    2016: "Brazil",
    2020: "Japan",
    2024: "France",
    2028: "United States"
}

# 数据预处理
medals = pd.read_csv('summerOly_medal_counts.csv')
programs = pd.read_csv('summerOly_program.csv')
medals_and_programs = pd.merge(medals, programs, on='Year', how='inner')
medals_and_programs.to_csv('summerOly_medals_and_programs.csv', index=False)
medals_and_programs = pd.read_csv('summerOly_medals_and_programs.csv')
noc_change = {
    'Russian Empire': 'Russia',
    'Soviet Union': 'Russia',
    'United Team of Germany': 'Germany',
    'East Germany': 'Germany',
    'West Germany': 'Germany',
}
medals_and_programs['NOC'] = medals_and_programs['NOC'].replace(noc_change)
medals_and_programs['Host_Country'] = medals_and_programs.apply(
    lambda row: 1 if hosts.get(row['Year']) == row['NOC'] else 0,
    axis=1
)

# 添加num_a和num_m列
athletes = pd.read_csv('summerOly_athletes.csv')
num_a = athletes.groupby(['Year', 'Team']).size().reset_index(name='num_a')
num_a.rename(columns={'Team': 'NOC'}, inplace=True)
medals_and_programs = pd.merge(medals_and_programs, num_a, on=['Year', 'NOC'], how='left')
athlete_winners = athletes[athletes['Medal'] != 'No medal']
num_m = athlete_winners.groupby(['Year', 'Team'])['Medal'].count().reset_index(name='num_m')
num_m.rename(columns={'Team': 'NOC'}, inplace=True)
medals_and_programs = pd.merge(medals_and_programs, num_m, on=['Year', 'NOC'], how='left')

# 按聚类结果分别计算
divided = pd.read_csv('medal_totals.csv')
# class_1 = divided[(divided['Cluster'] == 2) | (divided['Cluster'] == 1)]['NOC'].unique()
# medals_and_programs = medals_and_programs[medals_and_programs['NOC'].isin(class_1)]
label_encoder = LabelEncoder()
medals_and_programs['NOC'] = label_encoder.fit_transform(medals_and_programs['NOC'])

Y = medals_and_programs['Gold']
X = medals_and_programs[['NOC'] + list(medals_and_programs.columns[6:])]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=37)
rf_gold = RandomForestRegressor(n_estimators=200, random_state=42)
# 使用 K 折交叉验证
scores = cross_val_score(rf_gold, X, Y, cv=5, scoring='neg_mean_squared_error')
mse_scores = -scores
print(f"Gold prediction mse: {np.mean(mse_scores):.4f}")
rf_gold.fit(X, Y)

Y = medals_and_programs['Silver']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=16)
rf_silver = RandomForestRegressor(n_estimators=200, random_state=42)
scores = cross_val_score(rf_silver, X, Y, cv=5, scoring='neg_mean_squared_error')
mse_scores = -scores
print(f"Silver prediction mse: {np.mean(mse_scores):.4f}")
rf_silver.fit(X, Y)

Y = medals_and_programs['Bronze']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=29)
rf_bronze = RandomForestRegressor(n_estimators=200, random_state=42)
scores = cross_val_score(rf_bronze, X, Y, cv=5, scoring='neg_mean_squared_error')
mse_scores = -scores
print(f"Bronze prediction mse: {np.mean(mse_scores):.4f}")
rf_bronze.fit(X, Y)

prediction = medals_and_programs[medals_and_programs['Year']==2024].copy()
prediction=prediction[prediction['NOC'] != 113]
prediction.reset_index(inplace=True,drop=True)
prediction['Year']=2028
prediction['Baseball']=prediction['Baseball']+1
prediction['Softball']=prediction['Softball']+1
prediction['Cricket']=prediction['Cricket']+2
prediction['Squash']=prediction['Squash']+2
prediction['Flag football']=prediction['Flag football']+2
prediction['Total events']=prediction['Total events']+8
prediction['Host_Country']=0
prediction.loc[prediction['NOC'] == label_encoder.transform(['United States'])[0], 'Host_Country'] = 1
prediction = prediction[['NOC'] + list(prediction.columns[6:])]

# 计算置信区间
def calculate_confidence_intervals(predict, confidence_level=0.95):
    se = np.std(predict) / np.sqrt(len(predict))
    z_alpha_half = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    lower_bound = predict - z_alpha_half * se
    upper_bound = predict + z_alpha_half * se
    return lower_bound, upper_bound

result = pd.DataFrame()
result['Gold'] = rf_gold.predict(prediction)
result['Gold'] = np.round(result['Gold'])
result['Gold Lower Bound'], result['Gold Upper Bound'] = calculate_confidence_intervals(result['Gold'])
result['Silver'] = rf_silver.predict(prediction)
result['Silver'] = np.round(result['Silver'])
result['Silver Lower Bound'], result['Silver Upper Bound'] = calculate_confidence_intervals(result['Silver'])
result['Bronze'] = rf_bronze.predict(prediction)
result['Bronze'] = np.round(result['Bronze'])
result['Bronze Lower Bound'], result['Bronze Upper Bound'] = calculate_confidence_intervals(result['Bronze'])
result['Total']=result['Gold']+result['Silver']+result['Bronze']
result['NOC']=label_encoder.inverse_transform(prediction['NOC'])
result = result.sort_values(by=['Gold','Silver','Bronze'],ascending=[False,False,False])
print(result.head())
result.to_csv('result.csv')