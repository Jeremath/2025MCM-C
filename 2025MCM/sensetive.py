import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# 读取从 Natural Earth 下载的数据
world = gpd.read_file("ne_10m_admin_0_countries.shp")

# 示例 GDP 数据（替换为真实数据，以下仅为模拟数据）
gdp_data = {
    "iso_a3": world["ISO_A3"],
    "GDP": [abs(hash(iso)) % 100000 for iso in world["ISO_A3"]]  # 随机生成 GDP 数据
}

# 合并 GDP 数据
gdp_df = pd.DataFrame(gdp_data)
world = world.merge(gdp_df, left_on="ISO_A3", right_on="iso_a3", how="left")

# 检查是否有未匹配的国家
if world['GDP'].isnull().any():
    print("Warning: Some countries are missing GDP data!")

# 绘制 GDP 明暗图
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# 绘制国家边界和填充颜色
world.boundary.plot(ax=ax, linewidth=0.5, color="black")
world.plot(column="GDP", ax=ax, legend=True, cmap="plasma",
           legend_kwds={'label': "GDP by Country (Simulated)", 'orientation': "horizontal"})

# 设置标题
ax.set_title("World GDP Map (Simulated Data)", fontsize=16)
ax.axis("off")  # 关闭坐标轴

# 显示图像
plt.show()
