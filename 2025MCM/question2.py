import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 创建数据
data = pd.read_csv("country_summary.csv")

df = pd.DataFrame(data)

# Compute correlations
correlation = df[["Gold", "Silver", "Bronze", "Total", 
                  "Participants", "Sports", "Events"]].corr()

# Create a visually enhanced heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar_kws={'label': 'Correlation Coefficient'},
    linewidths=0.5,
    square=True,
    annot_kws={"fontsize": 10, "fontweight": "bold"}
)
# plt.yticks(fontsize=12, rotation=0)  # 保持纵向标签不旋转
plt.title("Correlation Heatmap of Olympic Data", fontsize=16, fontweight="bold")
plt.xticks(fontsize=12, rotation=45, ha='right', fontweight="bold")
plt.yticks(fontsize=12, fontweight="bold",rotation=0)
plt.tight_layout()
plt.show()
