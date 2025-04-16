import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap

# 设置全局字体为 Calibri
plt.rcParams['font.family'] = 'Calibri'

# 读取CSV文件（而不是Excel）
# data = pd.read_excel('original_bone_combined_output.xlsx')
data = pd.read_csv('C:/Users/gavin/Downloads/231_3_combined/231_later_combined_output.csv')
data.columns = data.columns.str.lower()

# 清理数据：去除包含NaN或inf的行
data = data.dropna(subset=['area', 'deformability'])
data = data[(data['area'] != float('inf')) & (data['area'] != float('-inf'))]
data = data[(data['deformability'] != float('inf')) & (data['deformability'] != float('-inf'))]

# 添加面积和长宽比的筛选条件
# data = data[(data['area'] >= 300) & (data['area'] <= 850)]
# data = data[data['area_ratio'] < 1.05]
# data = data[data['area_ratio'] > 0.95]
data = data[data['deformability'] < 0.3]
data = data[data['deformability'] > 0]

# 自定义颜色映射
colors = [
    "#4d6fea", "#5f7bec", "#7187ee", "#8393f0", "#959ff2",  # 蓝色系（低密度）
    "#a7abf4", "#b9b7f6", "#cbc3f8", "#ddcffa", "#efdbfc",  # 蓝到粉紫（低中密度）
    "#f5e191", "#f2e06d", "#efe34a", "#ece626", "#ebe53d",  # 黄色系（中密度）
    "#f5d23d", "#f9bf3d", "#fdac3d", "#ff993d", "#ff863d",  # 黄到橙（中高密度）
    "#ff733d", "#ff603d", "#ff4d3d", "#ff3a3d", "#ff2e6d"   # 橙到红（高密度）
]
custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", colors)

# 定义不同细胞类型的标记形状
markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'X']  # 圆形、方形、菱形、上三角、下三角、五角星、星形、叉形

# 绘制总的散点图
plt.figure(figsize=(6, 5))
for i, cell_type in enumerate(data['condition'].unique()):
    subset = data[data['condition'] == cell_type]
    # 计算点密度
    density = gaussian_kde(subset[['area', 'deformability']].T)(subset[['area', 'deformability']].T)
    
    # 归一化密度并映射到散点大小
    size = (density - density.min()) / (density.max() - density.min()) * 35  # 将大小映射到0到100之间
    
    # 绘制散点图，使用自定义颜色映射表示密度，设置透明度
    scatter = plt.scatter(subset['area'], subset['deformability'], c=density, cmap=custom_cmap, alpha=0.6, 
                          label=cell_type, marker=markers[i % len(markers)], s=size)  # 使用不同的标记形状和大小
    
    # 添加密度为0.8的等密度线，使用更深的颜色
    line_color = custom_cmap(i / len(data['condition'].unique()))  # 使用自定义颜色映射
    darker_color = (line_color[0] * 0.5, line_color[1] * 0.5, line_color[2] * 0.5)  # 将颜色变暗
    try:
        sns.kdeplot(x=subset['area'], y=subset['deformability'], fill=False, color=darker_color, linewidth=2, levels=[0.8])
    except AttributeError:
        # 如果出现AttributeError，使用替代方法
        sns.kdeplot(x=subset['area'], y=subset['deformability'], fill=False, color=darker_color, linewidth=2)

    # 找到密度最高的位置
    max_density_index = density.argmax()
    max_density_x = subset['area'].iloc[max_density_index]
    max_density_y = subset['deformability'].iloc[max_density_index]
    
    # 添加细胞类型标签在密度最高的位置
    # plt.text(max_density_x, max_density_y, cell_type, fontsize=24, ha='center', va='center', color='black')

# 设置坐标轴标签
plt.xlabel('Cell size (μm²)', fontsize=24, fontweight='bold')
plt.ylabel('Deformation', fontsize=24, fontweight='bold')

# 设置坐标轴范围
# plt.xlim(0, 2500)
plt.ylim(0, 0.3)

# 设置刻度标签的字体大小
plt.tick_params(axis='both', which='major', labelsize=16)

# 移除右上边框
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 加粗左下边框
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

# 移除背景网格
plt.grid(False)

# 移除图例
plt.legend().set_visible(False)

# 移除颜色条的标题
cbar = plt.colorbar(scatter)
cbar.set_label('')  # 清空颜色条的标题

# 添加样本数标记
n_samples = len(data)
plt.text(0.05, 0.95, f'n={n_samples}', transform=plt.gca().transAxes, fontsize=24, ha='left', va='top', color='black')

# 显示总的散点图
plt.tight_layout()
plt.show()

# 为每组数据单独绘制散点图
for i, cell_type in enumerate(data['condition'].unique()):
    subset = data[data['condition'] == cell_type]
    
    plt.figure(figsize=(6, 5))
    density = gaussian_kde(subset[['area', 'deformability']].T)(subset[['area', 'deformability']].T)
    
    # 归一化密度并映射到散点大小
    size = (density - density.min()) / (density.max() - density.min()) * 35
    
    scatter = plt.scatter(subset['area'], subset['deformability'], c=density, cmap=custom_cmap, alpha=0.6, 
                          marker=markers[i % len(markers)], s=size)
    
    # 添加密度为0.8的等密度线，使用更深的颜色
    line_color = custom_cmap(i / len(data['condition'].unique()))
    darker_color = (line_color[0] * 0.5, line_color[1] * 0.5, line_color[2] * 0.5)
    try:
        sns.kdeplot(x=subset['area'], y=subset['deformability'], fill=False, color=darker_color, linewidth=2, levels=[0.8])
    except AttributeError:
        # 如果出现AttributeError，使用替代方法
        sns.kdeplot(x=subset['area'], y=subset['deformability'], fill=False, color=darker_color, linewidth=2)

    # 添加细胞类型标签
    plt.text(0.95, 0.8, cell_type, transform=plt.gca().transAxes, 
             fontsize=24, ha='right', va='top', color='black')

    # 设置坐标轴标签
    plt.xlabel('Cell size (μm²)', fontsize=24, fontweight='bold')
    plt.ylabel('Deformation', fontsize=24, fontweight='bold')

    # 设置坐标轴范围
    # plt.xlim(0, 1500)
    # plt.ylim(0, 0.2)

    # 设置刻度标签的字体大小
    plt.tick_params(axis='both', which='major', labelsize=16)

    # 移除右上边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # 加粗左下边框
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)

    # 移除背景网格
    plt.grid(False)

    # 移除颜色条的标题
    cbar = plt.colorbar(scatter)
    cbar.set_label('')  # 清空颜色条的标题

    # 添加样本数标记
    n_samples = len(subset)
    plt.text(0.05, 0.95, f'n={n_samples}', transform=plt.gca().transAxes, fontsize=24, ha='left', va='top', color='black')

    # 显示单独的散点图
    plt.tight_layout()
    plt.show()