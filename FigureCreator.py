import os
import pandas as pd
import matplotlib.pyplot as plt


device_name = '熱水爐-5 L1'
directory = 'F:\\PycharmProjects\\pythonProject\\results\\CNN_without_Noise_without_OD\\CNN_without_Noise_without_OD\\GT_result_reshuilu.csv'
gt_filename = f"GT_result_'{device_name}'.csv"
predict_filename = f"Predict_result_'{device_name}'.csv"

start = 300
ni = 144*2+1

# 从第一个CSV文件读取数据
# df1 = pd.read_csv(os.path.join(directory, gt_filename))
df1 = pd.read_csv(directory)
# GT_result_'主人房聽（洗衣機）4L1'
# GT_result_'冷氣-2L1'
# GT_result_'冷氣-2L2'
# GT_result_'冷氣主人房-3L2'
# GT_result_'冷氣分體機-2L3'
# GT_result_'冷氣聽-3L1'
# GT_result_'廚房蘇-4L3'
# GT_result_'浴室寶-6L2'
# GT_result_'熱水爐-5 L1'
# GT_result_'熱水爐-5 L2'
# GT_result_'熱水爐-5 L3'

data1 = df1.iloc[start:start+ni, 1]  # 选择第二列的前288个数据
print(data1)

# 从第二个CSV文件读取数据
df2 = pd.read_csv('F:\\PycharmProjects\\pythonProject\\results\\CNN_without_Noise_without_OD\\CNN_without_Noise_without_OD\\Predict_result_reshuilu.csv')
# Predict_result_'主人房聽（洗衣機）4L1'
# Predict_result_'冷氣-2L1'
# Predict_result_'冷氣-2L2'
# Predict_result_'冷氣主人房-3L2'
# Predict_result_'冷氣分體機-2L3'
# Predict_result_'冷氣聽-3L1'
# Predict_result_'廚房蘇-4L3'
# Predict_result_'浴室寶-6L2'
# Predict_result_'熱水爐-5 L1'
# Predict_result_'熱水爐-5 L2'
# Predict_result_'熱水爐-5 L3'

data2 = df2.iloc[start:start+ni, 1]  # 选择第二列的前288个数据
print(data2)

# 创建图形
plt.plot(data1, label='Ground Truth')
plt.plot(data2, label='Predict')
plt.xlabel('Sampling Points')
plt.ylabel('Power Consumption (kW)')
#plt.title('冷氣主人房-3L2')  # 设置图的标题为device_name
plt.legend()
#
# # 计算数据范围
data_range = len(data1) + 2  # 数据范围加上前后各5个点
# # 设置横坐标范围
# plt.xlim(-2, data_range)  # 设置横坐标的范围为0到数据长度-1
#
# # 调整图的布局
# plt.tight_layout()
# # 设置标签位置
plt.legend(loc='upper center', bbox_to_anchor=(0.85, 1), bbox_transform=plt.gca().transAxes)
# # 显示图形
plt.show()