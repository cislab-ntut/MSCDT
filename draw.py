import matplotlib.pyplot as plt
import numpy as np

#model encryption

# 定義數據
labels = ['Heart', 'Nursery', 'Weather', 'Bank']
data1 = [2, 6, 32, 50]
data2 = [13, 25, 446, 272]
data3 = [3, 6, 30, 97]
data4 = [8526, 20141, 114779, 200704]

# 設定x軸標籤位置
x = np.arange(len(labels))

# 設定長條的寬度
width = 0.1

# 繪製長條圖
fig, ax = plt.subplots()
bar1 = ax.bar(x + width, data1, width, label='MSCDT v1')
bar2 = ax.bar(x + width*2, data2, width, label='MSCDT v2')
bar3 = ax.bar(x + width*3, data3, width, label='MSCDT v3')
bar4 = ax.bar(x + width*4, data4, width, label='HEDT')

# 添加標籤和標題
ax.set_xlabel('Labels')
ax.set_ylabel('Values')
ax.set_title('Model Encryption')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# 顯示圖表
plt.show()
