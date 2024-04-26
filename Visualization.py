import pandas as pd
import matplotlib.pyplot as plt

#加载数据
results_file_path = './result.csv'
results_data = pd.read_csv(results_file_path)

#可视化结果
plt.figure(figsize=(10, 6))
plt.plot(results_data['M'], results_data['C'], marker='o', linestyle='-', color='royalblue', label='Threshold C vs. Memory M')
plt.scatter(results_data['M'], results_data['C'], color='darkorange')
plt.title('Critical Threshold vs. Memory Length', fontsize=16)
plt.xlabel('Memory Length ($M$)', fontsize=12)
plt.ylabel('Critical Threshold ($C$)', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
# 设定x轴和y轴范围
plt.xlim(left=min(results_data['M']) - 1)  #X轴开始于最小M-1
plt.xlim(right=max(results_data['M']) + 1)  #X轴终止于最大M+1
plt.ylim(bottom=min(results_data['C']) - 0.1)  #Y轴开始于最小C-0.1
plt.ylim(top=max(results_data['C']) + 0.1)  #Y轴终止于最大C+0.1
plt.tight_layout()
plt.savefig("result.png")
plt.show()
