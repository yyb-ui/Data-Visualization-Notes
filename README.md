# 可视化心得分享 —— 原生断轴术与 adjustText 标签避让

## 一、 原生 Matplotlib 手搓“断轴柱状图”

### 问题背景
在评估模型校准性能（如 HL 检验）时，KNN 模型因预测概率离散，得出的卡方值会高达 `1.7e11`（1700亿），而逻辑回归等模型的卡方值仅在 `50~100` 之间。如果画在同一坐标系里，正常模型的柱子会被压成一条线。
*解决方案：采用原生 `plt.subplots(2, 1)` 实现物理断轴，完全不依赖第三方库。*

### 核心实现技巧
1. **`subplots` 物理隔离**：使用 `sharex=False` 创建上下两张独立的子图，并分别强制锁定 `ax1` 和 `ax2` 的 Y 轴区间。
2. **“截断底座”解决视觉断层**：利用 `ax2.get_ylim()[1]` 获取底部子图顶点，为冲破天际的柱子（KNN）在下方图顶端画一个同色底座，实现视觉贯通。
3. **隐藏腰部轴刻**：通过 `ax1.tick_params` 彻底清理顶部子图底部的刻度线和标签，防止断开处出现混乱的刻度重叠。
4. **`transAxes` 绝对居中画断口斜杠**：利用相对坐标变换，在断口处精准绘制居中且永不缩放的 `//` 符号。

### 1、核心代码实现
```python
# 数据引入
names_hl = list(hl_results.keys())
chi2s = [hl_results[n]['chi2'] for n in names_hl]
ps = [hl_results[n]['p'] for n in names_hl]

# 创建上下子图画布
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(12, 8))

#分别标注子图y轴数值
ax1.set_ylim(1.5e11, 1.8e11)
ax2.set_ylim(0, 250)

# 拼合上下子图
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

x_start = -0.5
x_end = len(names_hl) - 0.5
ax1.set_xlim(x_start, x_end)
ax2.set_xlim(x_start, x_end)

x_pos = range(len(names_hl))

bottom_cap_height = ax2.get_ylim()[1] 

# 绘图
for i, (x, chi, p) in enumerate(zip(x_pos, chi2s, ps)):
    color = '#2ecc71' if p > 0.05 else '#e74c3c'
    if chi > 250: # 补全被隔断的柱子
        ax1.bar(x, chi, color=color, edgecolor='white', width=0.5)
        ax2.bar(x, bottom_cap_height, color=color, edgecolor='white', width=0.5)
    else:
        ax2.bar(x, chi, color=color, edgecolor='white', width=0.5)

ax2.axhline(y=15.507, color='red', linestyle='--', linewidth=1.5, label=r'$\chi^2$ critical (df=8, $\alpha$=0.05)')
ax1.axhline(y=15.507, color='red', linestyle='--', linewidth=1.5)

# 绘制隔断符
d = 0.5
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

# 补全标签与标题
ax2.set_xticks(range(len(names_hl)))
ax2.set_xticklabels(names_hl, rotation=15, ha='right', fontsize=9)

ax2.set_ylabel(r"HL's $\chi^2$", fontsize=11)
fig.suptitle('HL检验测试: 模型性能可信度', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "14c_hl_test_native_broken_axis.png"), dpi=150, bbox_inches='tight')
plt.show()
```

#### 1.1、核心代码解释
1. **fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(12, 8))**：建立上下两个同列子图
2. **ax.set_ylim()**：标注子图y轴取值范围
3. **ax1.spines.bottom.set_visible(False)；ax2.spines.top.set_visible(False)；ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)**：消除子图间空白以及上子图x轴信息，拼合上下子图

#### 图片展示

![HL检验结果对比（14c_hl_test_native_broken_axis.png）](./images/14c_hl_test_native_broken_axis.png)

