# 论文方向图参考

来源：Huaiguo Ban, Xin Shi, Donghuan Liu, *3D Vector Finite Element Modeling and Validation of High-Gain Parabolic Antennas*, Mathematics **14** (2026), 1706. DOI: [10.3390/math14101706](https://doi.org/10.3390/math14101706)。使用 §3.5 Case 4、Table 5 和 Figure 16（出版社 PDF 第 17–19 页）。本文开放获取，CC BY 4.0；以下 CSV 是对原图的数字化转换，非作者发布的原始数据。

出版社 PDF：[下载](https://mdpi-res.com/d_attachment/mathematics/mathematics-14-01706/article_deploy/mathematics-14-01706.pdf)。检索时间：2026-10-29。

## 场景参数和坐标

论文波长 1 m、频率 300 MHz，口径 D=2 m，焦距 f=0.5 m，PEC 上表面，焦点 (0,0,0.5) m 附近的 Y 极化 FEM 边施加电压约束。外层 3 m 立方盒使用 Silver–Müller ABC。Figure 16 给出 VFEM 和文中标为 MEEP 的归一化 E/H 面方向图。

本代码仍以波长 1 归一化，旋转坐标令论文的轴向 +z 对应本代码 +y，论文馈源极化 +y 对应本代码 +z。因此 **E 面为本代码 yz 平面，H 面为 xy 平面**。这些定义由几何和极化确定，未为使曲线吻合而交换 E/H 面。

论文未公开具体激励边的长度、网格坐标、电压约束数值实现、反射器背面几何和原始方向图数据。第一轮曾使用半径 0.12λ 的紧支撑球对称给定电流、有限厚度阶梯 PEC 壳和较大的 PML 计算域；不能将此次比较称为完全相同离散模型的严格复现。特别是 PEC 壳厚 0.25λ 是本次选定参数，不是论文给出的参数。

## 数字化方法

`ban2026_figure16.csv` 包含角度、E/H 面 VFEM 与 MEEP 曲线的 dB 值。提取脚本 `digitize.py` 从 PDF 内嵌的 Figure 16 两条原始图像带按像素颜色识别蓝色实线/红色虚线，剔除图例区域，取每列彩色笔划的中心位置，再插值到 1°。该角度间隔只是输出采样，不代表原始图像精度。

纵轴使用原始像素 y=61 对应 +5 dB、y=725 对应 -40 dB；左图横轴 x=138..683、右图 x=854..1399 对应 -180°..180°。原始分辨率约 0.66°/像素、0.068 dB/像素，线宽/抗锯齿会增加读数不确定性；没有作者原始数据，不能据此声称亚像素精度。超出绘图区的极小值及跨度大于 4° 的缺线区保留空白，不补造零点。

复现提取：

```bash
curl -L https://mdpi-res.com/d_attachment/mathematics/mathematics-14-01706/article_deploy/mathematics-14-01706.pdf -o /tmp/ban2026.pdf
pdfimages -f 19 -l 19 -png /tmp/ban2026.pdf /tmp/ban2026-figure
venv/bin/python -m case.parabolic_antenna.reference.digitize /tmp/ban2026-figure-002.png /tmp/ban2026-figure-003.png
```

生成的比较图保留两条论文曲线，并把 native 场景的模型差别写在图标题和 JSON 中。量化结果包括归一化功率形状的前半空间相对 L2、|theta|≤50° 范围 dB 均方根差及半功率波束宽度；这些是不同模型与数字化参考之间的差异，不是严格数值误差。

可选三维球面积分得到的是总辐射功率归一化的方向性系数。论文 Table 6 报告 VFEM 11.9860 dB、MEEP 12.4062 dB，但未充分公开端口功率归一化，因此不将本文方向性系数直接当成可严格比较的端口增益。

本次下载的出版社 PDF SHA-256：`0382c7f830868d9da317bc529732e58e615c2f30a89c9591db6f131ebad95c6a`。


## 后续对齐版本

当前 `paper_run` 已使用焦点处 z 向完整电压边（映射自论文的 y 向）、3λ 立方盒和一阶 Silver–Müller 外边界，并在反射器与馈源附近使用约 λ/16 局部网格。电压边长暂取 λ/15，与论文报告的平均网格长度相同，但没有证据表明它就是论文实际使用的那条边长。论文 FEM 采用罚法，native FR 采用质量矩阵加权的线积分约束投影；两者约束同一物理电压量，但不是同一离散矩阵。

PEC 壳厚 0.25λ 仍是建模选择。旧的 r2 结果是体电流/PML 历史基线；新版位于 `voltage_local4/`，不覆盖历史结果。
