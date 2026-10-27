# 周期标量平流（1D / 2D / 3D）

入口：[advection.py](../advection.py)。以下命令均从仓库根目录执行。

周期标量平流用于验证 Cartesian 1D/2D/3D 通量接口、时间推进与全局质量守恒。

```bash
python -m case.advection --dim 1
python -m case.advection --dim 2
python -m case.advection --dim 3
```

三维示例推进至 `t=0.1` 的质量变化约 `-1.78e-15`。非定常输出包含最终场和动画，三维绘制域中心的 xy 截面。
