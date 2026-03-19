# Quamba vs Quamba2 关键差异分析

## 1. 对抗激活异常值（Outliers）的策略

### Quamba (当前实现)
- **Hadamard 变换**: 在 `out_proj` 之前应用 Hadamard 变换（但当前代码中被注释掉，第937行）
- **运行时应用**: Hadamard 变换在推理时动态应用
- **全局处理**: 对整个激活张量应用统一的 Hadamard 变换

### Quamba2 (需要实现)
- **离线 Hadamard 融合**: 在量化前离线将 Hadamard 矩阵融合到权重中
  - 输入投影：在权重输入侧融合 Hadamard
  - 输出投影：在权重两侧融合 Hadamard
- **结构化处理**: 结合排序/聚类策略，而不是单纯依赖 Hadamard

## 2. 输入与参数（B、C等）量化的粒度

### Quamba (当前实现)
- **Per-tensor 量化**: 
  - `Bs_quant` 和 `Cs_quant` 使用简单的 `QAct`，对整个张量使用统一的量化刻度
  - 代码：`self.Bs_quant(Bs)` 和 `self.Cs_quant(Cs)` 都是 per-tensor
- **静态量化**: 使用校准数据计算固定的 scales/base

### Quamba2 (需要实现)
- **输入排序与聚类**:
  1. 根据校准数据集中通道的最大值对输入通道排序
  2. 使用聚类算法将相似通道分组
  3. 对每个聚类组分别计算量化参数
- **Per-state-group 量化**:
  - 将状态矩阵（Bs, Cs）的通道分为多个组
  - 每个组使用独立的量化刻度
  - 形状：Bs/Cs 是 `(B, K, d_state, L)`，需要按 `d_state` 维度分组
- **离线权重重排**: 根据聚类结果离线重排权重，保证计算结果不变

## 3. 代码实现差异

### 当前 quamba_arch.py 中的问题：

1. **Bs/Cs 量化** (第899行):
```python
# 当前：per-tensor 量化
self.Bs_quant(Bs), self.Cs_quant(Cs)
# Bs 形状: (B, K, d_state, L)
# Cs 形状: (B, K, d_state, L)
```

2. **Hadamard 使用** (第937行):
```python
# 当前：被注释掉，没有使用
# out = self.out_proj(self.out_proj_quant(self.out_proj_had(y)))
out = self.out_proj(self.out_proj_quant(y))
```

### Quamba2 需要的改进：

1. **Per-state-group QAct**:
   - 需要新的 `QActPerStateGroup` 类，支持按状态组量化
   - 对 Bs/Cs 的 `d_state` 维度分组量化

2. **输入排序/聚类**:
   - 在校准阶段对输入通道排序
   - 应用聚类算法
   - 根据聚类结果配置量化器

3. **离线 Hadamard 融合**:
   - 在量化前对权重应用 Hadamard 变换
   - 移除运行时的 `out_proj_had`

## 4. 建议的实现步骤

1. 创建 `QActPerStateGroup` 类，支持按状态组量化
2. 实现输入通道排序和聚类功能
3. 修改 `QSS2D` 的 `forward_core`，使用 per-state-group 量化
4. 实现离线 Hadamard 权重融合
5. 更新校准流程以支持排序/聚类

