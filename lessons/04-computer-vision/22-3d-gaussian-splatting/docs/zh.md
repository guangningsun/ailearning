# 从零实现 3D 高斯泼溅

> 一个场景是数百万个3D 高斯的集合。每个高斯都有位置、朝向、缩放、不透明度和取决于视角的颜色。将其光栅化，通过光栅化反向传播，完成了。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 4 第 13 课（3D 视觉与 NeRF）、阶段 1 第 12 课（张量运算）、阶段 4 第 10 课（扩散基础，可选）
**时间：** 约 90 分钟

## 学习目标

- 解释为什么 3D 高斯泼溅在 2026 年取代 NeRF 成为逼真 3D 重建的生产级默认方案
- 说明每个高斯的六个参数（位置、旋转四元数、缩放、不透明度、球谐颜色、可选特征）以及各自占用的浮点数
- 从零实现 2D 高斯泼溅光栅化器，使用 `alpha` 合成，然后展示3D 情况如何投射到同样的循环
- 使用 `nerfstudio`、`gsplat` 或 `SuperSplat` 从 20-50 张照片重建场景，并导出为 `KHR_gaussian_splatting` glTF 扩展或 OpenUSD 26.03 的 `UsdVolParticleField3DGaussianSplat`模式

## 问题

NeRF 将场景存储为 MLP 的权重。每条渲染光线上的每个像素都是数百次 MLP 查询。训练需要数小时，渲染需要秒，且权重无法编辑——如果你想移动场景中的椅子，就必须重新训练。

3D 高斯泼溅（Kerbl, Kopanas, Leimkühler, Drettakis, SIGGRAPH 2023）取代了所有这些。场景是一组显式的 3D 高斯。渲染是 GPU 光栅化，达 100+ fps。训练只需几分钟。编辑是直接的：平移一部分高斯，你就移动了椅子。到 2026 年，Khronos Group 已批准了高斯泼溅的 glTF 扩展，OpenUSD 26.03 发布了高斯泼溅模式，Zillow 和 Apartments.com 用它渲染房产，大多数新的 3D 重建研究论文都是核心 3DGS 思想的变体。

心智模型很简单，但数学有一定复杂度——大多数介绍都是从光栅化开始，然后跳过投影和球谐函数。本课构建完整的内容——先实现 2D 版本，然后扩展到 3D。

## 概念

### 一个高斯携带什么

一个 3D 高斯是空间中的一个参数化 blob，具有以下属性：

```
position         mu         (3,)    世界坐标中的中心
rotation         q          (4,)    编码朝向的单位四元数
scale            s          (3,)    每轴的对数缩放（在渲染时取指数）
opacity          alpha      (1,)    Sigmoid 后不透明度 [0, 1]
SH coefficients  c_lm       (3 * (L+1)^2,)   视角相关颜色
```

旋转 + 缩放构建一个 3x3 协方差：`Sigma = R S S^T R^T`。这就是 3D 中高斯的形状。球谐函数让颜色随视角变化——镜面高光、微妙的光泽、视角相关的发光——而无需存储每视角纹理。使用 SH 次数 3，你得到每个颜色通道 16 个系数，仅颜色部分每个高斯就有 48 个浮点数。

一个场景通常有 1-5 百万个高斯。每个存储约 60 个浮点数（3 + 4 + 3 + 1 + 48 + 其他）。五百万高斯的场景是 240 MB——远小于带每点纹理的等效点云，比高分辨率重新渲染的 NeRF 的 MLP 权重小一个数量级。

### 光栅化，而不是光线行进

```mermaid
flowchart LR
    SCENE["数百万个 3D 高斯<br/>(位置、旋转、缩放、<br/>不透明度、SH 颜色)"] --> PROJ["投影到 2D<br/>(相机外参 + 内参)"]
    PROJ --> TILES["分配到瓦片<br/>(16x16 屏幕空间)"]
    TILES --> SORT["每个瓦片内深度排序"]
    SORT --> ALPHA["Alpha 合成<br/>从前到后"]
    ALPHA --> PIX["像素颜色"]

    style SCENE fill:#dbeafe,stroke:#2563eb
    style ALPHA fill:#fef3c7,stroke:#d97706
    style PIX fill:#dcfce7,stroke:#16a34a
```

五步，全部 GPU 友好。每个像素不涉及 MLP 查询。一块 RTX 3080 Ti 以 147 fps 渲染 6百万个 splat。

### 投影步骤

世界位置 `mu` 处的 3D 高斯，带 3D 协方差 `Sigma`，投影到屏幕位置 `mu'` 处、2D 协方差 `Sigma'` 的 2D 高斯：

```
mu' = project(mu)
Sigma' = J W Sigma W^T J^T          (2 x 2)

W = 视图变换（相机旋转 + 平移）
J = mu' 处透视投影的雅可比矩阵
```

2D 高斯的覆盖范围是一个椭圆，其轴是 `Sigma'` 的特征向量。椭圆内的每个像素接收高斯的贡献，加权为 `exp(-0.5 * (p - mu')^T Sigma'^-1 (p - mu'))`。

### Alpha 合成规则

对于一个像素，覆盖它的高斯从前到后排序（或等效地从后到前，用反转公式）。颜色的合成与 1980 年代以来每个半透明光栅化器使用的方程相同：

```
C_pixel = sum_i alpha_i * T_i * c_i

T_i = prod_{j < i} (1 - alpha_j)       到 i 处的透射率
alpha_i = opacity_i * exp(-0.5 * d^T Sigma'^-1 d)   局部贡献
c_i = eval_SH(SH_i, view_direction)    视角相关颜色
```

这**与 NeRF 的体积渲染方程相同**，只是现在是在一条射线上的一组显式稀疏高斯上积分，而不是密集采样。这就是为什么渲染质量与 NeRF 相当——两者都在积分相同的辐射场方程。

### 为什么这是可微的

每一步——投影、瓦片分配、alpha 合成、SH 求值——相对于高斯参数都是可微的。给定真值图像，计算渲染像素损失，通过光栅化器反向传播，用梯度下降更新所有 `(mu, q, s, alpha, c_lm)`。经过约 30,000 次迭代，高斯找到了它们正确的位置、缩放和颜色。

### 致密化与剪枝

固定的高斯集合无法覆盖复杂场景。训练包含两种自适应机制：

- **克隆（Clone）** — 当高斯的梯度幅值大但缩放小时，在其当前位置克隆一个高斯——重建在这里需要更多细节。
- **分裂（Split）** — 当大缩放高斯的梯度高时，将其分裂为两个较小的高斯——一个大高斯太平滑，无法拟合该区域。
- **剪枝（Prune）** — 移除不透明度降到阈值以下的高斯——它们没有贡献。

致密化每 N 次迭代运行一次。一个场景通常从约 100k 个初始高斯（从 SfM 点播种）增长到训练结束时的 1-5M。

### 一段话说清球谐函数

视角相关颜色是单位球面上的函数 `c(direction)`。球谐函数是球面上的傅里叶基。截断到次数 `L` 得到每通道 `(L+1)^2` 个基函数。对于新视角求值颜色，就是学习到的 SH 系数与在视角方向上求值的基之间的点积。0 次 = 一个系数 = 常数颜色。3 次 = 16 个系数 = 足以捕捉 Lambertian 着色、镜面反射和轻度反射。SD Gaussian Splatting 论文默认使用 3 次。

### 2026 年生产技术栈

```
1. 采集        智能手机 / DJI 无人机 / 手持扫描仪
2. SfM / MVS COLMAP 或 GLOMAP 推导相机姿态 + 稀疏点
3. 训练 3DGS    nerfstudio / gsplat / inria 官方 / PostShot（RTX 4090 上约 10-30 分钟）
4. 编辑         SuperSplat / SplatForge（清理浮点数、分段）
5. 导出         .ply -> glTF KHR_gaussian_splatting 或 .usd（OpenUSD 26.03）
6. 查看         Cesium / Unreal / Babylon.js / Three.js / Vision Pro
```

### 4D 和生成式变体

- **4D 高斯泼溅** — 高斯是时间的函数；用于体积视频（Superman 2026, A$AP Rocky 的 "Helicopter"）。
- **生成式 splat** — 文生 splat 模型（World Labs 的 Marble）可以幻想出整个场景。
- **3D 高斯无损变换** — NVIDIA NuRec 的变体，用于自动驾驶仿真。

## 动手实现

### 第 1 步：2D 高斯

我们首先构建一个 2D 光栅化器。3D 情况在投影后归结为它。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def eval_2d_gaussian(means, covs, points):
    """
    means:  (G, 2)      中心点
    covs:   (G, 2, 2)   协方差矩阵
    points: (H, W, 2)   像素坐标
    returns: (G, H, W)  每个高斯在每个像素处的密度
    """
    G = means.size(0)
    H, W, _ = points.shape
    flat = points.view(-1, 2)
    inv = torch.linalg.inv(covs)
    diff = flat[None, :, :] - means[:, None, :]
    d = torch.einsum("gpi,gij,gpj->gp", diff, inv, diff)
    density = torch.exp(-0.5 * d)
    return density.view(G, H, W)
```

`einsum` 对每个（高斯，像素）对执行二次型 `diff^T Sigma^-1 diff`。

### 第 2 步：2D splatting 光栅化器

从前到后的 alpha 合成。2D 中深度没有意义，所以我们使用一个学习到的每高斯标量来排序。

```python
def rasterise_2d(means, covs, colours, opacities, depths, image_size):
    """
    means:     (G, 2)
    covs:      (G, 2, 2)
    colours:   (G, 3)
    opacities: (G,)     在 [0, 1] 范围内
    depths:    (G,)     用于排序的每高斯标量
    image_size: (H, W)
    returns:   (H, W, 3) 渲染图像
    """
    H, W = image_size
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=means.device),
        torch.arange(W, dtype=torch.float32, device=means.device),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1)

    densities = eval_2d_gaussian(means, covs, points)
    alphas = opacities[:, None, None] * densities
    alphas = alphas.clamp(0.0, 0.99)

    order = torch.argsort(depths)
    alphas = alphas[order]
    colours_sorted = colours[order]

    T = torch.ones(H, W, device=means.device)
    out = torch.zeros(H, W, 3, device=means.device)
    for i in range(means.size(0)):
        a = alphas[i]
        out += (T * a)[..., None] * colours_sorted[i][None, None, :]
        T = T * (1.0 - a)
    return out
```

不快——真正的实现使用基于瓦片的 CUDA 内核——但数学完全正确且完全可微。

### 第 3 步：可训练的 2D splat 场景

```python
class Splats2D(nn.Module):
    def __init__(self, num_splats=128, image_size=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        H, W = image_size, image_size
        self.means = nn.Parameter(torch.rand(num_splats, 2, generator=g) * torch.tensor([W, H]))
        self.log_scale = nn.Parameter(torch.ones(num_splats, 2) * math.log(2.0))
        self.rot = nn.Parameter(torch.zeros(num_splats))  # 2D 中单个角度
        self.colour_logits = nn.Parameter(torch.randn(num_splats, 3, generator=g) * 0.5)
        self.opacity_logit = nn.Parameter(torch.zeros(num_splats))
        self.depth = nn.Parameter(torch.rand(num_splats, generator=g))

    def covs(self):
        s = torch.exp(self.log_scale)
        c, si = torch.cos(self.rot), torch.sin(self.rot)
        R = torch.stack([
            torch.stack([c, -si], dim=-1),
            torch.stack([si, c], dim=-1),
        ], dim=-2)
        S = torch.diag_embed(s ** 2)
        return R @ S @ R.transpose(-1, -2)

    def forward(self, image_size):
        covs = self.covs()
        colours = torch.sigmoid(self.colour_logits)
        opacities = torch.sigmoid(self.opacity_logit)
        return rasterise_2d(self.means, covs, colours, opacities, self.depth, image_size)
```

`log_scale`、`opacity_logit` 和 `colour_logits` 都是无约束参数，在渲染时通过正确的激活函数映射。这是每个 3DGS 实现的的标准模式。

### 第 4 步：将 2D 高斯拟合到目标图像

```python
import math
import numpy as np

def make_target(size=64):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.zeros((size, size, 3), dtype=np.float32)
    # 红色圆
    mask = (xx - 20) ** 2 + (yy - 20) ** 2 < 10 ** 2
    img[mask] = [1.0, 0.2, 0.2]
    # 蓝色方块
    mask = (np.abs(xx - 45) < 8) & (np.abs(yy - 40) < 8)
    img[mask] = [0.2, 0.3, 1.0]
    return torch.from_numpy(img)


target = make_target(64)
model = Splats2D(num_splats=64, image_size=64)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(200):
    pred = model((64, 64))
    loss = F.mse_loss(pred, target)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 40 == 0:
        print(f"step {step:3d}  mse {loss.item():.4f}")
```

经过 200 步，64 个高斯沉淀为两个形状。这就是全部思想——在显式几何原语上进行梯度下降。

### 第 5 步：从 2D 到 3D

3D 扩展保持相同的循环。新增的部分：

1. 每个高斯的旋转是一个四元数，而不是单个角度。
2. 协方差是 `R S S^T R^T`，其中 `R` 由四元数构建，`S = diag(exp(log_scale))`。
3. 投影 `(mu, Sigma) -> (mu', Sigma')` 使用相机外参和 `mu` 处透视投影的雅可比矩阵。
4. 颜色变为球谐展开；在视角方向上求值。
5. 深度排序基于实际相机空间 z，而不是学习到的标量。

每个生产实现（`gsplat`、`inria/gaussian-splatting`、`nerfstudio`）在 GPU 上用基于瓦片的 CUDA 内核完全按照这样做。

### 第 6 步：球谐函数求值

SH 基最多到 3 次，每通道有 16 项。求值：

```python
def eval_sh_degree_3(sh_coeffs, dirs):
    """
    sh_coeffs: (..., 16, 3)  最后一维是 RGB 通道
    dirs:      (..., 3)       单位向量
    returns:   (..., 3)
    """
    C0 = 0.282094791773878
    C1 = 0.488602511902920
    C2 = [1.092548430592079, 1.092548430592079,
          0.315391565252520, 1.092548430592079,
          0.546274215296039]
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result = C0 * sh_coeffs[..., 0, :]
    result = result - C1 * y[..., None] * sh_coeffs[..., 1, :]
    result = result + C1 * z[..., None] * sh_coeffs[..., 2, :]
    result = result - C1 * x[..., None] * sh_coeffs[..., 3, :]

    result = result + C2[0] * xy[..., None] * sh_coeffs[..., 4, :]
    result = result + C2[1] * yz[..., None] * sh_coeffs[..., 5, :]
    result = result + C2[2] * (2.0 * z2 - x2 - y2)[..., None] * sh_coeffs[..., 6, :]
    result = result + C2[3] * xz[..., None] * sh_coeffs[..., 7, :]
    result = result + C2[4] * (x2 - y2)[..., None] * sh_coeffs[..., 8, :]

    # 为简洁起见省略了 3 次项；完整 16 系数版本在代码文件中
    return result
```

学习到的 `sh_coeffs` 存储该高斯"每个方向的颜色"。在渲染时，根据当前视角方向求值，得到一个 3 维 RGB 向量。

## 实际使用

对于真正的 3DGS 工作，使用 `gsplat`（Meta）或 `nerfstudio`：

```bash
pip install nerfstudio gsplat
ns-download-data example
ns-train splatfacto --data path/to/data
```

`splatfacto` 是 nerfstudio 的 3DGS 训练器。在 RTX 4090 上，一个典型场景的训练需要 10-30 分钟。

2026 年重要的导出选项：

- `.ply` — 原始高斯云（可移植，文件最大）。
- `.splat` — PlayCanvas / SuperSplat 量化格式。
- glTF `KHR_gaussian_splatting` — Khronos 标准，可在查看器和引擎间移植（2026 年 2 月 RC）。
- OpenUSD `UsdVolParticleField3DGaussianSplat` — USD 原生，用于 NVIDIA Omniverse 和 Vision Pro 流程。

对于 4D / 动态场景，`4DGS` 和 `Deformable-3DGS` 用随时间变化的均值和不透明度扩展了相同的机制。

## 交付物

本课产出：

- `outputs/prompt-3dgs-capture-planner.md` — 一个提示词，为给定场景类型规划采集会话（照片数量、相机路径、光照）。
- `outputs/skill-3dgs-export-router.md` — 一个技能，根据下游查看器或引擎选择正确的导出格式（`.ply` / `.splat` / glTF / USD）。

## 练习

1. **（简单）** 在不同的合成图像上运行上面的 2D splat 训练器。将 `num_splats` 设为 `[16, 64, 256]`，绘制每个的 MSE vs 步数曲线。找出收益递减点。
2. **（中等）** 扩展 2D 光栅化器，支持通过次数 2 的谐波依赖标量"视角"变化的每高斯 RGB 颜色。在一对目标图像上训练，并验证模型能重建两者。
3. **（困难）** 克隆 `nerfstudio` 并用 `splatfacto` 训练你拥有的任意场景的 20 张照片采集（桌子、植物、面部、房间）。导出为 glTF `KHR_gaussian_splatting`，并在查看器中打开（Three.js `GaussianSplats3D`、SuperSplat、Babylon.js V9）。报告训练时间、高斯数量和渲染 fps。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|----------------------|
| 3DGS | "高斯泼溅" | 作为数百万个 3D 高斯集合的显式场景表示，每个高斯有独立的位置、旋转、缩放、不透明度、SH 颜色 |
| 协方差（Covariance） | "高斯的形状" | `Sigma = R S S^T R^T`；一个高斯的朝向和各向异性缩放 |
| Alpha 合成（Alpha compositing） | "从前到后混合" | 与 NeRF体积渲染相同的方程，现在作用在一组显式稀疏高斯上 |
| 致密化（Densification） | "克隆和分裂" | 在重建欠拟合的地方自适应添加新高斯 |
| 剪枝（Pruning） | "删除低不透明度" | 移除在训练期间崩溃到接近零不透明度的高斯 |
| 球谐函数（Spherical harmonics） | "视角相关颜色" | 球面上的傅里叶基；将颜色存储为视角方向的函数 |
| Splatfacto | "nerfstudio 的 3DGS" | 2026 年训练 3DGS 最简单的途径 |
| `KHR_gaussian_splatting` | "glTF 标准" | Khronos 2026 年扩展，使 3DGS 可在查看器和引擎间移植 |

## 延伸阅读

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering（Kerbl et al., SIGGRAPH 2023）](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — 原始论文
- [gsplat（Meta/nerfstudio）](https://github.com/nerfstudio-project/gsplat) — 生产级 CUDA 光栅化器
- [nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) — 参考训练配方
- [Khronos KHR_gaussian_splatting 扩展](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md) — 2026 年可移植格式
- [OpenUSD 26.03 发布说明](https://openusd.org/release/) — `UsdVolParticleField3DGaussianSplat` 模式
- [THE FUTURE 3D 2026 年高斯泼溅现状](https://www.thefuture3d.com/blog-0/2026/4/4/state-of-gaussian-splatting-2026) — 行业概览