# PU-Diff

《原型不确定性驱动潜在扩散的半监督医学图像分割》的 PyTorch 方法实现。
主分割网络使用 U-Net，训练辅助分支由语义原型不确定性量化（SPUQ）和不确定性感知扩散校正（UADR）组成。
测试时仅加载主 U-Net。

仓库包含数据转换、划分列表准备、训练、断点恢复、带标签评价和不依赖标签的独立预测入口。

本次修改未运行代码、训练、推理、测试或性能验证。这里提供的是按给定论文整理的源码，不能据此声称复现了论文中的实验数值。

## 方法与源码对应

| 论文部分 | 实现位置 | 实现内容 |
| --- | --- | --- |
| 3.1，式（1）—（2） | `networks/spuq.py` | 按弱视图最大类别概率，以 `>= tau` 与 `< tau` 解耦高、低置信度 one-hot 标签，排除掩膜区域被误当作背景的情况 |
| 3.2，式（3）—（4） | `networks/spuq.py` | 主 U-Net 最终分类卷积前的特征；跨批次的 EMA 类别原型；首次出现初始化，当前批次缺失则沿用历史值；不反传原型更新 |
| 3.2，式（5）—（7） | `networks/spuq.py` | 欧氏距离、Softmax、香农熵、逐图 Min-Max 归一化 |
| 3.3，式（8） | `networks/uadr.py` | 原图与不确定性图拼接，经专用编码器生成 64×64 条件嵌入 |
| 3.3，式（9）—（10） | `networks/uadr.py` | 低置信度标签的潜在编码、余弦调度正向加噪，以及完整的条件高斯反向采样 |
| 3.4，式（11） | `train_PU_Diff.py` | `Lsup + lambda_unsup * Lunsup + lambda_lat * Llat + lambda_rect * Lrect` |
| 3.4，式（12）—（13）、（15） | `utils/losses.py` | CE + Dice：有标签监督、弱强一致性、校正标签对弱视图的反馈 |
| 3.4，式（14） | `networks/uadr.py` | 随机时间步下直接预测干净潜在状态 z0，使用 MSE；不使用噪声预测目标 |
| 4.2，监督预热 | `train_PU_Diff.py` | 第 1—1000 次迭代只对有标签图像前向和监督优化；第 1001 次才开始无标签数据流、原型更新与扩散分支 |
| 4.2，测试 | `test_2D.py`、`utils/evaluation.py` | 仅主网络预测，输出前景各类与均值 DSC、IoU、HD95、ASD |

运行配套：`dataloaders/io.py` 统一读取数据与元信息；`utils/runtime.py` 处理设备、原子写入及检查点；`utils/inference.py` 供评价与无标签预测复用。入口之间不依赖外部研究项目或个人服务器路径。

已移除旧版判别器、对抗训练、DGDR、HSH-PC、原型对比损失、BoundaryDoU、Gamma 调制、LDE、其他 U-Net 变体以及原有通用 guided-diffusion 库。保留的方法不依赖这些模块。

## 论文原式及未明确细节

以下条目是阅读所给 PDF 后必须说明的实现边界，不应当被当作论文已经给出的实验配置。

1. **式（5）的符号**：给定 PDF 印刷的是 `Sc = Softmax(||F - Pc||_2)`，指数中的距离前没有负号。本实现严格保留正号，没有改成常用的 `Softmax(-distance)`。该原式与通常的“近距离具有高相似度”解释存在差异，正式发布时应与论文作者确认是否为排版问题。
2. **原型冷启动**：论文规定首次出现时初始化，但没有说明从未出现过的类别如何参加距离计算。本实现仅在已初始化类别间计算分布；没有有效类别时返回零不确定性图，仅一个有效类别时熵不提供区分信息。缺失类别不使用未初始化的零向量充当语义原型。
3. **潜在编码与解码**：论文指定 64×64 空间压缩，但没有定义标签编码器/解码器的结构、预训练权重或其重构训练损失。本实现采用固定的 one-hot 面积平均下采样作为编码、双线性上采样与 argmax 作为解码，潜在形状为 `[B,K,64,64]`。条件编码器和去噪 U-Net 可学习，由 `Llat` 更新；不额外添加 VAE、标签自编码或像素重构损失。此选择不是论文给定的唯一网络结构。
4. **反向分布**：论文式（10）未具体规定方差参数化。本实现采用标准 DDPM 固定后验方差，由预测的 z0 计算后验均值；以 `q(zT | z0)` 而非独立纯噪声开始，完整执行 50 步反向过程，无 DDIM 步数压缩。余弦调度偏置采用 0.008，并将 beta 上限截为 0.999。
5. **完整校正目标**：只有低置信度困难标签输入扩散；将生成的低置信度区域与保留的高置信度标签合成完整 `YR`，再计算式（15）。这样不会将未输入扩散的高置信度区域错误地解释为背景。论文未单列此合成公式。
6. **梯度方向**：弱视图 argmax 是一致性目标，强视图预测接收 `Lunsup` 梯度；`YR` 是停止梯度的伪真值，弱视图预测接收 `Lrect` 梯度。SPUQ、初始标签与图像条件不会把扩散梯度传回主网络；`Llat` 只优化 UADR。不存在判别器或额外的对抗目标。
7. **空间对齐**：先取得共同的原始裁剪视野；弱视图做随机翻转，强视图增加大角度旋转、颜色/亮度抖动和高斯模糊。弱标签同步旋转到强视图，旋转新增的填充区域不计入一致性损失。小图补边通过有效像素掩膜从监督损失、原型统计、不确定性归一化和校正损失中排除。进入 UADR 前将弱视图的不确定性图和标签反翻转到原图坐标，输出后再映射回弱视图。原图条件没有强增强。
8. **其余实现选择**：EMA 动量默认 0.95；三个损失权重均默认 1；扩散分支默认 Adam、学习率 0.001；时间嵌入为正弦余弦编码与 MLP；条件通道为 64。论文没有报告这些数值/结构。主 U-Net 保留原项目的 `[16,32,64,128,256]` 通道配置，返回最终分类前的 16 通道特征；去噪 U-Net 按论文指定使用 `[64,128,256,512]`。没有引入论文未说明的学习率衰减或第二段扩散预热。

## 环境

使用 Python 3.10 或以上版本。按运行服务器的 CUDA 环境安装相互匹配的 PyTorch 与 torchvision，再安装其余依赖：

```bash
pip install -r requirements.txt
```

`SimpleITK` 仅用于可选的 NIfTI 数据转换。训练和评价不依赖 MedPy 或旧版 NumPy 全局别名补丁。
依赖尚未在本次修改中安装或执行验证；`requirements.txt` 是最低版本范围，不是已验证的环境锁文件。

`--device auto` 为默认设置：有可用 CUDA 时使用 GPU，否则使用 CPU。可显式指定 `--device cuda:0` 或 `--device cpu`；显式要求 CUDA 而环境不支持时会给出错误。CPU 可用于运行入口，但完整 50 步扩散训练的计算开销较大。

默认数据和输出目录相对于仓库根目录解析；显式传入的相对路径相对于启动命令的当前目录解析。命令示例默认在仓库根目录执行。Windows 多进程加载由 `if __name__ == "__main__"` 入口保护；需要单进程加载时使用 `--num_workers 0`。

## 数据

原有 `datasets/ACDC/*.list` 保留。未重新随机划分训练、验证或测试集。

### HDF5：ACDC / MS-CMRSeg / MSD Prostate

```text
datasets/ACDC/
  train.list                # 训练体积名称，不含 .h5，例如 patient099_frame01
  train_slices.list         # 训练切片名称，不含 .h5
  val.list
  test.list
  data/
    patient099_frame01.h5
    slices/
      patient099_frame01_slice_0.h5
```

切片文件：`image` 为 `[H,W]`、`[C,H,W]` 或 `[H,W,C]`，`label` 为 `[H,W]` 的类别索引。
体积文件：灰度 `image` 为 `[D,H,W]`，多通道为 `[D,C,H,W]`，`label` 为 `[D,H,W]`。
体积可提供 `spacing` 数据集，按数组轴顺序 `[z,y,x]`，单位 mm。
新转换的体积另保存 `origin`（物理坐标 x,y,z）、`direction`（3×3）以及 `is_volume` 属性，便于预测结果还原为空间一致的 NIfTI。
HDF5 列表允许包含或省略 `.h5` 后缀，训练切片兼容 `data/slices/` 与旧版 `data/` 布局。训练有标签文件必须含有类别索引标签；无标签文件允许只有 `image`。读取时会检查影像有限性、标签尺寸及类别编号。
已归一化至 `[0,1]` 的影像保留原有尺度；其他 HDF5 影像执行 Min-Max 归一化。
推荐先按整幅体积归一化再切片，以保持训练和体积评价的一致性。

可选 NIfTI 转换命令（标签默认是同名加 `_gt`）：

```bash
python prepare_data.py --image_dir /data/ACDC/images --label_dir /data/ACDC/labels --output_dir ./datasets/ACDC
```

转换不会生成或覆盖划分列表。多通道 NIfTI 加 `--img_channels 2`；训练同样指定通道数。训练切片需与列表名称对应，类别编号为 `0..K-1`。

ACDC 原始数据按患者目录组织、影像与 `_gt` 标签放在一起时：

```bash
python prepare_data.py --image_dir /data/ACDC/training --recursive --output_dir ./datasets/ACDC
python prepare_splits.py --root_path ./datasets/ACDC --dataset acdc
```

转换会跳过 `_gt` 标签文件和 `_4d` 时间序列文件。没有真实标签的独立影像可以加 `--allow_unlabeled` 转成 image-only HDF5。
已有同名输出默认报错，显式加 `--overwrite` 才覆盖。

### 二维 RGB：Kvasir-SEG

```text
datasets/Kvasir-SEG/
  train.list                # 每行包含扩展名，例如 image001.jpg
  val.list
  test.list
  images/image001.jpg
  masks/image001.png        # 同名 PNG，二值 0/1 或 0/255
```

使用 `--data_format images --img_channels 3 --num_classes 2`。其他数据集需自行提供论文规定的固定划分；本仓库只附带原有 ACDC 列表。

优先读取同名 PNG/TIFF/BMP 掩膜，也兼容同名同扩展名掩膜。0/1 掩膜按大于 0 处理，其余二值掩膜按大于 127 处理，避免 JPEG 零值附近压缩噪声被误认为前景。

### 生成或复用划分列表

```bash
python prepare_splits.py --root_path ./datasets/Kvasir-SEG --dataset kvasir
```

已有 `train.list`、`val.list`、`test.list` 时默认保留三者，只补充 HDF5 所需的 `train_slices.list`；已有训练切片列表若样本集合相同，也保留其原始顺序。检测到同一患者跨训练/验证/测试集时停止并给出具体错误。

新数据没有划分列表时，使用独立的固定随机种子按患者/病例分组划分，ACDC 不会将同一患者的不同帧分到不同集合。默认病例数为：

| `--dataset` | 训练 | 验证 | 测试 |
| --- | --- | --- | --- |
| `acdc` | 70 | 10 | 20 |
| `mscmrseg` | 30 | 5 | 10 |
| `prostate` | 32 | 0 | 16 |
| `kvasir` | 800 | 100 | 100 |

其他规模通过 `--counts 训练数 验证数 测试数` 指定，三个数量之和必须等于实际患者/病例总数。使用 `--seed` 固定生成结果；已有列表需要重新生成时必须显式使用 `--overwrite`。新生成的列表属于本次数据准备配置，不能当作论文未公开的原始病例划分。

### 标注划分

默认按 `train.list` 的顺序选取前 `--labeled_num` 个患者/病例，ACDC 同一患者的不同帧合并计数。
不再用硬编码的“患者数量→切片数量”表，防止切片顺序变化时发生标注泄漏。
每次训练保存 `partition.json`，记录实际使用的有标签/无标签样本名称；无标签样本不读取真实标签。

如果需要精确复用论文实验中的标注子集，使用 `--labeled_list /path/to/labeled_cases.list` 明确指定患者名、体积名或切片名。给定论文没有列出其具体标注患者、三个随机种子及全部划分，不能仅凭默认的前若干病例认定复现了该实验。

## 训练

默认：256×256 输入、批量 12（半监督阶段 6 有标签 + 6 无标签）、30000 次迭代、1000 次纯监督预热、阈值 0.8、64×64 潜在空间、50 步余弦扩散。主网络 SGD，学习率 0.001、动量 0.9、权重衰减 0.0001。预热的 12 个样本全部来自有标签集合。

数据集预设提供可覆盖的通道数、类别数、默认目录和标注病例数：

| `--dataset` | 默认目录 | 输入通道 | 类别数 | 默认标注病例数 |
| --- | --- | --- | --- | --- |
| `acdc` | `datasets/ACDC` | 1 | 4 | 7 |
| `mscmrseg` | `datasets/MS-CMRSeg` | 1 | 4 | 3 |
| `prostate` | `datasets/MSD-Prostate` | 2 | 3 | 3 |
| `kvasir` | `datasets/Kvasir-SEG` | 3 | 2 | 80 |

这些标注数量是训练集约 10% 的使用示例，不代表论文指定的病例身份。使用其他标注率时明确传入 `--labeled_num` 或 `--labeled_list`。Prostate 预设没有验证集，默认关闭训练期间验证；其他预设默认每 200 次迭代评价验证集。

```bash
python train_PU_Diff.py --root_path ./datasets/ACDC --labeled_num 7 --output_dir ./outputs/PU-Diff/ACDC_7_labeled
```

使用其他数据集预设：

```bash
python train_PU_Diff.py --dataset mscmrseg --root_path /data/MS-CMRSeg --output_dir ./outputs/PU-Diff/MS-CMRSeg_run1
python train_PU_Diff.py --dataset prostate --root_path /data/MSD-Prostate --output_dir ./outputs/PU-Diff/Prostate_run1
python train_PU_Diff.py --dataset kvasir --root_path /data/Kvasir-SEG --output_dir ./outputs/PU-Diff/Kvasir_run1
```

自定义损失权重与明确的标注划分：

```bash
python train_PU_Diff.py --root_path ./datasets/ACDC --labeled_list /data/splits/labeled_cases.list --lambda_unsup 1 --lambda_lat 1 --lambda_rect 1 --ema_decay 0.95 --seed 1337 --output_dir ./outputs/PU-Diff/ACDC_custom
```

不执行训练期间的验证：

```bash
python train_PU_Diff.py --root_path ./datasets/ACDC --val_interval 0 --output_dir ./outputs/PU-Diff/ACDC_no_val
```

恢复训练：

```bash
python train_PU_Diff.py --resume ./outputs/PU-Diff/ACDC_7_labeled/checkpoint_last.pth
```

完整训练检查点保存主网络、SPUQ 原型及初始化标记、UADR、两个优化器、迭代计数、全局随机状态、三个数据流采样器的随机状态和实际标注划分。`--resume` 自动读取保存的配置，无需重复填写架构及损失参数；明确传入的命令行参数可覆盖保存值，涉及方法/架构变化时会拒绝混用。
迁移数据目录可以通过 `--root_path`、`--labeled_list` 指定新路径，但实际样本名称与标注划分必须一致。新检查点不依赖旁边的 `partition.json` 才能恢复；旧格式检查点仍读取该文件。恢复导出的 U-Net 权重不能代替完整训练检查点。
恢复后会重新建立数据加载迭代器，不保存工作进程中尚未消费的预取队列，因此不保证与中断前逐样本完全一致。
`unet_best.pth`（启用验证时）和 `unet_last.pth` 仅包含主网络权重与配置，供独立推理使用。
日志和模型存入所指定输出目录，不写入源码目录或数据目录。
检查点和 JSON 通过临时文件写入后原子替换；JSON 中的 `inf` / `nan` 保存为字符串。新训练遇到已有 `config.json` 时拒绝覆盖，使用新输出目录或明确恢复训练。训练输出还包括 TensorBoard、`validation_last.json` 和 `validation_best.json`（开启验证时）。

## 评价

```bash
python test_2D.py --root_path ./datasets/ACDC --ckpt ./outputs/PU-Diff/ACDC_7_labeled/unet_best.pth --split test --output_dir ./outputs/PU-Diff/ACDC_7_labeled/test_predictions
```

禁用训练验证时，可将检查点替换为 `unet_last.pth`。
输出每例 `.npy` 预测和 `metrics_test.json`；只实例化 U-Net。验证仅读取 `val.list`，测试仅在此命令明确指定时读取 `test.list`，不参与训练选模。
评价默认从检查点读取数据根目录，也可以用 `--root_path` 迁移；通过 `--batch_size` 控制同时处理的切片数。二维标签额外保存类别索引 PNG，具有完整空间元信息的三维体积额外保存 NIfTI。重复输出默认报错，指定 `--overwrite` 后替换。

DSC/IoU 输出为 0—1 的原始比例。HD95 使用双向表面距离合并后的第 95 百分位；ASD 为预测表面至真实表面的平均距离。
有真实体素间距时按 mm 计算，否则按单位体素/像素计算并标注单位，不将无间距结果标作 mm。
双方均为空时 DSC/IoU 为 1，距离为 0；只有一方为空时 DSC/IoU 为 0，距离为无穷大，JSON 使用字符串 `"inf"` 保存。上述评价约定在论文中未进一步指定。
若同一集合中部分病例含真实间距、部分没有，逐例报告各自单位，不混合平均 mm 和像素距离，集合距离均值记为 `"nan"`。

## 无标签独立预测

无需 `label`、掩膜目录或划分列表，只需已训练的 U-Net 权重和输入影像：

```bash
python predict.py --ckpt ./outputs/PU-Diff/ACDC_7_labeled/unet_best.pth --input /data/example.nii.gz --output_dir ./outputs/predict_case
python predict.py --ckpt ./outputs/PU-Diff/ACDC_7_labeled/unet_last.pth --input /data/unlabeled_h5 --output_dir ./outputs/predict_h5 --batch_size 4
```

支持单文件或包含图像/HDF5/NIfTI 的目录；目录按第一层文件枚举，避免把患者目录中的标签一起递归当作输入。二维 RGB 输入要求使用以 3 通道训练的权重，医学灰度/多序列输入要求与检查点通道数一致。
所有结果保存无损类别索引 `.npy`；二维影像增加 `.png`，有完整几何元信息的三维影像增加 `.nii.gz`，另有 `predictions.json` 记录输入、尺寸和输出路径。PNG 是类别编号，不是彩色可视化或自动放大的二值掩膜。
旧 HDF5 的 `[1,H,W]` 等形状存在切片/体积歧义时，可用 `--layout slice` 或 `--layout volume` 明确指定；新转换文件通过 `is_volume` 属性区分。
预测使用保存的输入尺寸对切片做双线性缩放，输出 logits 恢复到原始大小后 argmax；不裁掉原图视野，不使用辅助扩散分支。覆盖已有预测需显式指定 `--overwrite`。

## 入口清单

| 文件 | 用途 |
| --- | --- |
| `prepare_data.py` | NIfTI 转换为体积及切片 HDF5 |
| `prepare_splits.py` | 创建或复用固定病例划分、准备训练切片列表 |
| `train_PU_Diff.py` | 完整监督预热、半监督训练及断点恢复 |
| `test_2D.py` | 带标签的验证/测试指标计算 |
| `predict.py` | 不需要真实标签的独立推理 |

## 发布

源码入口为 `train_PU_Diff.py`，旧名称 `train_DHPCL_ACDC.py` 不再保留。
`.gitignore` 排除模型、训练输出、实际影像数据和 Python 缓存，仅保留数据划分列表。
旧源码已删除；工作区中遗留的 `__pycache__` 缓存删除被工具策略拦截，已通过忽略规则排除出 Git 发布范围。
方法源码、本说明和依赖清单可用于后续 GitHub 整理；当前修改不包含 Git 初始化、提交或推送，也不包含已训练权重。
