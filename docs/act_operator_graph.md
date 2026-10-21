# ACT 策略推理算子图与量化方案

本文梳理 LeRobot 中 ACT（Action Chunking Transformer）策略在**推理阶段**用到的全部算子、各算子所属模块、张量形状，以及部署到开源 GPGPU 时的建议量化精度。

- 代码：`src/lerobot/policies/act/modeling_act.py`
- 分析所用 checkpoint：`outputs/train/act_so101_pickplace_test/checkpoints/last/pretrained_model`
- 输入配置：batch=1，`observation.state` 6 维，两路相机（`up` / `side`）3×480×640，输出 action 6 维，`chunk_size=100`
- 模型超参：`resnet18` backbone，`dim_model=512`，`n_heads=8`，`dim_feedforward=3200`，`n_encoder_layers=4`，`n_decoder_layers=1`，`pre_norm=false`，`activation=relu`

算子数量和形状来自对该 checkpoint 执行 `torch.export` 后导出的 aten 图，不是手工推算（脚本见附录）。

## 1. 模型组成

| 模块 | 结构 | 推理时是否执行 |
|---|---|---|
| `backbone` | ResNet18，取到 layer4，BN 冻结（FrozenBatchNorm2d） | ✅ 每路相机各执行一次 |
| `encoder` | 4 层 Transformer encoder（post-norm） | ✅ |
| `decoder` | **1 层** Transformer decoder | ✅ |
| `vae_encoder` | 4 层 Transformer encoder | ❌ 仅训练时使用 |

推理时 latent 直接取全 0，因此 VAE encoder（约 1730 万参数，占全模型约 1/3）可以整块不部署。

参数分布（全部为 FP32，训练时 `use_amp=false`）：

| 模块 | 参数量 |
|---|---|
| backbone | 11.19 M |
| encoder | 17.33 M |
| decoder | 5.39 M |
| vae_encoder 及其投影（仅训练） | 17.42 M |
| 其他投影与位置编码 | 0.34 M |
| **合计** | **51.67 M** |

## 2. 推理数据流与张量形状

```
state(1,6) ──[Normalize: sub,div]──► Linear 6→512 ───────────────┐ token×1
latent=zeros(1,32) ─► Linear 32→512 (输出恒等于 bias，可常量折叠) ─┤ token×1
                                                                   │
img_up/side (1,3,480,640) ─[Normalize: sub,div]                    │
   └► ResNet18 (Conv+BN+ReLU+MaxPool+Add) ─► (1,512,15,20)          │
       └► Conv1x1 512→512 ─► flatten(h w) ─► 300 token/路 ×2 ──────┤ token×600
   └► 2D 正弦位置编码 (cumsum,sin,cos…，可预计算成常量)               │
                                                                   ▼
                              concat → (602,1,512) + pos_embed
                                         │
                        ┌──── Encoder ×4 (post-norm) ────┐
                        │ Q/K = x+pos ; V = x            │
                        │ Linear Q,K,V (512→512)×3       │
                        │ mul(1/√64) → BMM(QKᵀ) 8×602×602│
                        │ Softmax → BMM(·V) → Linear out │
                        │ Add → LayerNorm                │
                        │ Linear 512→3200 → ReLU         │
                        │ Linear 3200→512 → Add → LN     │
                        └───────────────┬────────────────┘
                                        │ memory (602,1,512)
decoder_in = zeros(100,1,512)           ▼
 + query_pos(100,512)   ┌──── Decoder ×1 ────────────────┐
                        │ Self-Attn: 8×100×100 → Add → LN│
                        │ Cross-Attn: Q 100 / KV 602     │
                        │   8×100×602 → Add → LN         │
                        │ FFN 512→3200→512 ReLU → Add→LN │
                        │ 最后再接一个 LayerNorm          │
                        └───────────────┬────────────────┘
                                        ▼
                     Linear 512→6 → (1,100,6) → [Unnormalize: mul,add] → action chunk
```

token 数：1（latent）+ 1（state）+ 2 × 15 × 20（两路相机特征图）= **602**。

## 3. 完整算子表

**当前精度**：checkpoint 中所有权重均为 **FP32**，所有算子目前都按 FP32 执行。最后一列是部署到 GPGPU 时的建议精度。

| # | 算子 (aten) | 所属部分 | 次数 | 典型形状 | 建议部署精度 |
|---|---|---|---|---|---|
| 1 | sub / div（MEAN_STD 归一化） | 预处理 | 3 | state(6)、image(3,480,640) | FP32/FP16 |
| 2 | conv2d 7×7 s2 | CNN stem | 2 | 3→64，480×640→240×320 | INT8 W/A（首层建议 FP16，对精度敏感） |
| 3 | FrozenBN (mul/add/rsqrt/sub) | CNN | 40 | — | 部署前折叠进 conv，不需要单独的 kernel |
| 4 | relu | CNN | 34 | — | 与 conv 融合 |
| 5 | max_pool2d 3×3 s2 | CNN stem | 2 | 64×240×320→120×160 | INT8 |
| 6 | conv2d 3×3 / 1×1 下采样 | CNN layer1–4 | 38 | 最终输出 512×15×20 | INT8 W/A（权重 per-channel） |
| 7 | add（残差） | CNN BasicBlock | 16 | — | INT8/FP16，与 conv 融合 |
| 8 | conv2d 1×1 | 图像投影 `encoder_img_feat_input_proj` | 2 | 512→512 @15×20 | INT8 |
| 9 | cumsum/sin/cos/pow/div | 2D 位置编码 | ~30 | (1,512,15,20) | 离线预计算成常量 |
| 10 | linear | state/latent 投影 | 2 | 6→512、32→512 | FP16（latent 投影可直接常量折叠） |
| 11 | cat / stack / permute / view | token 拼接 | — | (602,1,512) | 只是内存布局操作 |
| 12 | add（+pos_embed） | Enc/Dec 的 Q、K | 12+6 | — | FP16 |
| 13 | linear Q/K/V/Out | Encoder 自注意力 | 16 | (602,512)×(512,512) | INT8 W/A |
| 14 | mul（缩放 1/√d） | 注意力 | 4+2 | — | 融合进 Q 投影 |
| 15 | bmm QKᵀ | Encoder | 4 | (8,602,64)×(8,64,602) | INT8 或 FP16 |
| 16 | softmax | Encoder | 4 | (8,602,602) | FP16/FP32（不要量化） |
| 17 | bmm Attn·V | Encoder | 4 | (8,602,602)×(8,602,64) | FP16（attention 概率用 INT8 误差大） |
| 18 | linear FFN1 + relu | Encoder | 4 | 512→3200 | INT8 W/A |
| 19 | linear FFN2 | Encoder | 4 | 3200→512 | INT8 W/A（这一层激活值的离群点最多，需重点校准） |
| 20 | add（残差）+ layer_norm | Encoder | 8+8 | (602,1,512) | LN 用 FP16 计算、FP32 累加 |
| 21 | linear Q/K/V/Out | Decoder 自注意力 | 4 | (100,512)×(512,512) | INT8 |
| 22 | bmm + softmax | Decoder 自注意力 | 2+1 | (8,100,100) | FP16 |
| 23 | linear Q/K/V/Out | Decoder 交叉注意力 | 4 | Q：100，KV：602 | INT8 |
| 24 | bmm + softmax | Decoder 交叉注意力 | 2+1 | (8,100,602) | FP16 |
| 25 | linear FFN1 + relu、FFN2 | Decoder | 2 | 512↔3200 | INT8 |
| 26 | layer_norm | Decoder（3 个 + 最后 1 个） | 4 | (100,1,512) | FP16/FP32 |
| 27 | linear | action_head | 1 | 512→6 | FP16/FP32（直接决定输出精度） |
| 28 | mul / add（反归一化） | 后处理 | 1 | (100,6) | FP32 |

导出图中共有 42 个 conv2d、37 个 linear、12 个 bmm、6 个 softmax、12 个 layer_norm，其余算子都是 view、transpose、split 这类布局操作。

说明：Q/K 加了位置编码而 V 没有加，所以 `nn.MultiheadAttention` 的 in_proj 被拆成 Q、K、V 三个独立的 512→512 linear，每层注意力共 4 个 linear（含 out_proj）。

## 4. 算力分布（每帧，batch=1，粗略估算）

| 部分 | MAC 数 | 占比 |
|---|---|---|
| ResNet18 ×2（480×640） | 约 22 G | 约 63% |
| Encoder ×4（602 token） | 约 12 G（每层约 3 G，其中 FFN 约 2 G） | 约 34% |
| Decoder ×1 | 约 0.9 G | 约 3% |

在 GPGPU 上最要紧的两个 kernel 是 **Conv2d**（im2col 后走 GEMM，或写 direct conv）和 **GEMM/BMM**；其次是 Softmax 和 LayerNorm，它们是带归约的逐元素算子。

## 5. 部署前的图简化

1. FrozenBatchNorm2d 折叠进前一个 conv 的权重和 bias
2. 2D 正弦位置编码（图像）、`decoder_pos_embed`、`encoder_1d_feature_pos_embed` 预计算为常量
3. latent 恒为 0，`encoder_latent_input_proj` 的输出等于其 bias，直接常量折叠
4. 删除 Dropout（推理时为 no-op）和整个 VAE encoder
5. 注意力缩放 `1/√64` 合并进 Q 投影权重

## 6. 需要实现的 kernel 清单

1. **Conv2d**（7×7 s2、3×3 s1/s2、1×1 s1/s2），融合 bias、ReLU 和残差 Add
2. **MaxPool2d** 3×3 s2
3. **GEMM / Linear**（融合 bias，可选融合 ReLU）
4. **Batched GEMM**（QKᵀ、Attn·V）
5. **Softmax**（按行，长度 100 或 602）
6. **LayerNorm**（dim=512）
7. **逐元素 Add / Mul**（残差、加位置编码、归一化）
8. **Transpose / Reshape / Concat**（尽量通过图简化消掉）

## 7. 量化路线建议

1. FP32 跑通，作为参考结果
2. 整体切 FP16，对比 action 误差
3. Conv 和 Linear 换成 INT8：权重 per-channel 对称量化，激活 per-tensor，INT32 累加
4. Softmax、LayerNorm、action_head、前后处理保持 FP16/FP32

以上量化方案尚未在该模型上验证，INT8 的掉点需要在真机上评估。

## 附录：算子图导出脚本

```python
# 用法：python trace_act.py <pretrained_model 目录>
import collections
import sys

import torch

from lerobot.policies.act.modeling_act import ACTPolicy

p = ACTPolicy.from_pretrained(sys.argv[1]).eval().cpu()
m = p.model


class W(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, st, a, b):
        return self.m({"observation.state": st, "observation.images": [a, b]})[0]


B = 1
args = (torch.randn(B, 6), torch.randn(B, 3, 480, 640), torch.randn(B, 3, 480, 640))
ep = torch.export.export(W(m), args).run_decompositions({})

cnt = collections.Counter()
scope = collections.defaultdict(collections.Counter)
for n in ep.graph.nodes:
    if n.op != "call_function":
        continue
    name = str(n.target).replace("aten.", "")
    stack = n.meta.get("nn_module_stack", {})
    path = list(stack.values())[-1][0] if stack else "<top>"
    part = path.replace("L__self___m.", "").replace("m.", "", 1).split(".")[0]
    cnt[name] += 1
    scope[part][name] += 1

for k, v in cnt.most_common():
    print(f"{v:4d} {k}")
for part, c in scope.items():
    print(part, dict(c))
```
