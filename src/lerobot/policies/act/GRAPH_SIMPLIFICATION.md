# ACT 推理计算图简化方案

本文说明如何把训练得到的 ACT 模型（`modeling_act.py` 中的 `ACT`）离线转换成一张**数学等价、只含推理所需算子**的精简计算图，作为部署到 GPGPU 前的预处理步骤。算子全景见 `docs/act_operator_graph.md`。

## 0. 前提与原则

**适用配置**（转换代码应当对这些条件做 `assert`，不满足就拒绝转换）：

| 配置项 | 取值 | 被哪条简化依赖 |
|---|---|---|
| `n_decoder_layers` | 1 | B6、B7：decoder 自注意力常量化只对第 1 层成立 |
| `pre_norm` | False | B6、B7：常量的具体形式（先残差再 LN） |
| `feedforward_activation` | relu | 无强依赖 |
| 推理时 latent | 全 0（`ACT.forward` 中非训练分支） | B4 |
| 图像尺寸 | 固定，例如 480×640 → 特征图 15×20 | B5：位置编码按尺寸预计算 |
| 归一化模式 | `MEAN_STD`（VISUAL / STATE / ACTION） | C11–C13 |

**原则**：

- **不修改** `modeling_act.py`。简化是一次离线转换：输入 `ACTPolicy` checkpoint 和前后处理统计量，输出新的权重和常量。
- 每一条变换都要和原模型**逐输出比对**，FP32 下 (100, 6) 动作的最大绝对误差应在 1e-5 量级。
- 只做数学等价变换。kernel 级融合（conv+bias+ReLU+残差等）留给 GPGPU 端实现。

**记号**：`D = dim_model = 512`，`H = n_heads = 8`，`hd = D / H = 64`，`s = 1/√hd = 1/8`。

下文中 `m = ACTPolicy.from_pretrained(path).eval().cpu().model`。

---

## A. 删除推理用不到的部分

### A1. 删除 VAE encoder

**原因**：推理时 `ACT.forward` 走 `else` 分支，latent 直接取 0，`vae_encoder*` 相关的模块从不执行。

**涉及参数**（共约 17.4M）：

- `vae_encoder`（4 层 `ACTEncoderLayer`）
- `vae_encoder_cls_embed`
- `vae_encoder_robot_state_input_proj`、`vae_encoder_action_input_proj`、`vae_encoder_latent_output_proj`
- buffer `vae_encoder_pos_enc`

**代码做法**：精简模型不定义这些模块，从原 state_dict 只取需要的 key 即可：

```python
sd = {k: v for k, v in m.state_dict().items() if not k.startswith("vae_encoder")}
```

### A2. 删除 Dropout

**原因**：`eval()` 下 `nn.Dropout` 和 `nn.MultiheadAttention(dropout=...)` 都是恒等变换，但 trace 里仍会出现调用。

**代码做法**：精简模型里不写任何 dropout 调用。

### A3. 不再使用 `nn.MultiheadAttention`，改为手写注意力

**原因**：`nn.MultiheadAttention` 默认 `need_weights=True`，会额外把注意力权重在头维度上取平均（训练 trace 中的 10 个 `aten::mean`）。而 `ACTEncoderLayer` 和 `ACTDecoderLayer` 用 `[0]` 只取输出，这份平均结果被直接丢弃。此外，它内部的 Q/K/V 拆分会带来一串 `chunk`、`transpose`、`contiguous`。

**代码做法**：从 `in_proj_weight`（形状 `(3D, D)`）和 `in_proj_bias` 中拆出 Q/K/V 权重，注意力本体只保留两次 bmm 和一次 softmax。缩放因子已在 B9 中折进 Q：

```python
def attention(q, k, v, n_heads):
    """q: (Lq, D), k/v: (Lk, D)；q 已乘过 1/sqrt(hd)。"""
    lq, d = q.shape
    lk = k.shape[0]
    hd = d // n_heads
    q = q.view(lq, n_heads, hd).transpose(0, 1)   # (H, Lq, hd)
    k = k.view(lk, n_heads, hd).transpose(0, 1)   # (H, Lk, hd)
    v = v.view(lk, n_heads, hd).transpose(0, 1)   # (H, Lk, hd)
    attn = torch.softmax(q @ k.transpose(1, 2), dim=-1)  # (H, Lq, Lk)
    return (attn @ v).transpose(0, 1).reshape(lq, d)     # (Lq, D)

Wq, Wk, Wv = layer.self_attn.in_proj_weight.chunk(3, dim=0)
bq, bk, bv = layer.self_attn.in_proj_bias.chunk(3, dim=0)
Wo, bo = layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias
```

推理时 encoder 和 decoder 都不需要 mask：推理不传 `key_padding_mask`，decoder 也没有因果 mask。

---

## B. 常量折叠

### B4. latent token 折叠为常量

**推导**：`latent_sample = zeros(B, latent_dim)`，所以 `encoder_latent_input_proj(0) = W·0 + b = b`。

**代码做法**：

```python
latent_token = m.encoder_latent_input_proj.bias.clone()      # (D,)
```

这个 token 是 encoder 输入序列的第 0 个，直接作为常量写入。`encoder_latent_input_proj.weight` 可以丢弃。

### B5. encoder 位置编码整体预计算

**推导**：

- `ACTSinusoidalPositionEmbedding2d.forward(x)` 只用到 `x` 的形状 `(H, W)`，不用到它的值，所以 2D 编码是尺寸的函数。两路相机特征图尺寸相同，所以**两路的编码完全相同**。
- `encoder_1d_feature_pos_embed.weight` 是 `(2, D)` 的参数，分别对应 [latent, state]。
- encoder 的 token 顺序是 `[latent, state, cam0 的 h*w 个, cam1 的 h*w 个]`，位置编码按同样顺序拼接。

**代码做法**：

```python
fh, fw = 15, 20                                           # 480x640 经 ResNet18 下采样 32 倍
cam_pos = m.encoder_cam_feat_pos_embed(torch.zeros(1, D, fh, fw))    # (1, D, fh, fw)
cam_pos = einops.rearrange(cam_pos, "1 c h w -> (h w) c")            # (300, D)
n_cams = len(config.image_features)
enc_pos = torch.cat([m.encoder_1d_feature_pos_embed.weight] + [cam_pos] * n_cams, dim=0)  # (602, D)
```

`enc_pos` 会在 B8 中进一步被各层的 Q/K 投影吸收。如果采用 B8 的方案二，就直接作为常量保存。

### B6. decoder 自注意力整块折叠为常量

**推导**：decoder 的输入是 `decoder_in = zeros(chunk_size, B, D)`，第 1 层自注意力为：

```
q = k = 0 + qpos          v = 0
V  = 0·Wvᵀ + bv = bv                    每一行都等于 bv
out = softmax(QKᵀ)·V = bv               softmax 每行和为 1，与注意力权重无关
x  = norm1(0 + out_proj(bv))            post-norm：残差（加 0）后接 LN
```

所以 `x = norm1(Wo·bv + bo)` 是一个与输入无关的 D 维常量，100 个位置的值完全相同。整个自注意力块可以删掉：4 个 linear、2 个 bmm、1 个 softmax、1 个 LayerNorm，外加 decoder 位置编码的 add。

**代码做法**：

```python
dl = m.decoder.layers[0]
bv_self = dl.self_attn.in_proj_bias[2 * D:]
dec_c = dl.norm1(dl.self_attn.out_proj(bv_self))           # (D,)
dec_x = dec_c.expand(config.chunk_size, D).clone()         # (100, D)，作为 cross-attn 的残差输入
```

**已验证**：用 checkpoint `act_so101_pickplace_test` 对比原 `self_attn + norm1` 的输出，最大误差为 1.8e-7。

### B7. decoder cross-attn 的 Q 投影折叠为常量

**推导**：cross-attn 的 query 是 `dec_x + qpos`，两者都是常量（B6 和 `decoder_pos_embed.weight`），所以 Q 投影的结果也是常量。

**代码做法**：

```python
Wq, Wk, Wv = dl.multihead_attn.in_proj_weight.chunk(3, dim=0)
bq, bk, bv = dl.multihead_attn.in_proj_bias.chunk(3, dim=0)
qpos = m.decoder_pos_embed.weight                          # (100, D)
dec_q = F.linear(dec_x + qpos, Wq, bq) * s                 # (100, D) 常量，已含缩放
```

运行时 decoder 只需要计算 K/V 投影（输入是 encoder 输出）、cross-attn 本体、out_proj、norm2、FFN、norm3，以及最后的 `decoder.norm`。

### B8. 位置编码吸收进 Q/K 投影，并合并 Q/K/V 的 GEMM

**推导**：encoder 每一层都有 `q = k = x + pos`、`v = x`：

```
Q = Wq(x + pos) + bq = Wq·x + (Wq·pos + bq)
K = Wk(x + pos) + bk = Wk·x + (Wk·pos + bk)
V = Wv·x + bv
```

括号里的部分是常量，是一个逐 token 的 bias 表。于是三个投影可以合成一次 GEMM：

```python
W_qkv = torch.cat([Wq * s, Wk, Wv], dim=0)                                   # (3D, D)
B_qkv = torch.cat([(F.linear(enc_pos, Wq, bq)) * s,
                   F.linear(enc_pos, Wk, bk),
                   bv.expand(enc_pos.shape[0], D)], dim=1)                   # (602, 3D)
qkv = x @ W_qkv.T + B_qkv                                                    # (602, 3D)
q, k, v = qkv.split(D, dim=1)
```

**代价**：bias 表每层 602×1536 个 FP32，约 3.7 MB，4 层共约 14.8 MB。在显存或片上存储紧张的 GPGPU 上可能不划算。所以给出两种方案：

| 方案 | 运行时计算 | 额外常量 | 适合 |
|---|---|---|---|
| 一：全表 | 1 次 GEMM (D→3D) + 逐 token bias | 每层 602×3D | 存储充裕，想要最少 kernel 数 |
| **二：add + 两次 GEMM**（推荐先用） | `xp = x + enc_pos`（1 次逐元素 add）；`[Q,K] = xp @ [Wq·s; Wk]ᵀ + [bq·s; bk]`（D→2D）；`V = x @ Wvᵀ + bv`（D→D） | 只有 `enc_pos` 一份 (602, D) | 存储受限；逐元素 add 的代价可以忽略 |

decoder cross-attn 的 K 也可以同样处理：`K = Wk·enc_out + (Wk·enc_pos + bk)`。这里的常量表只有一份 (602, D)，约 1.2 MB，建议直接用方案一。

**已验证**：Q 的 pos 折叠（含缩放）与原始计算的最大误差为 3.6e-7。

### B9. 注意力缩放 1/√hd 折进 Q

**推导**：`softmax((Q·s)Kᵀ)`，所以 `Q·s = (Wq·s)·x + bq·s`。

**代码做法**：`Wq_fold = Wq * s`，`bq_fold = bq * s`，已体现在 A3、B7、B8 的代码中，运行时不再有缩放的 mul。

---

## C. 权重折叠

### C10. FrozenBatchNorm2d 折进前一个 conv

**推导**：torchvision `FrozenBatchNorm2d` 的计算为 `y = (x - rm) * w / sqrt(rv + eps) + b`，`eps = 1e-5`（已在 checkpoint 上确认）。令 `scale = w * rsqrt(rv + eps)`，则：

```
conv_fold.weight = conv.weight * scale[:, None, None, None]
conv_fold.bias   = b - rm * scale                     # ResNet 的 conv 本身不带 bias
```

**需要折叠的 conv/BN 对**（ResNet18 共 20 对）：

- `backbone.conv1` / `backbone.bn1`
- `backbone.layer{1..4}.{0,1}.conv1` / `.bn1`
- `backbone.layer{1..4}.{0,1}.conv2` / `.bn2`
- `backbone.layer{2..4}.0.downsample.0` / `.downsample.1`

**代码做法**：

```python
def fold_bn(conv: nn.Conv2d, bn: FrozenBatchNorm2d) -> nn.Conv2d:
    scale = bn.weight * (bn.running_var + bn.eps).rsqrt()
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                      conv.stride, conv.padding, conv.dilation, conv.groups, bias=True)
    fused.weight.data = conv.weight * scale[:, None, None, None]
    fused.bias.data = bn.bias - bn.running_mean * scale
    return fused
```

遍历 `backbone` 里每个 `BasicBlock`，把 `(conv, bn)` 替换为 `(fold_bn(conv, bn), nn.Identity())`。stem 和 downsample 也一样处理。

**已验证**：`conv1+bn1` 折叠后在 480×640 随机输入上的最大误差为 3.3e-6。

### C11. state 归一化折进 `encoder_robot_state_input_proj`

**推导**：前处理是 `s_n = (s - μ) / (σ + ε)`（`ε = 1e-8`，见 `processor/normalize_processor.py`）。代入 linear：

```
W·s_n + b = (W / (σ+ε))·s + (b - (W / (σ+ε))·μ)
```

**代码做法**（统计量在 `policy_preprocessor_step_3_normalizer_processor.safetensors`，key 为 `observation.state.mean` / `observation.state.std`）：

```python
mu, sd = stats["observation.state.mean"], stats["observation.state.std"] + 1e-8
W, b = m.encoder_robot_state_input_proj.weight, m.encoder_robot_state_input_proj.bias
W_state = W / sd[None, :]
b_state = b - W_state @ mu
```

运行时直接把原始关节角送进这个 linear 即可，省掉一次 sub 和一次 div。

### C12. action 反归一化折进 `action_head`

**推导**：后处理是 `a = y * σ + μ`。注意反归一化**不加 ε**，见 `normalize_processor.py` 的 `inverse` 分支。代入 `y = W·h + b`：

```
a = (σ[:,None] * W)·h + (σ * b + μ)
```

**代码做法**（统计量在 `policy_postprocessor_step_0_unnormalizer_processor.safetensors`，key 为 `action.mean` / `action.std`）：

```python
mu, sd = stats["action.mean"], stats["action.std"]
W_act = sd[:, None] * m.action_head.weight
b_act = sd * m.action_head.bias + mu
```

模型输出直接就是真实动作。

### C13. 图像归一化折进 `backbone.conv1`（可选，默认关闭）

**推导**：`x_n = (x - μ_c) / (σ_c + ε)`，逐通道。conv 对输入是线性的：

```
W'[:, c] = W[:, c] / (σ_c + ε)
b'      = b - Σ_c Σ_kh,kw W'[:, c, kh, kw] · μ_c
```

**坑**：`conv1` 是 7×7、`padding=3` 的 zero padding。原模型是对**归一化后**的图补 0，相当于原图边缘补 μ。折叠后变成对**原图**补 0，所以输出在边缘 3 个像素宽的区域上**不等价**，误差会沿网络传播。

**第二个坑**：每路相机的统计量不同（`observation.images.up.*` 和 `observation.images.side.*` 各有一套 μ/σ）。折叠后每路相机需要各自的 conv1 权重，就不能再按 D15 把多路相机拼成一个 batch 共享 backbone。

**处理方式**：

- **当前实现不做 C13**。图像归一化在 GPGPU 上作为一个逐通道的 `(x - μ) * inv_σ` kernel 单独实现，代价很小。导出的常量为 `img_mean`、`img_inv_std`，形状都是 `(n_cams, 3, 1, 1)`。
- 如果以后要开启，可以对原图先按 μ 做 padding，再用 `padding=0` 的折叠 conv，同时放弃 D15 的相机合批。

注意：送进 ACT 的图像已经是 [0, 1] 的 float。如果 GPGPU 端拿到的是 uint8，`/255` 这一步也可以并进同一个逐通道 kernel。

---

## D. 布局与调度

### D14. batch=1 统一用 2D 张量

原代码使用 `(seq, B, D)` 布局，并多次调用 `einops.rearrange`、`permute`、`transpose`。部署时 batch 固定为 1，所以全部张量统一成 `(tokens, D)`，行主序：

- encoder 输入、输出：`(602, D)`
- decoder：`(100, D)`
- 输出：`(100, 6)`

linear 统一写成 `x @ W.T + b`。transpose 可以在转换时一次性做掉，直接保存 `W.T`，具体取决于 GPGPU 端 GEMM 的布局约定。

### D15. 多路相机合并成一个 batch 过 backbone

原代码用 `for img in batch[OBS_IMAGES]` 对每路相机分别调用 backbone。由于共享同一份权重，可以把图像拼成 `(n_cams, 3, 480, 640)` 一次跑完：

```python
imgs = torch.stack([img_up, img_side], dim=0)              # (2, 3, 480, 640)
feat = backbone_folded(imgs)                               # (2, 512, 15, 20)
feat = img_proj(feat)                                      # 1x1 conv = (2*300, 512) @ (512, 512) 的 GEMM
cam_tokens = feat.flatten(2).permute(0, 2, 1).reshape(-1, D)   # (600, D)，顺序与 enc_pos 一致
```

这样 kernel launch 次数减半，GEMM 的 M 维更大，利用率更好。`encoder_img_feat_input_proj` 是 1×1 conv，在 GPGPU 上可以直接按 GEMM 实现。

---

## E. 简化后的完整推理图

```python
def act_infer(state, img_up, img_side):                    # state: (6,) 原始关节角；img: (3,480,640) [0,1]
    imgs = img_norm(torch.stack([img_up, img_side]))       # C13 关闭时保留的逐通道归一化
    feat = backbone_folded(imgs)                           # C10：只剩 conv(+bias) / relu / maxpool / add
    cam = img_proj(feat).flatten(2).permute(0, 2, 1).reshape(-1, D)   # (600, D)

    state_tok = state @ W_state.T + b_state                # C11
    x = torch.cat([latent_token[None], state_tok[None], cam], dim=0)  # (602, D)；B4

    for L in enc_layers:                                   # 4 层，B8 方案二
        xp = x + enc_pos
        qk = xp @ L.W_qk.T + L.b_qk                        # (602, 2D)，Q 已含缩放
        v = x @ L.Wv.T + L.bv
        q, k = qk.split(D, dim=1)
        x = layer_norm(x + attention(q, k, v, H) @ L.Wo.T + L.bo, L.n1)
        x = layer_norm(x + relu(x @ L.W1.T + L.b1) @ L.W2.T + L.b2, L.n2)

    # decoder：B6 自注意力已消失，B7 Q 是常量
    k = x @ Wk_c.T + kbias_c                               # kbias_c = Wk·enc_pos + bk，(602, D)
    v = x @ Wv_c.T + bv_c
    y = layer_norm(dec_x + attention(dec_q, k, v, H) @ Wo_c.T + bo_c, dn2)
    y = layer_norm(y + relu(y @ W1_d.T + b1_d) @ W2_d.T + b2_d, dn3)
    y = layer_norm(y, dnorm)                               # ACTDecoder.norm
    return y @ W_act.T + b_act                             # (100, 6) 真实动作；C12
```

**剩下的算子种类**：conv2d（含 bias）、relu、max_pool2d、add、GEMM、bmm、softmax、layer_norm，以及逐通道的图像归一化。

**规模变化**（估算）：

| | 原模型 | 简化后 |
|---|---|---|
| 参数 | 51.7M | 约 34M |
| decoder 运行时 MAC | 约 0.87G | 约 0.75G |
| 运行时常量 | — | `latent_token`、`enc_pos`、`dec_x`、`dec_q`、`kbias_c` |

## F. 代码实现

```
src/lerobot/policies/act/deploy/
├── simplify.py      # ACTPolicy + 归一化统计 → 折叠后的权重/常量 dict
├── act_infer.py     # 第 E 节的精简推理模型 ACTInference（纯 PyTorch），以及导出/加载
└── export.py        # CLI：转换 → 与原 ACTPolicy（含前后处理）逐输出比对 → 导出
tests/policies/act/test_deploy_simplify.py
```

### F.1 `simplify.py` 中各函数与变换的对应

| 函数 | 负责的变换 |
|---|---|
| `check_supported` | 第 0 节的前提检查，不满足就抛 `ValueError` |
| `load_normalization_stats` | 从 `policy_preprocessor.json` / `policy_postprocessor.json` 找到 normalizer 步骤，读取其 safetensors 统计量和 eps |
| `simplify_backbone` / `fold_bn` | C10；同时记录每个 conv 的 stride/padding、各 layer 的 block 数、maxpool 参数 |
| `encoder_pos_embed` | B5 |
| `simplify_encoder` | A3、B8（`qkv_mode="add"` 或 `"table"`）、B9 |
| `simplify_decoder` | A3、B6、B7、B9，cross-attn 的 K 用 B8 的表形式 |
| `simplify_io` | B4、C11、C12，图像归一化常量，D15 的 1×1 投影改写成 GEMM 权重 |
| `simplify_act` | 依次调用上面的函数，返回 `(tensors, meta)`；A1、A2 体现为这些模块的权重根本不被导出 |

### F.2 用法

```bash
uv run python -m lerobot.policies.act.deploy.export \
    --pretrained-path outputs/train/act_so101_pickplace_test/checkpoints/last/pretrained_model \
    --output-dir outputs/deploy/act_so101_pickplace_test \
    --qkv-mode add            # 或 table
```

脚本会先和原始路径 `preprocessor → ACTPolicy.predict_action_chunk → postprocessor` 在随机输入上比对，误差超过 `--atol`（默认 1e-3）就拒绝导出。

输出两个文件：

- `act_simplified.safetensors`：所有权重和常量，meta 以 JSON 形式写在 safetensors 的 metadata 里
- `act_simplified.json`：meta、每个张量的形状、总参数量，方便 GPGPU 端 runtime 对接

Python 端用 `ACTInference.from_file(path)` 加载，调用 `model(state, images)`：

- `state` 是 (6,) 的原始关节角
- `images` 是 (n_cams, 3, H, W)、取值 [0, 1] 的图像，顺序和 meta 里的 `camera_keys` 一致
- 返回 (100, 6) 的真实动作

### F.3 实测结果（checkpoint `act_so101_pickplace_test`，FP32，CPU）

| `qkv_mode` | 参数量（含常量） | 与原模型的动作最大误差 |
|---|---|---|
| 原模型 | 51.60M | — |
| `add` | 33.56M | 3.1e-5 |
| `table` | 36.95M | 2.7e-5 |

### F.4 测试

`tests/policies/act/test_deploy_simplify.py` 使用小尺寸、随机初始化的 ACT，不依赖 checkpoint：

- 随机化 FrozenBN 的统计量，确保 C10 真正被测到
- 用真实的 `make_act_pre_post_processors` 作为参考路径，比对两种 `qkv_mode` 的输出，并检查导出、加载往返后结果一致
- 检查导出结果里没有 VAE encoder 和 decoder 自注意力的权重
- 检查 `n_decoder_layers=2`、`pre_norm=True` 这类不支持的配置会被拒绝
