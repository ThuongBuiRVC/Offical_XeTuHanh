# SPEC: Driving World Model — DiT + Shortcut Flow Matching (Single-Stage Joint Training)

> **Mục tiêu.** Huấn luyện MỘT mô hình DiT duy nhất, end-to-end, đồng thời học **world model** (latent video tương lai trong không gian V-JEPA) và **action model** (waypoints lái), benchmark trên **NuPlan closed-loop**.
>
> **Nguồn gốc & quyết định cốt lõi.** Kiến trúc lấy ý tưởng từ **DreamZero (NVIDIA, "World Action Models are Zero-shot Policies")**. Điểm mấu chốt của DreamZero là **joint prediction video + action trong MỘT model end-to-end**, KHÔNG tách thành nhiều model/nhiều pha. Trích nguyên văn paper:
> - *"we train a single model end-to-end with joint prediction objective."*
> - *"We update all DiT blocks, the state encoder, action encoder, and action decoder, while freezing the text encoder, image encoder, and VAE."*
>
> **=> Vì vậy SPEC này huấn luyện TẤT CẢ trong MỘT pha duy nhất (joint).** Action decoder được train CHUNG với DiT, không có pha riêng đóng băng DiT. Đây là thay đổi quan trọng nhất so với các bản nháp trước (vốn tách Phase 1 / Phase 2).

---

## 0. Bản đồ tài liệu (đọc theo thứ tự này)

1. §1 Quy ước & ký hiệu
2. §2 V-JEPA backbone (frozen) — cách lấy latent, suy `N_v` động
3. §3 Inputs chi tiết & các MLP nâng chiều
4. §4 Conditioning `(t, d)` qua AdaLN-Zero (KHÔNG nối vào sequence)
5. §5 Kiến trúc tổng thể (sơ đồ luồng dữ liệu)
6. §6 DiT block (AdaLN-Zero, self-attn + cross-attn + FFN)
7. §7 Flow Matching + Shortcut (noising, velocity target, self-consistency)
8. §8 Action Decoder (train CHUNG, dùng ước lượng clean một bước)
9. §9 **Joint Training — pha duy nhất + toàn bộ hàm loss**
10. §10 Inference / ODE solver cho closed-loop
11. §11 Visualize PCA latent (debug tool)
12. §12 File structure đề xuất
13. §13 Acceptance checks (AI agent tự verify)
14. §14 Câu hỏi cần làm rõ + §15 Bảng quyết định thiết kế

---

## 1. Quy ước & Ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| `B` | Batch size |
| `D` | Embedding dim chính của DiT = **768** (khớp V-JEPA 2.1 ViT-B) |
| `T_past` | Số frame quá khứ = **8** (2 Hz × 3.5s) |
| `T_fut` | Số frame tương lai = **8** (2 Hz × 4s) |
| `N_v` | Số token V-JEPA cho 1 clip 8 frame = **suy động bằng forward thực tế** (xem §2) |
| `N_route` | Số waypoint route trong ego-frame hiện tại = **20** |
| `N_ego` | Số bước ego history = **8** |
| `N_act` | Số waypoint tương lai cần dự đoán = **8** |
| `z` | Future latent (V-JEPA features của video tương lai) — đối tượng "world model" |
| `a` | Future action latent (waypoints đã nâng chiều 768) — đối tượng "action model" |
| `t` | Flow Matching timestep ∈ [0, 1] |
| `d` | Shortcut step-size ∈ {0, 1/128, ..., 1/2} |

**Convention CỨNG (không được vi phạm):**
- **KHÔNG project latent V-JEPA qua bất kỳ MLP nào.** Lấy raw output `[B, N_v, 768]` từ encoder, dùng trực tiếp (cả past lẫn future). Lý do: giữ nguyên không gian đặc trưng đã pretrain; world-model loss học đúng dynamics trong không gian đó.
- Chỉ **Nav route, Ego history, Action waypoints** (vì chiều input < 768) mới đi qua MLP nâng chiều lên 768.
- **`(t, d)` KHÔNG nối vào sequence.** Chúng đi qua AdaLN-Zero conditioning (§4). Đây là điểm sửa so với bản nháp cũ (vốn `cat([z_t, a_t, t_embed])`).
- Convention flow matching: `t=0` là noise thuần, `t=1` là clean. Đường nội suy tuyến tính.

---

## 2. V-JEPA 2.1 Backbone (FROZEN)

### 2.1 Variant
Dùng **V-JEPA 2.1 ViT-B/16 (~80M params, 384×384)**.
- `embed_dim = 768`, `patch_size = 16`, `tubelet_size = 2`
- Số token lý thuyết cho 1 clip 8 frame @384: `(384/16)² × (8/2) = 576 × 4 = 2304`

> ⚠️ Tài liệu nháp gốc viết "2048 token" — đó là **ước lượng sai** với V-JEPA 2.1 ViT-B/384. Code **PHẢI suy `N_v` động** từ forward pass thực, KHÔNG hardcode. Nếu sau đổi resolution (256 → `N_v=1024`) hoặc đổi số frame thì code vẫn chạy.

### 2.2 Load & freeze
```python
from vjepa2 import load_vjepa_2_1_vitb_384
encoder = load_vjepa_2_1_vitb_384(pretrained=True)
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False   # FROZEN suốt toàn bộ training
```

### 2.3 Forward & cache `N_v`
- Input: `[B, T=8, 3, 384, 384]` → Output raw (KHÔNG project): `[B, N_v, 768]`.
- Ở `__init__` của model wrapper: chạy **1 dummy forward**, lưu `self.n_v`.
- Gọi encoder **2 lần riêng biệt** (past và future), đều dưới `torch.no_grad()`:
  - Past → `camera_past [B, N_v, 768]` (làm context K,V)
  - Future → `z_clean [B, N_v, 768]` (làm target để pha nhiễu flow matching)

---

## 3. Inputs chi tiết & MLP nâng chiều

| Input | Shape thô | Xử lý | Shape sau |
|---|---|---|---|
| Camera quá khứ | `[B, 8, 3, 384, 384]` | Frozen V-JEPA (no grad) | `[B, N_v, 768]` |
| Camera tương lai | `[B, 8, 3, 384, 384]` | Frozen V-JEPA (no grad) → target | `[B, N_v, 768]` |
| Route waypoints | `[B, 20, 2]` | **NavMLP** | `[B, 20, 768]` |
| Ego history | `[B, 8, 7]` | **EgoMLP** | `[B, 8, 768]` |
| Action waypoints GT | `[B, 8, 2]` | **ActionMLP** | `[B, 8, 768]` = `a_clean` |
| Timestep `t` | `[B]` scalar | Sinusoidal + MLP (§4) | góp vào `cond [B,768]` |
| Shortcut step `d` | `[B]` scalar | Sinusoidal + MLP (§4) | góp vào `cond [B,768]` |

```python
# Tất cả MLP nâng chiều dùng GELU + LayerNorm
NavMLP    : Linear(2,        384) → GELU → LayerNorm(384) → Linear(384, 768)
EgoMLP    : Linear(ego_dim,  384) → GELU → LayerNorm(384) → Linear(384, 768)
ActionMLP : Linear(2,        384) → GELU → LayerNorm(384) → Linear(384, 768)
```

> **LƯU Ý:** KHÔNG có `VJepaProjection`/`VJepaMLP`. Latent V-JEPA đi thẳng vào DiT.

---

## 3.5 Data Preparation Cache

Training **không đọc ảnh/JPEG, query map, hoặc transform pose trực tiếp trong training loop**. Tất cả bước nặng phải chạy trước bằng `scripts/prepare_nuplan_cache.py`, tạo cache `.pt` theo split:

```
Data/nuplan_cache/
├── train/<log_name>_<timestamp_us>.pt
├── val/<log_name>_<timestamp_us>.pt
├── manifest_train.csv
├── manifest_val.csv
├── stats_train.json
└── stats_val.json
```

Mỗi sample `.pt` chứa đúng các key tensor:

| Key | Shape | Nội dung |
|---|---:|---|
| `past_cam` | `[8, 3, 384, 384]` | CAM_F0 quá khứ, ImageNet-normalized |
| `fut_cam` | `[8, 3, 384, 384]` | CAM_F0 tương lai, ImageNet-normalized |
| `route` | `[20, 2]` | route lane-centerline lookahead trong ego-frame hiện tại |
| `ego` | `[8, 7]` | `[x, y, yaw, vx, vy, ax, ay]` trong ego-frame hiện tại |
| `wp_gt` | `[8, 2]` | waypoint GT tại `0.5s..4.0s`, mét, ego-frame hiện tại |

Waypoint GT được tạo từ future ego poses:

```python
ref_pos = ego_pos[t0]
ref_from_world = world_from_ego[t0].T
wp_gt = (ref_from_world @ (future_pos - ref_pos).T).T[:, :2]
```

Route cũng phải được đưa về **ego-frame hiện tại** trước khi vào `NavMLP`; không dùng tọa độ map/global tuyệt đối làm input trực tiếp vì scale và offset thay đổi theo log/map.

---

## 4. Conditioning `(t, d)` qua AdaLN-Zero

`t` và `d` được biến thành **MỘT vector `cond [B, 768]`** rồi inject vào MỌI DiT block (và Final layer) qua AdaLN-Zero. KHÔNG đưa vào sequence.

```python
def sinusoidal_embed(x: Tensor, dim: int = 768) -> Tensor:
    # x: [B] -> [B, dim]
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / half)
    args = x[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class CondEmbedder(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.step_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: Tensor, d: Tensor) -> Tensor:
        # t, d: [B] -> cond: [B, 768]
        return self.time_mlp(sinusoidal_embed(t)) + self.step_mlp(sinusoidal_embed(d))
```

**Tại sao AdaLN-Zero, không phải concat token:** với sequence rất dài (`N_v≈2304`), nếu nối `t_embed` thành 1 token thì tín hiệu `(t,d)` bị pha loãng và phải đi gián tiếp qua self-attn mới tới các token khác. AdaLN bơm `(t,d)` trực tiếp vào LayerNorm modulation của mọi token ở mọi layer → hội tụ nhanh hơn ~30–50%, ổn định hơn (gate init=0), và là chuẩn của DiT/SD3/Flux/Shortcut-FM. Shortcut FM đặc biệt cần phân biệt rõ `t` (mức nhiễu) và `d` (bước nhảy), nên conditioning trực tiếp rất quan trọng.

---

## 5. Kiến trúc tổng thể

```
PAST INPUTS (Context)                       NOISY FUTURE TARGETS (Core)
─────────────────────                       ───────────────────────────
Past video [B,8,3,384,384]                  Future video [B,8,3,384,384]
   │ frozen V-JEPA (no grad)                   │ frozen V-JEPA (no grad)
   ▼                                           ▼
camera_past [B,N_v,768]                      z_clean [B,N_v,768]
                                                │ flow-matching noising (§7)
Route waypoints [B,20,2]                        ▼
   │ NavMLP                                  z_t [B,N_v,768]
   ▼
route_tokens [B,20,768]                      Action GT [B,8,2]
                                                │ ActionMLP
Ego history [B,8,7]                             ▼
   │ EgoMLP                                  a_clean [B,8,768]
   ▼                                            │ flow-matching noising (§7)
ego_tokens [B,8,768]                            ▼
                                             a_t [B,8,768]

Timestep t [B]        Shortcut step d [B]
   └────── CondEmbedder (§4) ──────┘
                  ▼
            cond [B,768]  ──► inject vào MỌI DiT block + Final qua AdaLN-Zero
                              (KHÔNG nối vào sequence)

CONTEXT (K,V cho cross-attn):
    past_ctx  = cat([camera_past, route_tokens, ego_tokens], dim=1)   # [B, N_v+28, 768]

NOISY CORE (Q cho self-attn, rồi cross-attn sang past_ctx):
    noisy_seq = cat([z_t, a_t], dim=1)                                 # [B, N_v+8, 768]

                  ▼
   ┌─────────────────────────────────────────┐
   │  DiT × 8 layers (AdaLN-Zero theo cond)    │
   │   • RMSNorm → modulate → Self-Attn(Q)     │
   │   • RMSNorm → Cross-Attn(Q=core, KV=ctx)  │  (cross KHÔNG modulate)
   │   • RMSNorm → modulate → FFN              │
   └─────────────────────┬─────────────────────┘
                         ▼  FinalLayer (AdaLN-Zero, init zero)
              output [B, N_v+8, 768]
                         │
         ┌───────────────┴────────────────┐
         ▼ (cắt N_v token đầu)             ▼ (cắt 8 token cuối)
   v_z_pred [B,N_v,768]              v_a_pred [B,8,768]
   (latent velocity = world)        (action velocity = lái)
         │                                  │
         │                                  ├─ ước lượng clean 1 bước:
         │                                  │     â = a_t + (1−t)·v_a_pred   [B,8,768]
         │                                  ▼
         │                            ActionDecoder (MLP, train CHUNG)
         │                                  ▼
         │                            waypoint_pred [B,8,2]
         ▼                                  ▼
   (chỉ dùng lúc viz: PCA → ảnh)     (điều khiển xe + loss task)
```

**Slicing đầu ra (chính xác):**
```python
v_z_pred = output[:, :n_v, :]        # [B, N_v, 768]
v_a_pred = output[:, n_v:n_v+8, :]   # [B, 8, 768]
```

---

## 6. DiT Block (8 layers, AdaLN-Zero)

### 6.1 Hyperparameters cố định
| Param | Value |
|---|---|
| `num_layers` | **8** |
| `hidden_dim` | 768 |
| `num_heads` | 12 (head_dim = 64) |
| `mlp_ratio` | 4.0 (FFN hidden = 3072) |
| `attn_dropout` / `ff_dropout` | 0.0 |
| `qk_norm` | True (ổn định attention) |
| `norm` | RMSNorm |
| `activation` (FFN) | SiLU |
| `position_embedding` | RoPE 1D dọc trục sequence cho cả Q và KV |

### 6.2 Cấu trúc 1 block
Mỗi block nhận `x` (noisy core, làm Q), `ctx` (past context, làm K,V), và `cond [B,768]`.
AdaLN modulator sinh **6 vector** từ `cond`: (scale1, shift1, gate1) cho self-attn, (scale2, shift2, gate2) cho FFN. **Cross-attn KHÔNG modulate.**

```python
class DiTBlock(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn  = SelfAttention(dim, num_heads, qk_norm=True, rope=True)
        self.norm2 = RMSNorm(dim)
        self.cross = CrossAttention(dim, num_heads, qk_norm=True, rope=True)
        self.norm3 = RMSNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)), nn.SiLU(),
            nn.Linear(int(dim*mlp_ratio), dim),
        )
        # AdaLN-Zero modulator: cond -> 6 vector dim D
        self.ada_mod = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada_mod.weight)   # CRITICAL: init 0 -> block = identity lúc đầu
        nn.init.zeros_(self.ada_mod.bias)

    def forward(self, x, ctx, cond):
        # x:[B,L_q,D]  ctx:[B,L_kv,D]  cond:[B,D]
        s1, sh1, g1, s2, sh2, g2 = self.ada_mod(cond).chunk(6, dim=-1)
        s1, sh1, g1 = (v.unsqueeze(1) for v in (s1, sh1, g1))   # [B,1,D]
        s2, sh2, g2 = (v.unsqueeze(1) for v in (s2, sh2, g2))

        # 1) Self-attn (có modulation)
        h = self.norm1(x) * (1 + s1) + sh1
        x = x + g1 * self.attn(h)

        # 2) Cross-attn sang context (KHÔNG modulation)
        x = x + self.cross(self.norm2(x), ctx)

        # 3) FFN (có modulation)
        h = self.norm3(x) * (1 + s2) + sh2
        x = x + g2 * self.ffn(h)
        return x
```

**Vì sao init `ada_mod = 0`:** scale=shift=gate=0 → `gate × output = 0` → mỗi block ban đầu là **identity** (chỉ residual đi qua). Mạng khởi đầu như "no-op stack", học dần khi gate tăng → cực ổn định, tránh exploding gradient ở những step đầu. Đây là trick chính của DiT, bắt buộc.

### 6.3 Final layer (sau 8 block)
```python
class FinalLayer(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.ada_mod = nn.Linear(dim, 2 * dim)     # chỉ scale + shift
        nn.init.zeros_(self.ada_mod.weight); nn.init.zeros_(self.ada_mod.bias)
        nn.init.zeros_(self.linear.weight);  nn.init.zeros_(self.linear.bias)  # output velocity = 0 lúc đầu

    def forward(self, x, cond):
        s, sh = self.ada_mod(cond).chunk(2, dim=-1)
        h = self.norm(x) * (1 + s.unsqueeze(1)) + sh.unsqueeze(1)
        return self.linear(h)
```

### 6.4 Ước lượng tham số
| Component | Params |
|---|---|
| Self-attn (8×~2.4M) | ~19M |
| Cross-attn (8×~2.4M) | ~19M |
| FFN (8×~4.7M) | ~38M |
| AdaLN modulators (8 block + final) | ~28M |
| Norms + final linear + projections | ~3M |
| Nav/Ego/Action MLP | ~1M |
| CondEmbedder | ~2M |
| **Action Decoder (train CHUNG)** | ~0.2M |
| **TỔNG (trainable)** | **~110M** |

> Hơi vượt 100M do AdaLN modulator (~28M). Muốn về ~85M: dùng **shared modulator** (1 Linear chung cho 8 block). Mặc định SPEC giữ modulator riêng mỗi block (chuẩn DiT). V-JEPA (80M) KHÔNG tính vì frozen.

---

## 7. Flow Matching + Shortcut

### 7.1 Forward noising (linear, `t=0` noise → `t=1` clean)
```python
def noise(x_clean, x_noise, t):
    # t:[B] -> broadcast theo số chiều của x
    t = t.view(-1, *([1] * (x_clean.dim() - 1)))
    return (1 - t) * x_noise + t * x_clean

# velocity target (constant dọc đường thẳng):
v_target = x_clean - x_noise
```
Áp dụng RIÊNG cho video và action (noise độc lập, cùng `t` per-sample):
```python
z_noise = torch.randn_like(z_clean);  z_t = noise(z_clean, z_noise, t)
a_noise = torch.randn_like(a_clean);  a_t = noise(a_clean, a_noise, t)
v_z_target = z_clean - z_noise
v_a_target = a_clean - a_noise
```

### 7.2 Shortcut (Frans et al. 2024, "One Step Diffusion via Shortcut Models")
Model nhận thêm step-size `d`. Self-consistency: dự đoán với bước `d` phải bằng trung bình 2 dự đoán nối tiếp với bước `d/2`.

**Chiến lược sample `(t, d)` mỗi batch:**
- **75% sample:** `d = 0` → flow matching thuần, target = `x_clean − x_noise`.
- **25% sample:** `d > 0`, sample `d ∈ {1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2}` uniform → tính self-consistency target.

**Tính shortcut target (chỉ cho 25% kia), không gradient:**
```python
with torch.no_grad():
    v1     = model_core(x_t,                cond(t,     d/2), t,       d/2)   # [v_z, v_a]
    x_mid  = x_t + (d/2) * v1
    v2     = model_core(x_mid,              cond(t+d/2, d/2), t+d/2,   d/2)
    v_target_shortcut = (v1 + v2) / 2     # cho cả z và a
```
> `model_core` ở đây là forward DiT trả `[v_z, v_a]` (chưa qua decoder). Shortcut target áp dụng cho cả `v_z` và `v_a`.

---

## 8. Action Decoder (TRAIN CHUNG — không phải pha riêng)

### 8.1 Kiến trúc
```python
class ActionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.LayerNorm(256), nn.Linear(256, 2)
        )
    def forward(self, a_latent):       # [B,8,768] -> [B,8,2]
        return self.net(a_latent)
```

### 8.2 ⚠️ Input của decoder lúc training: dùng ước lượng CLEAN MỘT BƯỚC, KHÔNG dùng GT
Đây là điểm sửa quan trọng. Với convention `x_t = (1−t)·noise + t·clean` và `v = clean − noise`:
```
clean = x_t + (1 − t) · v
```
Nên trong training, action latent "sạch" mà decoder nhìn được phải tái dựng từ **velocity DiT dự đoán**:
```python
a_hat_clean = a_t + (1 - t).view(-1,1,1) * v_a_pred   # [B,8,768]
waypoint_pred = action_decoder(a_hat_clean)            # [B,8,2]
```

**Vì sao KHÔNG dùng `a_clean = ActionMLP(wp_gt)`:** nếu decode từ `a_clean`, decoder chỉ học **đảo ngược ActionMLP** (trivial), không hề phụ thuộc chất lượng DiT — đúng cái sai của thiết kế 2-pha cũ. Dùng `a_hat_clean` để **gradient của waypoint loss chảy ngược qua `v_a_pred` vào DiT**, biến tín hiệu nhiệm vụ (NuPlan) thành lực định hình representation của world-action model. KHÔNG `detach`. (Nếu gặp bất ổn hiếm gặp, có thể stop-grad tạm thời, nhưng mặc định để gradient chảy đầy đủ — đây mới là tinh thần joint của DreamZero.)

---

## 9. JOINT TRAINING — PHA DUY NHẤT (toàn bộ loss)

### 9.1 Trạng thái mạng (1 optimizer, 1 vòng train)
| Component | Trainable? |
|---|---|
| V-JEPA 2.1 ViT-B | ❌ **Frozen** (luôn luôn) |
| Nav / Ego / Action input MLP | ✅ |
| CondEmbedder (Time + Step MLP) | ✅ |
| DiT 8 layers (gồm AdaLN modulators) | ✅ |
| Final layer | ✅ |
| **Action Decoder** | ✅ **(train CHUNG, KHÔNG đóng băng)** |

> KHÔNG còn Phase 1 / Phase 2. Mọi thành phần (trừ V-JEPA) học đồng thời trong cùng một pha với cùng một optimizer.

### 9.2 Một bước training (forward + loss)
```python
# --- Encode (frozen) ---
camera_past = vjepa(past_video)          # no grad, [B,N_v,768]
z_clean     = vjepa(future_video)        # no grad, [B,N_v,768]
route_tok   = nav_mlp(route)             # [B,20,768]
ego_tok     = ego_mlp(ego)               # [B,8,768]
a_clean     = action_mlp(wp_gt)          # [B,8,768]

# --- Sample (t, d) theo §7.2; noising theo §7.1 ---
t, d = sample_t_d(B)                     # 75% d=0, 25% d>0
z_noise = randn_like(z_clean); z_t = noise(z_clean, z_noise, t)
a_noise = randn_like(a_clean); a_t = noise(a_clean, a_noise, t)
v_z_target = z_clean - z_noise
v_a_target = a_clean - a_noise

# --- DiT forward ---
past_ctx  = cat([camera_past, route_tok, ego_tok], dim=1)   # [B,N_v+28,768]
noisy_seq = cat([z_t, a_t], dim=1)                          # [B,N_v+8,768]
cond      = cond_embedder(t, d)
out       = dit(noisy_seq, past_ctx, cond)                  # [B,N_v+8,768]
v_z_pred  = out[:, :n_v, :]
v_a_pred  = out[:, n_v:n_v+8, :]

# --- Action decode từ ước lượng clean 1 bước (§8.2) ---
a_hat_clean   = a_t + (1 - t).view(-1,1,1) * v_a_pred
waypoint_pred = action_decoder(a_hat_clean)                 # [B,8,2]
```

### 9.3 Các hàm loss
```python
# (1) World-model flow loss (MSE trong latent V-JEPA 768-dim, KHÔNG pixel)
L_flow_z = F.mse_loss(v_z_pred, v_z_target)

# (2) Action flow loss (MSE trong latent action 768-dim)
L_flow_a = F.mse_loss(v_a_pred, v_a_target)

# (3) Waypoint loss (Huber/smooth-L1) — tín hiệu metric NuPlan, bám quỹ đạo chuyên gia
L_wp = F.smooth_l1_loss(waypoint_pred, wp_gt, beta=0.5)

# (4) Temporal smoothness — phạt sai phân bậc 2, triệt giật vô lăng
def smoothness_loss(wp):                 # wp:[B,8,2]
    diff2 = wp[:, 2:] - 2*wp[:, 1:-1] + wp[:, :-2]   # [B,6,2]
    return (diff2 ** 2).mean()
L_smooth = smoothness_loss(waypoint_pred)

# (5) Shortcut self-consistency (chỉ 25% sample có d>0; còn lại = 0)
#     áp cho cả v_z và v_a; target tính theo §7.2 dưới no_grad
L_sc = F.mse_loss(v_z_pred[mask_sc], v_target_sc_z[mask_sc]) \
     + F.mse_loss(v_a_pred[mask_sc], v_target_sc_a[mask_sc])
#     (nếu mask_sc rỗng trong batch -> L_sc = 0)
```

### 9.4 Tổng loss (một biểu thức duy nhất)
```python
L_total = (  L_flow_z
           + lambda_a      * L_flow_a
           + lambda_wp     * L_wp
           + lambda_smooth * L_smooth
           + lambda_sc     * L_sc )
```

| Hệ số | Giá trị mặc định | Ghi chú |
|---|---|---|
| `lambda_a` | 1.0 | cân bằng world vs action trong latent |
| `lambda_wp` | **5.0** | waypoint là tín hiệu quyết định metric NuPlan, nhưng magnitude nhỏ (2-dim, đơn vị mét) so với flow loss (768-dim) → cần trọng số lớn hơn. Tune trong [1, 10]. |
| `lambda_smooth` | 0.1 | tránh giật; quá lớn sẽ làm xe "lười" đánh lái |
| `lambda_sc` | 1.0 | chỉ tác động lên 25% sample shortcut |

> **Lưu ý cân bằng gradient:** `L_flow_z` (2304×768 phần tử) áp đảo về magnitude. Theo dõi riêng từng loss trong log; nếu `L_wp` không giảm, tăng `lambda_wp`. Có thể chuẩn hoá waypoint về cùng scale (ví dụ chia cho độ lệch chuẩn quỹ đạo) trước khi tính Huber.

### 9.5 Optimizer & lịch học (một cấu hình duy nhất)
- **AdamW**: `lr = 1e-4`, `betas = (0.9, 0.95)`, `weight_decay = 0.05`
- Warmup 2000 steps → cosine decay
- Gradient clipping: norm `1.0`
- Mixed precision **bf16**
- **EMA** weights, decay `0.9999` (dùng EMA cho eval/inference)
- Effective batch ≥ 32. Latent V-JEPA tốn RAM (~14 MB/sample cho past+future @2304×768) → dùng **gradient accumulation 4–8**.

---

## 10. Inference / ODE Solver (closed-loop NuPlan)

Closed-loop cần nhanh → tận dụng shortcut để đi ít bước.

```python
@torch.no_grad()
def rollout(model, past_ctx, steps=4):
    # khởi từ noise thuần (t=0)
    z = torch.randn(B, model.n_v, 768, device=dev)
    a = torch.randn(B, 8,        768, device=dev)
    d_step = 1.0 / steps
    for i in range(steps):
        t = torch.full((B,), i * d_step, device=dev)
        d = torch.full((B,), d_step,     device=dev)   # shortcut: bước lớn = nhanh
        v_z, v_a = model.core(torch.cat([z, a], 1), past_ctx, cond=model.cond(t, d))
        z = z + d_step * v_z       # Euler (dx/dt = v, đường thẳng)
        a = a + d_step * v_a
    waypoint = model.action_decoder(a)    # [B,8,2] — điều khiển xe
    return waypoint, z                     # z dùng cho viz nếu cần
```
- Mặc định `steps = 4` (shortcut). Có thể dùng Heun để chính xác hơn với chi phí gấp đôi.
- Chỉ `waypoint` được đưa vào planner NuPlan; `z` chỉ để debug/viz.

---

## 11. Visualize PCA Latent (debug tool, KHÔNG phải decoder)

Mục đích: kiểm tra định tính world model có học được cấu trúc không, bằng cách chiếu `[N_v,768]` xuống RGB 3-channel để xem như video. V-JEPA không có pixel decoder nên dùng PCA là đủ cho qualitative (đường/vỉa hè/xe ra cụm màu khác nhau).

```python
@torch.no_grad()
def visualize_latent_pca(z_clean, T_fut=8, tubelet=2, patch_grid=24, out_size=384):
    # z_clean: [B, N_v, 768] (sau ODE solve)
    B, N_v, Dn = z_clean.shape
    T_tube = T_fut // tubelet                                   # 4
    assert N_v == T_tube * patch_grid * patch_grid, f"{N_v} != {T_tube*patch_grid*patch_grid}"
    flat = z_clean.reshape(B*N_v, Dn)
    flat = flat - flat.mean(0, keepdim=True)
    U, S, V = torch.pca_lowrank(flat, q=3, niter=4)
    proj = flat @ V[:, :3]                                      # [B*N_v, 3]
    proj = (proj - proj.min(0, keepdim=True).values) / \
           (proj.max(0, keepdim=True).values - proj.min(0, keepdim=True).values + 1e-8)
    vis = proj.reshape(B, T_tube, patch_grid, patch_grid, 3)
    vis = vis.repeat_interleave(tubelet, dim=1)                 # [B,8,24,24,3]
    vis = vis.permute(0,1,4,2,3).reshape(B*8, 3, patch_grid, patch_grid)
    vis = F.interpolate(vis, size=(out_size, out_size), mode='bilinear', align_corners=False)
    vis = vis.reshape(B, 8, 3, out_size, out_size).permute(0,1,3,4,2)
    return vis.clamp(0, 1)                                      # [B,8,384,384,3]
```
- Normalize **global trên cả video** (không per-frame) để màu không nháy.
- **Khuyến nghị:** fit PCA offline trên ~1000 sample, cache `V` + `mean`, dùng cố định lúc viz → màu nhất quán giữa các batch.
- Chỉ gọi ở `validation_step` với batch nhỏ (B=2–4), lưu `logs/vis/step_{i}.mp4`. KHÔNG viz mỗi training step.

---

## 12. File structure đề xuất

```
project_root/
├── SPEC.md                          # file này
├── configs/
│   └── train.yaml                   # MỘT config (joint training, không còn phase1/phase2)
├── scripts/
│   ├── prepare_nuplan_cache.py      # precompute camera/pose/route/waypoint cache trước training
│   ├── precompute_pca_basis.py
│   └── eval_nuplan_closedloop.py
├── src/
│   ├── data/
│   │   ├── nuplan_dataset.py        # load past+future camera + ego + route + waypoint GT
│   │   └── transforms.py
│   ├── models/
│   │   ├── vjepa_wrapper.py         # Frozen V-JEPA 2.1 ViT-B, lazy N_v inference
│   │   ├── input_embedders.py       # NavMLP, EgoMLP, ActionMLP, CondEmbedder
│   │   ├── dit.py                   # DiTBlock (AdaLN-Zero) + FinalLayer + attention (RoPE, qk_norm)
│   │   ├── flow_matching.py         # noising, velocity target, shortcut (t,d) sampling + self-consistency
│   │   ├── action_decoder.py        # MLP head (train CHUNG)
│   │   └── full_model.py            # Orchestrator: forward end-to-end, slice, decode
│   ├── train/
│   │   ├── train.py                 # MỘT vòng training joint
│   │   └── losses.py                # L_flow_z, L_flow_a, L_wp, L_smooth, L_sc, tổng hợp
│   ├── inference/
│   │   ├── ode_solver.py            # Euler/Heun với shortcut
│   │   └── rollout.py               # closed-loop NuPlan
│   ├── viz/
│   │   └── pca_latent.py            # §11
│   └── utils/{ema.py, logging.py}
├── scripts/
│   ├── precompute_pca_basis.py
└── tests/
    ├── test_shapes.py
    ├── test_noising.py
    ├── test_joint_loss.py           # verify gradient chảy từ L_wp vào DiT
    └── test_pca_viz.py
```

---

## 13. Acceptance Checks (AI coding agent tự verify)

### 13.1 Shape sanity
```python
def test_shapes():
    B = 2
    model = FullModel()
    out = model(
        past_cam=torch.randn(B,8,3,384,384), fut_cam=torch.randn(B,8,3,384,384),
        route=torch.randn(B,20,2), ego=torch.randn(B,8,7),
        wp_gt=torch.randn(B,8,2), t=torch.rand(B), d=torch.zeros(B))
    assert out["v_z_pred"].shape == (B, model.n_v, 768)
    assert out["v_a_pred"].shape == (B, 8, 768)
    assert out["waypoint_pred"].shape == (B, 8, 2)      # decoder chạy CHUNG ngay trong forward
```

### 13.2 Frozen / Trainable đúng ở pha duy nhất
```python
assert all(not p.requires_grad for p in model.vjepa.parameters())          # V-JEPA frozen
assert any(p.requires_grad for p in model.dit.parameters())                # DiT trainable
assert any(p.requires_grad for p in model.action_decoder.parameters())     # decoder TRAINABLE (không đóng băng)
```

### 13.3 No-MLP-on-VJEPA
- Trong `full_model.py`, `camera_past` và `z_clean` phải đi thẳng từ output V-JEPA vào `cat()` mà KHÔNG qua Linear/MLP nào.

### 13.4 N_v consistency
- Forward 1 dummy ở init; assert `model.n_v == 2304` cho config 384×384, 8 frame, patch 16, tubelet 2.

### 13.5 Gradient liên thông (test cốt lõi của joint training)
```python
def test_waypoint_grad_reaches_dit():
    # backward CHỈ từ L_wp; phải thấy grad ở tham số DiT (qua v_a_pred -> a_hat_clean -> decoder)
    out = model(...); L = F.smooth_l1_loss(out["waypoint_pred"], wp_gt)
    L.backward()
    g = next(p.grad for p in model.dit.parameters() if p.grad is not None)
    assert g is not None and g.abs().sum() > 0      # chứng tỏ KHÔNG detach, joint thật sự
```

### 13.6 AdaLN identity tại init
```python
# Với ada_mod & final.linear init 0: output velocity ban đầu ~ 0
out = model(...); assert out["v_z_pred"].abs().mean() < 1e-5
```

### 13.7 PCA viz smoke
- `visualize_latent_pca(torch.randn(2,2304,768))` → `[2,8,384,384,3]` ∈ [0,1], không NaN.

---

## 14. Câu hỏi cần làm rõ (hỏi trước khi code nếu thiếu)

1. **Closed-loop adapter cụ thể?** Xác nhận API cuối cùng giữa `rollout()` và scorer/simulator NuPlan/nav2sim.
2. **Coordinate frame của waypoint** (ego-centric vs world?) — ảnh hưởng scale của Huber `beta`.
3. **Số bước ODE lúc inference** (mặc định 4 với shortcut).
4. **PCA basis precompute trên bao nhiêu sample** (mặc định 1000).
5. **`lambda_wp` khởi đầu** (mặc định 5.0; tune theo log loss).

---

## 15. Bảng quyết định thiết kế

| Quyết định | Lý do |
|---|---|
| **Train MỘT pha duy nhất (joint), bỏ Phase 1/Phase 2** | Đúng triết lý DreamZero: "single model end-to-end with joint prediction objective"; tách pha phá vỡ video-action alignment và làm decoder không học từ chất lượng DiT |
| **Action Decoder train CHUNG, decode từ `â = a_t+(1−t)·v_a_pred`** | Gradient task (NuPlan) chảy ngược vào DiT; nếu decode từ GT thì decoder chỉ đảo ngược ActionMLP (vô nghĩa) |
| **AdaLN-Zero để inject `(t,d)`, KHÔNG nối token** | Tín hiệu conditioning trực tiếp ở mọi layer/token; không bị pha loãng với sequence dài (~2304); hội tụ nhanh; chuẩn DiT/SD3/Flux/Shortcut-FM |
| `cond = t_embed + d_embed` (sum) | Additive đơn giản, đủ expressive |
| Init `ada_mod` + `final.linear` = 0 | Mạng bắt đầu là identity → training cực ổn định |
| 8 lớp DiT | Giảm tải tính toán, đủ cho ~110M |
| Không MLP cho V-JEPA latent | Giữ raw feature đã pretrain; world loss học đúng không gian đó |
| 768 dim xuyên suốt | Khớp tự nhiên V-JEPA 2.1 ViT-B |
| `N_v` suy động | Robust khi đổi resolution/frame count |
| Shortcut FM thay DDPM | Cho phép 1–4 step inference, closed-loop cần nhanh |
| Cross-attn KHÔNG modulate | Context (past) đã đủ informative; modulate cross gây nhiễu |
| PCA cho viz, không decoder pixel | Không cần train thêm component, đủ qualitative |
| Loss waypoint = Huber + smoothness bậc 2 | Bám quỹ đạo chuyên gia + triệt giật vô lăng cho NuPlan closed-loop |
