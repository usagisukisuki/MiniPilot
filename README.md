<p align="center">
  <img src="assets/minipilot_logo.svg" alt="MiniPilot logo" width="600">
</p>

<p align="center">
  <img src="assets/model_size_comparison.png" alt="Model size comparison" width="800">
</p>


## Model

| 名前 | Vision Encoder | Text Encoder | グリッド | `d_vis` | パラメータ (学習対象) |
|------|---------------|-------------|---------|---------|----------------------|
| **MiniPilot** | DINOv3 ViT-S/16 (`dinov3_vits`) | MiniLM-L6 (`minilm-l6`) | 14×14 (196) | 384 | ~4M |
| **NanoPilot** | ViT-Tiny/16 (`vit_tiny`) | MiniLM-L6 (`minilm-l6`) | 14×14 (196) | 192 | ~2M |
| **PicoPilot** | EfficientFormerV2-S0 (`efficientformerv2_s0`) | MiniLM-L6 (`minilm-l6`) | 7×7 (49) | 176 | ~1.1M |
| **FemtoPilot** | MobileViT-XXS (`mobilevit_xxs`) | MiniLM-L6 (`minilm-l6`) | 8×8 (64) | 320 | ~2M |

```bash
# MiniPilot
python train.py --config configs/config.yaml --vision-encoder dinov3_vits --text-encoder minilm-l6 --out ./ckpts_minipilot

# NanoPilot
python train.py --config configs/config.yaml --vision-encoder vit_tiny --text-encoder minilm-l6 --out ./ckpts_nanopilot

# PicoPilot
python train.py --config configs/config.yaml --vision-encoder efficientformerv2_s0 --text-encoder minilm-l6 --out ./ckpts_picopilot

# FemtoPilot
python train.py --config configs/config.yaml --vision-encoder mobilevit_xxs --text-encoder minilm-l6 --out ./ckpts_femtopilot
```

## Result

CORE benchmark の代替評価（`test.py`、`--no-steervit`）の結果。値は % で、太字は各列の最高値。
全モデル共通の学習設定: `--teacher bbox --dataset all --epochs 30`。

- **PA** (Pointing Accuracy): heatmap の argmax パッチの中心が GT bbox 内に入る割合 (↑)
- **AHP** (Avg Heatmap Precision): GT bbox 内に入る heatmap 質量の割合 (↑)
- **PA_wrong**: テキストをシャッフルしたときの PA。PA との差が大きいほどテキストに依存して位置を決めている

### PA (%)

| Model | Vision | Text | RefCOCO val | RefCOCO testA | RefCOCO testB | RefCOCO+ val | RefCOCO+ testA | RefCOCO+ testB | RefCOCOg val | VG val | GQA val |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **MiniPilot** | DINOv3 ViT-S/16 | MiniLM-L6 | **84.5** | **88.2** | **82.0** | **72.0** | **78.3** | **62.6** | **79.2** | **73.8** | **59.4** |
| NanoPilot | ViT-Tiny/16 | MiniLM-L6 | 79.0 | 79.4 | 77.6 | 64.5 | 69.4 | 58.5 | 71.4 | 64.9 | 51.7 |

### AHP (%)

| Model | Vision | Text | RefCOCO val | RefCOCO testA | RefCOCO testB | RefCOCO+ val | RefCOCO+ testA | RefCOCO+ testB | RefCOCOg val | VG val | GQA val |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **MiniPilot** | DINOv3 ViT-S/16 | MiniLM-L6 | **76.0** | **79.1** | **73.1** | **64.5** | **71.1** | **56.0** | **69.7** | **63.7** | **53.5** |
| NanoPilot | ViT-Tiny/16 | MiniLM-L6 | 67.3 | 68.2 | 66.4 | 54.3 | 57.6 | 50.0 | 59.5 | 53.6 | 44.8 |

### PA_wrong (%)

| Model | Vision | Text | RefCOCO val | RefCOCO testA | RefCOCO testB | RefCOCO+ val | RefCOCO+ testA | RefCOCO+ testB | RefCOCOg val | VG val | GQA val |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **MiniPilot** | DINOv3 ViT-S/16 | MiniLM-L6 | 20.9 | 25.0 | 22.3 | 23.7 | 26.1 | 26.6 | 30.5 | 23.3 | 17.5 |
| NanoPilot | ViT-Tiny/16 | MiniLM-L6 | 22.5 | 24.9 | 22.8 | 24.6 | 27.4 | 26.1 | 30.3 | 23.3 | 17.3 |

- MiniPilot (DINOv3) が全データセットで最も高く、NanoPilot (ViT-Tiny) に対して PA は RefCOCO で +4〜9pt、VG で +8.9pt、GQA で +7.7pt。
- NanoPilot の RefCOCO+ / RefCOCOg / GQA はテキストログ (`logs/test_output_minilm_l6_*.txt`) から転記。

### モデルサイズ・計算量 (vs. SteerViT)

| モデル | Vision / Text | 総パラメータ | 学習対象 | FLOPs (B=1) | RefCOCO val PA |
|---|---|---:|---:|---:|---:|
| SteerViT (元モデル) | DINOv2 ViT-B/14 / RoBERTa-large | 464.4M | 23.1M | 135.76G (336px) | — |
| **MiniPilot** | DINOv3 ViT-S/16 / MiniLM-L6 | 53.3M (11.5%) | 9.03M | 11.55G (8.5%) | 84.5 |
| NanoPilot | ViT-Tiny/16 / MiniLM-L6 | 30.5M (6.6%) | 2.30M | 3.38G (2.5%) | 79.0 |

- (%) は SteerViT 比。学習対象 = connector + GatedCrossAttn + lin_seg_head。
- 値は `test.py` が出力するパラメータ数 / FLOPs 表から転記。
- SteerViT のパラメータの 76% はテキストエンコーダ (RoBERTa-large 355M)。MiniPilot / NanoPilot でもテキスト側 (MiniLM-L6 22.7M) が総パラメータの 4〜7 割を占める。
- SteerViT 自体の精度は未評価（全評価が `--no-steervit` で実行されているため）。

---

## アーキテクチャ

```
テキスト入力
  └─ テキストエンコーダ (frozen, --text-encoder で選択)
       └─ connector: Linear(d_text → d_vis)  ★学習
            └─ GatedCrossAttn × n_blocks  ★学習
                 └─ ビジョンバックボーン (pretrained, frozen, --vision-encoder で選択)
                      └─ lin_seg_head: Linear(d_vis → 1)  ★学習
                           └─ patch logits (grid×grid patches)

損失: SoftCE(student_logits, teacher_soft)
```

| コンポーネント | パラメータ数 | 学習 |
|--------------|------------|------|
| ビジョンバックボーン | 下表参照 | ✗ frozen (ImageNet pretrained) |
| テキストエンコーダ | 下表参照 | ✗ frozen |
| connector (d_text→d_vis) | d_text × d_vis | ✓ |
| GatedCrossAttn × n | ~1–3M | ✓ |
| lin_seg_head | d_vis + 1 | ✓ |

---

## ビジョンエンコーダ (`--vision-encoder`)

`--vision-encoder` でバックボーンを切り替えられる。

| キー | モデル | 入力解像度 | グリッド | `d_vis` | GCA 挿入箇所 |
|------|--------|-----------|---------|---------|-------------|
| `vit_tiny` (デフォルト) | `vit_tiny_patch16_224` | 224×224 | 14×14 (196) | 192 | 各 Transformer ブロック後 (×12) |
| `dinov3_vits` | `vit_small_patch16_dinov3.lvd1689m` | 224×224 | 14×14 (196) | 384 | 各 Transformer ブロック後 (×12) |
| `efficientformerv2_s0` | `efficientformerv2_s0.snap_dist_in1k` | 224×224 | 7×7 (49) | 176 | 各ブロック後 (×14: s0×2, s1×2, s2×6, s3×4) |
| `fastvit_t8` | `fastvit_t8.apple_dist_in1k` | 256×256 | 8×8 (64) | 768 | 各ステージ後 (×4) |
| `mobilevit_xxs` | `mobilevit_xxs` | 256×256 | 8×8 (64) | 320 | 各ステージ後 (×5) |

- `vit_tiny` / `dinov3_vits`: patch token ベースの ViT。CLS トークンあり (num_prefix=1)
- `efficientformerv2_s0`: 階層 ViT (stem + 4 stages)。前半ステージは MLP-only、後半に Attention2d。CLS トークンなし (num_prefix=0)
- `fastvit_t8` / `mobilevit_xxs`: CNN ライクな stem+stages+final_conv 構造。CLS トークンなし (num_prefix=0)

ビジョンエンコーダを変えると `connector` のサイズ (`d_text → d_vis`) と GCA 数が自動的に変わる。  
チェックポイントに保存され、`test.py` ロード時に自動復元される。

```bash
python train.py --config configs/config.yaml --vision-encoder efficientformerv2_s0 --out ./ckpts_picopilot
python train.py --config configs/config.yaml --vision-encoder fastvit_t8 --out ./ckpts_fastvit
python train.py --config configs/config.yaml --vision-encoder mobilevit_xxs --out ./ckpts_mobilevit
```

---

## テキストエンコーダ (`--text-encoder`)

| キー | モデル | `d_text` | パラメータ数 |
|------|--------|----------|------------|
| `distilroberta` (デフォルト) | `distilbert/distilroberta-base` | 768 | ~82M |
| `tinybert-l6` | `huawei-noah/TinyBERT_General_6L_768D` | 768 | ~67M |
| `tinybert-l4` | `huawei-noah/TinyBERT_General_4L_312D` | 312 | ~14M |
| `minilm-l6` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | ~23M |

テキストエンコーダを変えると `connector` のサイズ (`d_text → 192`) も自動的に変わる。  
チェックポイントにはテキストエンコーダの情報が保存され、`test.py` ロード時に自動復元される。

---

## Teacher の種類 (`--teacher`)

`--teacher` オプションで teacher の soft target を切り替えられる。

### `steervit` (デフォルト)

SteerViT pretrained の heatmap を teacher とする。

```
full_img (224px)
  → bicubic resize → 336px
  → SteerViT (DINOv2-base, frozen)
  → patch logits (24×24 = 576)
  → softmax → bilinear resize → 14×14 → teacher_soft
```

- 利点: テキストと画像の対応を学習済みモデルが生成する高精度な soft target
- 欠点: SteerViT のロードに GPU メモリが必要 (~1.5GB)

### `bbox`

GT bbox の patch overlap 面積を正規化した分布を teacher とする。

```
bbox [x1, y1, x2, y2] (相対座標)
  → 各 patch (14×14) との intersection 面積を計算
  → 面積で正規化 → teacher_soft
```

- 利点: SteerViT 不要・高速・アノテーション忠実
- 欠点: bbox 内を一様に扱うためテキストの意味は反映されない

### `both`

SteerViT heatmap と bbox mask の平均を teacher とする。

```
teacher_soft = 0.5 × steervit_soft + 0.5 × bbox_soft
```

- 利点: テキスト意味と GT 位置の両方を supervision に使える

---

## 環境

```bash
conda activate steerdet
```

必要パッケージ: `torch`, `timm >= 1.0`, `transformers`, `torchvision`, `pycocotools`, `tqdm`, `matplotlib`, `pyyaml`

---

## データセット

| データセット | パス | 備考 |
|-------------|------|------|
| RefCOCO | `/data2/kato/dataset/coco/refcoco/` | `refs(unc).p` + `instances.json` |
| COCO 2017 | `/data2/kato/dataset/coco/` | `train2017/`, `val2017/`, `annotations/` |
| Visual Genome | `/data2/kato/dataset/vg/` | `region_descriptions.json` |
| GQA | `/data2/kato/dataset/GQA/` | `*_sceneGraphs.json` |

パスは `config.yaml` の `paths` セクションで変更できる。

---

## Config ファイル (`config.yaml`)

全パラメータとパスを `config.yaml` で管理できる。  
**優先順位: CLI 引数 > config.yaml > argparse デフォルト値**

```yaml
paths:
  coco_root:     "/data2/kato/dataset/coco"
  vg_root:       "/data2/kato/dataset/vg"
  gqa_root:      "/data2/kato/dataset/GQA"
  steervit_ckpt: "/path/to/steervit_dinov2_base.pth"

train:
  text_encoder: "distilroberta"   # distilroberta | tinybert-l6 | tinybert-l4 | minilm-l6
  teacher:      "steervit"        # steervit | bbox | both
  dataset:      "refcoco"
  epochs:       50
  batch_size:   64
  lr:           3.0e-4
  out:          "./ckpts"
  ...

test:
  ckpt:       null   # test.py の --ckpt デフォルト値
  dataset:    "refcoco"
  batch_size: 64
  ...
```

```bash
# config.yaml をベースに実行 (CLI 引数で上書き可)
python train.py --config configs/config.yaml

# config の一部だけ上書き
python train.py --config configs/config.yaml --epochs 30 --batch-size 128 --text-encoder tinybert-l4
```

---

## 学習

```bash
# config.yaml のデフォルト設定で実行
python train.py --config configs/config.yaml

# テキストエンコーダを変えて学習
python train.py --config configs/config.yaml --text-encoder tinybert-l4 --out ./ckpts_tinybert_l4
python train.py --config configs/config.yaml --text-encoder tinybert-l6 --out ./ckpts_tinybert_l6
python train.py --config configs/config.yaml --text-encoder minilm-l6   --out ./ckpts_minilm_l6

# bbox mask teacher (SteerViT 不要・高速)
python train.py --config configs/config.yaml --teacher bbox

# ビジョンエンコーダを切り替えて学習
python train.py --config configs/config.yaml --vision-encoder dinov3_vits         --out ./ckpts_dinov3_vits
python train.py --config configs/config.yaml --vision-encoder efficientformerv2_s0 --out ./ckpts_picopilot
python train.py --config configs/config.yaml --vision-encoder fastvit_t8           --out ./ckpts_fastvit
python train.py --config configs/config.yaml --vision-encoder mobilevit_xxs        --out ./ckpts_mobilevit

# 教師キャッシュを使った学習 (SteerViT をロードしない)
python precompute_teacher.py --dataset refcoco --out ./teacher_cache/refcoco       # 事前計算 (初回のみ)
python train.py --config configs/config.yaml --teacher-cache ./teacher_cache/refcoco       # heatmap 蒸留のみ
python train.py --config configs/config.yaml --teacher-cache ./teacher_cache/refcoco --attn-distil  # attn distil も

# 全データセットで学習
python train.py --config configs/config.yaml --dataset all --batch-size 512 --teacher bbox
```

デフォルト設定:

| 設定 | 値 |
|------|---|
| backbone | vit_tiny_patch16_224 (ImageNet pretrained) |
| text encoder | `distilroberta` (frozen) |
| teacher | `steervit` |
| full image size (student) | 224×224 |
| full image size (teacher) | 336×336 (内部で bicubic リサイズ) |
| epochs | 50 |
| batch size | 64 |
| lr | 3e-4 (CosineAnnealing → 3e-6) |
| optimizer | AdamW (weight_decay=1e-2) |

---

## Attention Distillation (`--attn-distil`)

`--teacher steervit` または `--teacher both` のとき、出力 heatmap に加えて各 Transformer ブロックの **Self-Attention (SA)** と **Cross-Attention (CA)** の分布も teacher から蒸留できる。

### 蒸留する情報

| 種類 | 取得元 | 集約方法 |
|------|--------|----------|
| **SA** | 各ブロックの ViT self-attention | CLS→patch 注意を head 平均 → (B, N_patches) |
| **CA** | 各ブロックの GatedCrossAttn | patch→text 注意を head 平均 + text token 平均 → (B, N_patches) |

Teacher (24×24) と Student (各モデルのグリッドサイズ: vit_tiny=14×14, efficientformerv2_s0=7×7 等) の空間サイズは bilinear 補間で整合し、KL-div で蒸留する。

### 損失の構成

```
total_loss = heatmap_loss
           + attn_weight × (sa_weight × SA_loss + ca_weight × CA_loss)
```

### 使用例

```bash
# デフォルト重み (attn_weight=1.0, sa=1.0, ca=1.0)
python train.py --config configs/config.yaml --teacher steervit --attn-distil

# heatmap : attn = 1 : 0.5, SA のみ
python train.py --config configs/config.yaml --teacher steervit --attn-distil \
    --attn-weight 0.5 --ca-weight 0.0

# both モードでも使用可
python train.py --config configs/config.yaml --teacher both --attn-distil \
    --attn-weight 1.0 --sa-weight 1.0 --ca-weight 0.5
```

> **注意**: `--teacher bbox` のときは attn-distil は無視される（SteerViT 不使用のため）。

### ログ出力

`--attn-distil` 有効時は各エポックのログに内訳が追記される:

```
[  1/ 50] train=16.01  val=15.83  lr=3.00e-04  12.3s  hmap=5.46  sa=5.28  ca=5.28
```

`history.json` にも `last_heatmap_loss` / `last_sa_loss` / `last_ca_loss` が記録される。

### 速度・メモリへの影響

attn-distil を有効にすると、通常の `--teacher steervit` に比べて以下のオーバーヘッドが生じる:

| 項目 | 影響 |
|------|------|
| Teacher forward 回数 | 変化なし（heatmap と attn を 1 回の forward で同時取得） |
| Teacher forward の速度 | Flash Attention が全ブロックで無効化されるため遅くなる |
| Student forward の速度 | 全ブロックで attention map を収集するため遅くなる |
| VRAM | 全ブロックの attention map を保持するため増加 |

→ 学習を高速化したい場合は **教師キャッシュ** (`precompute_teacher.py`) を使うこと。

---

## 教師キャッシュ (`precompute_teacher.py`)

`--teacher steervit` や `--attn-distil` を使うと毎バッチ SteerViT を実行するため学習が遅くなる。  
`precompute_teacher.py` で **SteerViT の出力を事前計算・保存**しておくと、学習中は SteerViT をロードせずにキャッシュから読むだけになる。

### 保存されるファイル

| ファイル | 内容 | 形状 | dtype |
|---------|------|------|-------|
| `soft.dat` | softmax heatmap | (N, 576) | float16 |
| `sa.dat` | SA 注意の圧縮分布 | (N, n_blocks, 576) | float16 |
| `ca.dat` | CA 注意の圧縮分布 | (N, n_blocks, 576) | float16 |
| `meta.json` | メタデータ | — | JSON |

`--no-attn` を付けると `sa.dat` / `ca.dat` を省略（heatmap 蒸留のみのとき推奨）。

SA/CA の圧縮方法:
- **SA**: CLS→patch 注意をヘッド平均 → 正規化 → (576,) per block
- **CA**: patch→text 注意をヘッド平均 + テキスト平均 → 正規化 → (576,) per block

### ステップ 1: キャッシュ生成

**ローカル** (`precompute_teacher.py`):

```bash
# heatmap + SA/CA を保存（--attn-distil と組み合わせる場合）
python precompute_teacher.py \
    --dataset refcoco \
    --out ./teacher_cache/refcoco \
    --split train

# heatmap のみ（高速、attn distil なしで使う場合）
python precompute_teacher.py \
    --dataset refcoco \
    --out ./teacher_cache/refcoco_noattn \
    --split train \
    --no-attn

# val set も同様に
python precompute_teacher.py --dataset refcoco --out ./teacher_cache/refcoco_val --split val
```

生成後のディレクトリ構造:

```
teacher_cache/
  refcoco/    meta.json  soft.dat  sa.dat  ca.dat
  refcoco+/   meta.json  soft.dat  ...
  refcocog/   ...
  vg/         ...
  gqa/        ...
```

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--dataset` | (必須) | `refcoco` / `refcoco+` / `refcocog` / `vg` / `gqa` / `coco` |
| `--out` | (必須) | 出力ディレクトリ |
| `--split` | `train` | `train` / `val` |
| `--no-attn` | off | SA/CA を保存しない (soft heatmap のみ) |
| `--steervit-ckpt` | config | SteerViT チェックポイントパス |
| `--batch-size` | `32` | バッチサイズ |
| `--num-workers` | `4` | DataLoader ワーカー数 |

### ステップ 2: キャッシュを使って学習

**ローカル** (`train.py`):

```bash
# heatmap 蒸留のみ（SteerViT をロードしない）
python train.py --config configs/config.yaml \
    --dataset refcoco \
    --teacher-cache ./teacher_cache/refcoco_noattn

# attn distil もキャッシュから（SA/CA 込みで生成したキャッシュが必要）
python train.py --config configs/config.yaml \
    --dataset refcoco \
    --teacher-cache ./teacher_cache/refcoco \
    --attn-distil
```

`--teacher-cache` を指定すると:
- SteerViT を**ロードしない**（起動高速化・VRAM 節約）
- `fix_text=True` でテキスト選択を決定論的にしてキャッシュと一致させる
- バッチにサンプルインデックスを付加してキャッシュを正しく参照する
- `--dataset all` のときは `MultiTeacherCache` が自動的に各サブディレクトリを結合する

`--dataset all` 時のキャッシュディレクトリ構造:

```
teacher_cache/          ← --teacher-cache に渡すルートディレクトリ
  refcoco/   meta.json  soft.dat  [sa.dat  ca.dat]
  refcoco+/  ...
  refcocog/  ...
  vg/        ...
  gqa/       ...
```

> **注意 1**: キャッシュは **train split のみ**を対象にすること。  
> val set は元々 SteerViT を実行しないので高速（キャッシュ不要）。  
> **注意 2**: `--dataset all` + `--teacher-cache` を使う場合は `--max-train` を指定しないこと。  
> `max_train` を指定すると各サブデータセットのサイズが変わり、キャッシュのインデックスとずれが生じる。

---

## オプション一覧

### train.py

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--config` | — | YAML config ファイル (CLI 引数で上書き可) |
| `--text-encoder` | `distilroberta` | `distilroberta` / `tinybert-l6` / `tinybert-l4` / `minilm-l6` |
| `--vision-encoder` | `vit_tiny` | `vit_tiny` / `dinov3_vits` / `efficientformerv2_s0` / `fastvit_t8` / `mobilevit_xxs` |
| `--teacher` | `steervit` | teacher の種類: `steervit` / `bbox` / `both` |
| `--no-pretrained-backbone` | — | ビジョンバックボーンも完全ランダム初期化 |
| `--dataset` | `refcoco` | `refcoco` / `refcoco+` / `refcocog` / `vg` / `gqa` / `coco` / `all` |
| `--epochs` | `50` | 学習エポック数 |
| `--batch-size` | `64` | バッチサイズ |
| `--val-batch-size` | `16` | val loader のバッチサイズ |
| `--lr` | `3e-4` | 初期学習率 |
| `--weight-decay` | `1e-2` | AdamW weight decay |
| `--amp` | on | Mixed precision (bf16 対応 GPU では bf16、それ以外は FP16 + GradScaler) |
| `--resume` | — | チェックポイントから再開 |
| `--out` | `./ckpts` | 出力ディレクトリ |
| `--vis-every` | `5` | N エポックごとに heatmap 可視化 (0 で無効) |
| `--vis-n` | `4` | val set から自動収集する probe 数 |
| `--vis-samples` | — | カスタム probe JSON |
| `--attn-distil` | off | SA / CA attention map の蒸留を有効化 |
| `--attn-weight` | `1.0` | attention distil loss 全体の重み |
| `--sa-weight` | `1.0` | SA loss の重み |
| `--ca-weight` | `1.0` | CA loss の重み |
| `--teacher-cache` | — | 事前計算済み教師キャッシュのディレクトリ (`precompute_teacher.py` で生成) |
| `--coco-root` | (config) | COCO データセットのルートパス |
| `--vg-root` | (config) | Visual Genome のルートパス |
| `--gqa-root` | (config) | GQA のルートパス |
| `--steervit-ckpt` | (config) | SteerViT チェックポイントパス |

### test.py

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--config` | — | YAML config ファイル |
| `--ckpt` | (必須) | TinySteerViT チェックポイント (.pth) |
| `--dataset` | `refcoco` | `refcoco` / `refcoco+` / `refcocog` / `vg` / `gqa` |
| `--split` | 全 split | `val` / `testA` / `testB` (refcocog・vg・gqa は `val` のみ) |
| `--max-samples` | 全件 | 各 split の最大サンプル数 |
| `--batch-size` | `64` | バッチサイズ |
| `--num-workers` | `4` | DataLoader ワーカー数 |
| `--no-steervit` | — | SteerViT teacher との比較をスキップ |
| `--no-wrong-prompt` | — | Wrong-prompt 制御実験をスキップ |
| `--vision-encoder` | (ckpt) | チェックポイントに保存された値を上書き |
| `--text-encoder` | (ckpt) | チェックポイントに保存された値を上書き |
| `--log` | 自動生成 | 結果を保存する JSON ファイルパス |
| `--coco-root` | (config) | COCO データセットのルートパス |
| `--vg-root` | (config) | Visual Genome のルートパス |
| `--gqa-root` | (config) | GQA のルートパス |
| `--steervit-ckpt` | (config) | SteerViT チェックポイントパス |

---

## 出力

```
ckpts/
  train.log          # タイムスタンプ付きログ
  history.json       # エポックごとの train/val loss
  best_model.pth     # val_loss 最良チェックポイント
  vis/
    epoch_000.png    # 学習前 (初期状態)
    epoch_005.png
    ...
```

`best_model.pth` の構造:

```python
{
    "epoch":    int,
    "sv_state": dict,   # connector + gated_cross_attn + lin_seg_head の state_dict
    "val_loss": float,
    "args":     dict,   # text_encoder を含む学習時の全引数
}
```

再開:

```bash
python train.py --config configs/config.yaml --resume ./ckpts/best_model.pth
```

---

## 可視化

各エポックの PNG の列構成は `--teacher` によって変わる:

| `--teacher` | 列1 | 列2 | 列3 | 列4 |
|------------|-----|-----|-----|-----|
| `steervit` | full image + bbox | teacher (SteerViT) | student | — |
| `bbox`     | full image + bbox | teacher (bbox mask) | student | — |
| `both`     | full image + bbox | teacher (SteerViT) | teacher (bbox mask) | student |

- **full image + bbox**: 入力画像にテキストと GT bbox を黄枠で描画
- **teacher (SteerViT)**: SteerViT が 336px 入力で生成した heatmap (24×24 grid)
- **teacher (bbox mask)**: GT bbox の patch overlap を正規化した分布 (14×14 grid)
- **student**: TinySteerViT の出力 heatmap (14×14 grid)

カスタム probe を使う場合:

```json
[
  {"img_path": "/path/to/image.jpg", "text": "person on the left", "bbox": [0.1, 0.2, 0.6, 0.9]},
  {"img_path": "/path/to/image2.jpg", "text": "red car"}
]
```

`bbox` は `[x1, y1, x2, y2]` 相対座標。省略時は `[0, 0, 1, 1]`（画像全体）を使用。

```bash
python train.py --config configs/config.yaml --vis-samples probes.json --vis-every 1
```

---

## 評価 (CORE Benchmark)

論文 (arXiv:2604.02327) の CORE (COnditional REtrieval) ベンチマークの代替評価として、
RefCOCO / RefCOCO+ / RefCOCOg / Visual Genome / GQA の Pointing Accuracy を計測する。

テキストエンコーダはチェックポイントから自動復元される。

### 評価指標

| 指標 | 説明 |
|------|------|
| **PA** (Pointing Accuracy) | argmax heatmap patch の中心が GT bbox 内なら hit |
| **AHP** (Avg Heatmap Precision) | GT bbox 内の heatmap 質量の割合 |
| **PA_wrong** | テキストをシャッフルしたときの PA |
| **ΔPA** | PA − PA_wrong（大きいほどテキスト依存） |

### 実行例

```bash
# config.yaml のパスを使って評価
python test.py --config configs/config.yaml --ckpt ./ckpts_picopilot_all/best_model.pth

# split を指定
python test.py --config configs/config.yaml --ckpt ./ckpts_picopilot_all/best_model.pth --split val

# データセットを切り替え
python test.py --config configs/config.yaml --ckpt ./ckpts_picopilot_all/best_model.pth --dataset refcoco+
python test.py --config configs/config.yaml --ckpt ./ckpts_picopilot_all/best_model.pth --dataset vg
python test.py --config configs/config.yaml --ckpt ./ckpts_picopilot_all/best_model.pth --dataset gqa

# サンプル数を制限して高速確認
python test.py --config configs/config.yaml --ckpt ./ckpts_picopilot_all/best_model.pth --max-samples 500

# SteerViT との比較をスキップ (TinySteerViT のみ)
python test.py --config configs/config.yaml --ckpt ./ckpts_picopilot_all/best_model.pth --no-steervit
```

---

## ファイル構成

```
MiniPilot/
  configs/                # 全パラメータ・パスのデフォルト設定 (config.yaml / config_tiny.yaml / config_distill.yaml)
  model.py                # GatedCrossAttn / TinyViTBackbone / EfficientFormerV2Backbone
                          # FastViTBackbone / MobileViTXXSBackbone / TinySteerViT
                          # TeacherCache / MultiTeacherCache
                          # AttnDistilLoss / HeatmapSoftCELoss
                          # TEXT_ENCODERS / VISION_ENCODERS
  dataset.py              # RefCOCOLocalDataset / COCODetDataset / VGRegionDataset
                          # GQASceneGraphDataset / _IndexedWrapper / build_loaders
  train.py                # 学習スクリプト
  test.py                 # CORE benchmark 評価スクリプト
  requirements.txt        # requirements
  train.sh                # 学習の実行例 (bash train.sh <minipilot|nanopilot|picopilot|femtopilot>)
  test.sh                 # 評価の実行例 (bash test.sh <ckpt_path>)
```
