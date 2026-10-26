# 3D-FUTURE

安装和权重下载使用原仓库说明：
- TRELLIS2：[README.md](README.md)。
- Gen3DSR：[README.md](gen3dsr/README.md)、[INSTALL.md](gen3dsr/INSTALL.md)、[下载脚本](gen3dsr/external/checkpoints/download.sh)。

Gen3DSR 原说明中的仓库目录在这里对应 `gen3dsr/`，无需重复 clone。

以下命令均在仓库根目录执行，路径通过参数指定。
TRELLIS2 和 Gen3DSR 分别使用各自已安装的依赖环境。

## 推理

TRELLIS2：

```bash
python run_3dfuture_batch.py \
  --input-root data/3D-FUTURE/masked_images_test \
  --pretrained pretrained/TRELLIS.2-4B \
  --output-root outputs/trellis2 \
  --pipeline-type 1024_cascade
```

Gen3DSR（数据集 GT mask）：

```bash
python run_gen3dsr_batch.py \
  --input-root data/3D-FUTURE/masked_images_test \
  --gt-json data/3D-FUTURE-scene/GT/test_set.json \
  --checkpoint-dir pretrained/gen3dsr \
  --output-root outputs/gen3dsr
```

只推理前 200 个，在对应命令后加 `--limit 200`。

## 评测

全部已有场景，Gen3DSR 翻转配准、TRELLIS2 masked 配准：

```bash
python evaluate_scene_outputs.py \
  --trellis-dir outputs/trellis2 \
  --gen3dsr-dir outputs/gen3dsr \
  --gt-scene-dir data/3D-FUTURE/scene_test \
  --methods Gen3DSR_flipped_aligned TRELLIS2_scene_masked \
  --out-dir outputs/eval_all
```

只评 150 个，在上述命令后加 `--scene-id-file scene_ids_150.txt`，
并将 `--out-dir` 改为 `outputs/eval_150`。

保存前三个场景的可视化，加：
`--trellis-limit 3 --gen3dsr-limit 3 --save-aligned-method all --save-aligned-limit 3`。

汇总表：输出目录的 `scenegen_style_table.csv` / `.md`。
逐 ID 指标：`official_eval_raw/<方法>/per_scene_metrics.csv` / `.jsonl`。
