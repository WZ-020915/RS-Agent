# model_wrappers 下载、部署与使用说明

这份文档给拿到 `/root/autodl-tmp/model_wrappers` 代码的同事使用。代码仓库本身不应提交模型权重、外部模型源码、大型数据集和测试输出；同事需要按本文把依赖项目、权重和数据集放回约定位置，或者修改 `backends/*.py` 里的路径常量。

当前 `model_wrappers` 暴露 6 个 wrapper：

- `skyeyegpt`: 遥感图像描述、VQA、grounding、视觉对话
- `sarmae`: SAR 目标检测、SAR 语义分割
- `mtp`: 光学遥感目标检测，支持水平框和旋转框
- `dofa`: 遥感语义分割
- `dehazeformer`: RGB 图像去雾/去云雾复原
- `sattxt`: 遥感图像零样本分类、图文检索

## 目录约定

默认代码写死了这些外部路径：

```text
/root/autodl-tmp/model_wrappers
/root/autodl-tmp/SkyEyeGPT
/root/autodl-tmp/MiniGPT-4-main
/root/autodl-tmp/SARMAE-main
/root/autodl-tmp/mmrotate
/root/autodl-tmp/MTP-main
/root/autodl-tmp/DOFA-master
/root/autodl-tmp/DehazeFormer
/root/autodl-tmp/sattxt
/root/autodl-tmp/dataset
```

如果同事不使用这些路径，需要改：

- `backends/skyeyegpt_backend.py`
- `backends/sarmae_backend.py`
- `backends/mtp_backend.py`
- `backends/dofa_backend.py`
- `backends/dehazeformer_backend.py`
- `backends/sattxt_backend.py`

## 快速使用

把 `/root/autodl-tmp` 加入 Python path：

```python
import sys
sys.path.insert(0, "/root/autodl-tmp")

from model_wrappers import skyeyegpt, sarmae, mtp, dofa, dehazeformer, sattxt
```

示例：

```python
# SkyEyeGPT caption
print(skyeyegpt(
    image_path="/path/to/remote_sensing_image.png",
    task="caption",
    prompt="Give a concise caption in one complete sentence.",
))

# SARMAE SAR detection
print(sarmae(
    image_path="/path/to/sar_image.jpg",
    task="detect",
))

# MTP optical object detection
print(mtp(
    image_path="/path/to/optical_image.jpg",
    task="rotated",
    target_class="ship",
))

# DOFA segmentation
print(dofa(
    dataset="m-pv4ger-seg",
    input_path="/path/to/sample.hdf5",
))

# DehazeFormer
print(dehazeformer(
    image_path="/path/to/hazy_rgb_image.png",
))

# SATtxt zero-shot classification
print(sattxt(
    task="zero_shot_classification",
    image_paths="/path/to/image.jpg",
    categories=["AnnualCrop", "Forest", "Residential", "River", "SeaLake"],
))
```

## SkyEyeGPT

任务：

- 遥感图像 caption
- 遥感 VQA
- visual grounding/referring
- 多轮视觉对话

wrapper 入口：

- Python: `skyeyegpt(...)`
- 后端: `backends/skyeyegpt_backend.py`
- Conda env: `minigptv`

需要的外部源码：

- SkyEyeGPT 官方仓库: `https://github.com/ZhanYang-nwpu/SkyEyeGPT`
- MiniGPT-4/MiniGPT-v2 运行时代码: `https://github.com/Vision-CAIR/MiniGPT-4`

需要的权重：

| 用途 | 文件名/目录 | 默认放置路径 | 下载来源 |
| --- | --- | --- | --- |
| SkyEyeGPT 主权重 | `SkyEyeGPT.pth` | `/root/autodl-tmp/SkyEyeGPT/SkyEyeGPT.pth` | SkyEyeGPT 官方 Hugging Face/模型页；本地 `SkyEyeGPT/README.md` 只保留了 model card metadata，未写直接权重 URL |
| MiniGPT-v2 依赖权重 | 由 `minigptv2_eval.yaml` 指定 | `/root/autodl-tmp/MiniGPT-4-main` 相关目录 | MiniGPT-4 官方说明 |

对应数据集名称：

- SkyEye-968k: `ZhanYang-nwpu/SkyEye-968k`
- VRSBench eval files used by tests: `VRSBench_EVAL_Cap.json`, `VRSBench_EVAL_vqa.json`, `VRSBench_EVAL_referring.json`

测试数据放置约定：

```text
/root/autodl-tmp/dataset/skyeyegpt/Images_val/*.png
/root/autodl-tmp/dataset/skyeyegpt/VRSBench_EVAL_Cap.json
/root/autodl-tmp/dataset/skyeyegpt/VRSBench_EVAL_vqa.json
/root/autodl-tmp/dataset/skyeyegpt/VRSBench_EVAL_referring.json
```

## SARMAE

任务：

- `task="detect"`: SAR 目标检测
- `task="segment"`: SAR 语义分割

wrapper 入口：

- Python: `sarmae(...)`
- 后端: `backends/sarmae_backend.py`
- Conda env: detection 使用 `sarmae`，segmentation 使用兼容 mmseg 的环境

需要的外部源码：

- SARMAE 官方仓库: `https://github.com/MiliLab/SARMAE`
- mmrotate: `https://github.com/open-mmlab/mmrotate`
- mmsegmentation: `https://github.com/open-mmlab/mmsegmentation`

公开下载来源：

- SARMAE 预训练权重 Hugging Face: `https://huggingface.co/Wenquandan777/SARMAE`
- SAR-1M 数据集 Hugging Face: `https://huggingface.co/datasets/Wenquandan777/SAR-1M`
- SARMAE README 也提供百度网盘：
  - 数据集: `https://pan.baidu.com/s/1ok4QCfeTVSJlPpAuLxEVxQ?pwd=0717`
  - 预训练权重: `https://pan.baidu.com/s/1DOsZolLZ--gMuNUgUXeyVg?pwd=0717`

当前 wrapper 需要的 finetune 权重：

| 任务 | 文件 | 默认放置路径 | 说明 |
| --- | --- | --- | --- |
| SAR detection on SSDD | `detect_epoch_34.pth` | `/root/autodl-tmp/SARMAE-main/weights/detect_epoch_34.pth` | 这是本 wrapper 实际加载的检测 checkpoint；如果官方仓库未提供同名文件，需要从团队资产或训练输出拷贝 |
| SAR segmentation on Raw AIR-PolarSAR-Seg | `seg_iter_20000.pth` | `/root/autodl-tmp/SARMAE-main/weights/seg_iter_20000.pth` | 这是本 wrapper 实际加载的分割 checkpoint；如果官方仓库未提供同名文件，需要从团队资产或训练输出拷贝 |

配置文件：

```text
/root/autodl-tmp/mmrotate/configs/SARMAE/SSDD/vitb_ssdd_local.py
/root/autodl-tmp/SARMAE-main/SARMAE_Fintune/Segmentation/work_dirs/vit-b-airseg-polar-20260525_132443/vit-b-airseg-polar-20260525.py
```

对应数据集名称：

- SAR-1M: 预训练数据
- SSDD: SAR ship detection
- HRSID: dataset builder 里用于 SAR detection 测试样本
- Raw AIR-PolarSAR-Seg: SAR segmentation

测试数据放置约定：

```text
/root/autodl-tmp/dataset/sarmae/HRSID_JPG/JPEGImages/*.jpg
/root/autodl-tmp/dataset/sarmae/HRSID_JPG/annotations/*.json
/root/autodl-tmp/dataset/sarmae/seg/test-00000-of-00001_extracted/images/*.png
/root/autodl-tmp/dataset/sarmae/seg/test-00000-of-00001_extracted/labels/*.png
```

## MTP

任务：

- 光学遥感目标检测
- `task="horizontal"`: 水平框检测
- `task="rotated"`: 旋转框检测
- `task="both"`: 同时跑水平框和旋转框

wrapper 入口：

- Python: `mtp(...)`
- 后端: `backends/mtp_backend.py`
- Conda env: `mtp`

需要的外部源码：

- MTP 官方仓库: `https://github.com/ViTAE-Transformer/MTP`
- SAMRS 数据说明: `https://github.com/ViTAE-Transformer/SAMRS`

公开下载来源：

- MTP README 的模型下载区提供 OneDrive/Baidu 链接。
- 预训练模型表使用：
  - Baidu: `https://pan.baidu.com/s/1Zh6yv2AouboGEP4phyR7xA?pwd=yqv9`
  - OneDrive: `https://1drv.ms/f/s!AimBgYV7JjTlgcpa7t2sywuWOm3HQA?e=LAh8WN`
- DIOR 水平检测 finetune 权重所在表项使用：
  - Baidu: `https://pan.baidu.com/s/1yiJISQYg0Xl84PvZr_r84w?pwd=ag0x`
  - OneDrive: `https://1drv.ms/f/s!AiSncQLqo7V6gUNIOKO-VtlKyT4d?e=NXA4Nw`
- DIOR-R 旋转检测 finetune 权重所在表项使用：
  - Baidu: `https://pan.baidu.com/s/1K7yCPmr1kGd--QRWnaEjMg?pwd=1o98`
  - OneDrive: `https://1drv.ms/f/s!AiSncQLqo7V6gUqBkd0jFEDi2bkJ?e=ja8jJK`

当前 wrapper 需要的权重：

| 任务 | 文件 | 默认放置路径 |
| --- | --- | --- |
| DIOR horizontal detection, Faster R-CNN, ViT-B+RVSA | `dior-rvsa-b-mae-mtp-epoch_12.pth` | `/root/autodl-tmp/MTP-main/weights/dior-rvsa-b-mae-mtp-epoch_12.pth` |
| DIOR-R rotated detection, Oriented R-CNN, ViT-B+RVSA | `diorr-rvsa-b-mae-mtp-epoch_12.pth` | `/root/autodl-tmp/MTP-main/weights/diorr-rvsa-b-mae-mtp-epoch_12.pth` |

配置文件：

```text
/root/autodl-tmp/MTP-main/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/faster_rcnn_rvsa_b_800_mae_mtp_dior.py
/root/autodl-tmp/MTP-main/RS_Tasks_Finetune/Rotated_Detection/mmrotate1.x/configs/mtp/dior-r/oriented_rcnn_rvsa_b_800_mae_mtp_diorr.py
```

对应数据集名称：

- SAMRS: MTP 预训练数据
- SOTA-RBB: MTP 预训练数据的一部分
- DIOR: 水平框检测
- DIOR-R: 旋转框检测
- DOTA-v1.0/DOTA-v2.0: dataset builder 的 DOTA 测试样本和 MTP 论文相关数据

测试数据放置约定：

```text
/root/autodl-tmp/dataset/mtp/DOTAv1.0/val/images/*.png
/root/autodl-tmp/dataset/mtp/DOTAv1.0/labels/val/*.txt
/root/autodl-tmp/dataset/mtp/DOTAv1.0/train/images/*.png
/root/autodl-tmp/dataset/mtp/DOTAv1.0/labels/train/*.txt
```

## DOFA

任务：

- 遥感语义分割
- 支持 HDF5/H5、RGB PNG/JPG/JPEG/BMP、按 band 读取的 TIFF/TIF

wrapper 入口：

- Python: `dofa(...)`
- 后端: `backends/dofa_backend.py`
- Conda env: `dofa`

需要的外部源码：

- DOFA 官方仓库: `https://github.com/zhu-xlab/DOFA`
- DOFA Hugging Face 权重页: `https://huggingface.co/XShadow/DOFA`
- 当前本地 `download_weights.py` 使用的仓库: `https://huggingface.co/earthflow/DOFA`

下载方式：

```bash
cd /root/autodl-tmp/DOFA-master/checkpoints
python download_weights.py
```

当前 wrapper 需要的 backbone 权重：

| 用途 | 文件 | 默认放置路径 |
| --- | --- | --- |
| DOFA ViT-B backbone | `DOFA_ViT_base_e100.pth` | `/root/autodl-tmp/DOFA-master/checkpoints/DOFA_ViT_base_e100.pth` |

当前 wrapper 需要的 segmentation heads：

这些 head 是本地在 `DOFA-master/outputs/dofa_seg_rgb_ms` 下的下游分割 head，不属于 DOFA backbone 通用预训练权重。如果同事要复现当前 wrapper 的输出，需要从团队资产拷贝，或重新训练这些 head。

| dataset/head | 任务/数据集 | 必需文件 |
| --- | --- | --- |
| `m-NeonTree` | tree crown/vegetation segmentation | `best.pth`, `config.json` |
| `m-SA-crop-type` | South Africa crop type segmentation | `best.pth`, `config.json` |
| `m-cashew-plant` | cashew plant segmentation | `best.pth`, `config.json` |
| `m-chesapeake` | Chesapeake land-cover segmentation | `best.pth`, `config.json` |
| `m-nz-cattle` | New Zealand cattle segmentation | `best.pth`, `config.json` |
| `m-pv4ger-seg` | photovoltaic panel segmentation | `best.pth`, `config.json` |

默认放置路径：

```text
/root/autodl-tmp/DOFA-master/outputs/dofa_seg_rgb_ms/<dataset>/best.pth
/root/autodl-tmp/DOFA-master/outputs/dofa_seg_rgb_ms/<dataset>/config.json
```

对应数据集名称：

- `m-NeonTree`
- `m-SA-crop-type`
- `m-cashew-plant`
- `m-chesapeake`
- `m-nz-cattle`
- `m-pv4ger-seg`

数据集放置约定：

```text
/root/autodl-tmp/DOFA-master/datasets/seg_rgb-ms/<dataset>/*.hdf5
/root/autodl-tmp/DOFA-master/datasets/seg_rgb-ms/<dataset>/band_stats.json
/root/autodl-tmp/DOFA-master/datasets/seg_rgb-ms/<dataset>/1.00x_train_partition.json
```

输入格式说明：

- HDF5/H5: band 名必须和 selected head 对上。
- RGB PNG/JPG/JPEG/BMP: 只适合 RGB-only head，例如 `m-NeonTree`, `m-nz-cattle`, `m-pv4ger-seg`。
- TIFF/TIF: 使用 `rasterio` 读取真实通道数；多光谱 head 要求 TIFF band 数和顺序与 head 的 band 顺序一致。

## DehazeFormer

任务：

- 单张 RGB 图像去雾/去云雾复原

wrapper 入口：

- Python: `dehazeformer(...)`
- 后端: `backends/dehazeformer_backend.py`
- Conda env: `dehazeformer`

需要的外部源码：

- DehazeFormer 官方仓库: `https://github.com/IDKiro/DehazeFormer`

公开下载来源：

- Google Drive pretrained models and datasets:
  `https://drive.google.com/drive/folders/1Yy_GH6_bydYPU6_JJzFQwig4LTh86VI4?usp=sharing`
- BaiduPan:
  `https://pan.baidu.com/s/1WVdNccqDMnJ5k5Q__Y2dsg?pwd=gtuw`
- README 中旧版 beta pretrained models:
  `https://drive.google.com/drive/folders/1gnQiI_7Dvy-ZdQUVYXt7pW0EFQkpK39B?usp=sharing`

当前 wrapper 默认权重：

| 用途 | 文件 | 默认放置路径 |
| --- | --- | --- |
| RESIDE6K DehazeFormer-B | `dehazeformer-b.pth` | `/root/autodl-tmp/DehazeFormer/saved_models/reside6k/dehazeformer-b.pth` |

如果调用时传 `checkpoint=...`，会优先使用显式 checkpoint。

对应数据集名称：

- RESIDE-IN
- RESIDE-6K
- RS-Haze / RS-Haze-RGB
- RICE, if using local dataset-test convention

测试数据放置约定：

```text
/root/autodl-tmp/model_wrappers/dataset_test/dehazeformer
/root/autodl-tmp/model_wrappers/dataset_test/RICE
```

## SATtxt

任务：

- 遥感图像零样本分类
- 图像到文本检索
- 文本到图像检索

wrapper 入口：

- Python: `sattxt(...)`
- 后端: `backends/sattxt_backend.py`
- Conda env: `sattxt`

需要的外部源码：

- SATtxt 官方仓库: `https://github.com/ikhado/sattxt`
- SATtxt Hugging Face: `https://huggingface.co/ikhado/sattxt`
- DINOv3: `https://github.com/facebookresearch/dinov3`

公开下载来源：

- DINOv3 ViT-L/16: `https://github.com/facebookresearch/dinov3`
- LLM2Vec text encoder: `https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse`
- SATtxt vision head: `https://huggingface.co/ikhado/sattxt/blob/main/sattxt_vision_head.pt`
- SATtxt text head: `https://huggingface.co/ikhado/sattxt/blob/main/sattxt_text_head.pt`

当前 wrapper 需要的权重：

| 组件 | 文件/目录 | 默认放置路径 |
| --- | --- | --- |
| DINOv3 ViT-L/16 satellite backbone | `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` | `/root/autodl-tmp/sattxt/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` |
| SATtxt vision projection head | `sattxt_vision_head.pt` | `/root/autodl-tmp/sattxt/weights/sattxt_vision_head.pt` |
| SATtxt text projection head | `sattxt_text_head.pt` | `/root/autodl-tmp/sattxt/weights/sattxt_text_head.pt` |
| LLM2Vec local text encoder | `llm2vec_check` | `/root/autodl-tmp/sattxt/weights/llm2vec_check` |

注意：

- SATtxt 后端设置了 `TRANSFORMERS_OFFLINE=1` 和 `HF_HUB_OFFLINE=1`，所以运行前必须把 LLM2Vec 相关文件完整放到本地。
- 后端会确保存在这个 symlink:

```text
/root/autodl-tmp/sattxt/weights/llm2vec_check-unsup-simcse
  -> /root/autodl-tmp/sattxt/weights/llm2vec_check/unsup-simcse
```

对应数据集名称：

- EuroSAT-style categories are used in examples:
  `AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`,
  `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`
- SATtxt is also evaluated as a general satellite image-text retrieval model.

测试数据放置约定：

```text
/root/autodl-tmp/dataset/sattxt
/root/autodl-tmp/model_wrappers/dataset_test/sattxt
/root/autodl-tmp/sattxt/asset/Residential_167.jpg
```

## 构建小测试集

如果已经准备好 `/root/autodl-tmp/dataset` 和 DOFA HDF5 数据，可以重新生成小测试集：

```bash
python /root/autodl-tmp/model_wrappers/scripts/build_dataset_test.py
```

输出：

```text
/root/autodl-tmp/model_wrappers/dataset_test
```

批量测试和生成报告：

```bash
python /root/autodl-tmp/model_wrappers/scripts/run_dataset_test_report.py --max-cases 1
```

输出：

```text
/root/autodl-tmp/model_wrappers/dataset_test_reports
```

## GitHub 上传建议

建议只提交代码和说明文档，不提交以下目录：

```gitignore
outputs/
dataset_test/
dataset_test_reports/
test/
__pycache__/
.ipynb_checkpoints/
*.pyc
*.pth
*.pt
*.ckpt
*.safetensors
*.bin
*.hdf5
*.h5
*.tif
*.tiff
```

如果未来希望让同事一键配置，建议再新增一个 `config.yaml` 或环境变量机制，把当前写死在 backend 里的绝对路径改成可配置路径。
