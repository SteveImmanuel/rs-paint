# Tackling Few-Shot Segmentation in Remote Sensing via Inpainting Diffusion Model

**ICLR Machine Learning for Remote Sensing Workshop, 2025 (Oral)**


This repo contains the official code for training and generation for the paper "Tackling Few-Shot Segmentation in Remote Sensing via Inpainting Diffusion Model".


<!-- [![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2312.02145) -->
<!-- [![Hugging Face Demo](https://img.shields.io/badge/🤗%20Hugging%20Face%20(LCM)-Space-yellow)](https://huggingface.co/spaces/prs-eth/marigold-lcm) -->
[![Website](figure/badge-website.svg)](https://steveimmanuel.github.io/rs-paint)
[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

[Steve Andreas Immanuel](https://steveimm.id), [Woojin Cho](https://woojin-cho.github.io/), [Junhyuk Heo](https://rokmc1250.github.io/), Darongsae Kwon.

We introduce a image-conditioned diffusion-based approach for to create diverse set of novel-classes samples for semantic segmentation in few-shot settings in remote sensing domain. By ensuring semantic consistency using cosine similarity between the generated samples and the conditioning image, and using the Segment Anything Model (SAM) to obtain the precise segmentation, our method can train off-the-shelf segmentation models with high-quality synthetic data, significantly improving performance in low-data scenarios.

![Teaser](figure/teaser.jpg)


## 📢 News
2024-05-28: Training code is released.<br>
2024-03-23: Added [LCM v1.0](https://huggingface.co/prs-eth/marigold-lcm-v1-0) for faster inference - try it out at <a href="https://huggingface.co/spaces/prs-eth/marigold-lcm"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face%20(LCM)-Space-yellow" height="16"></a><br>
2024-03-04: Accepted to CVPR 2024. <br>
2023-12-22: Contributed to Diffusers [community pipeline](https://github.com/huggingface/diffusers/tree/main/examples/community#marigold-depth-estimation). <br>
2023-12-19: Updated [license](LICENSE.txt) to Apache License, Version 2.0.<br>
2023-12-08: Added
<a href="https://huggingface.co/spaces/toshas/marigold"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow" height="16"></a> - try it out with your images for free!<br>
2023-12-05: Added <a href="https://colab.research.google.com/drive/12G8reD13DdpMie5ZQlaFNo2WCGeNUH-u?usp=sharing"><img src="doc/badges/badge-colab.svg" height="16"></a> - dive deeper into our inference pipeline!<br>
2023-12-04: Added <a href="https://arxiv.org/abs/2312.02145"><img src="https://img.shields.io/badge/arXiv-PDF-b31b1b" height="16"></a>
paper and inference code (this repository).

## News
- *2023-11-28* The recent work Asymmetric VQGAN improves the preservation of details in non-masked regions. For comprehensive details, please refer to the associated [paper](https://arxiv.org/abs/2306.04632), [github]( https://github.com/buxiangzhiren/Asymmetric_VQGAN).
- *2023-05-13* Release code for quantitative results.
- *2023-03-03* Release test benchmark.
- *2023-02-23* Non-official 3rd party apps support by [ModelScope](https://www.modelscope.cn/models/damo/cv_stable-diffusion_paint-by-example/summary) (the largest Model Community in Chinese).
- *2022-12-07* Release a [Gradio](https://gradio.app/) demo on [Hugging Face](https://huggingface.co/spaces/Fantasy-Studio/Paint-by-Example) Spaces.
- *2022-11-29* Upload code.


## Requirements
A suitable [conda](https://conda.io/) environment named `Paint-by-Example` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate Paint-by-Example
```

## Pretrained Model
We provide the checkpoint ([Google Drive](https://drive.google.com/file/d/15QzaTWsvZonJcXsNv-ilMRCYaQLhzR_i/view?usp=share_link) | [Hugging Face](https://huggingface.co/Fantasy-Studio/Paint-by-Example/resolve/main/model.ckpt)) that is trained on [Open-Images](https://storage.googleapis.com/openimages/web/index.html) for 40 epochs. By default, we assume that the pretrained model is downloaded and saved to the directory `checkpoints`.

## Testing

To sample from our model, you can use `scripts/inference.py`. For example, 
```
python scripts/inference.py \
--plms --outdir results \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_path examples/image/example_1.png \
--mask_path examples/mask/example_1.png \
--reference_path examples/reference/example_1.jpg \
--seed 321 \
--scale 5
```
or simply run:
```
sh test.sh
```
Visualization of inputs and output:

![](figure/result_1.png)
![](figure/result_2.png)
![](figure/result_3.png)

## Training

### Data preparing
- Download separate packed files of Open-Images dataset from [CVDF's site](https://github.com/cvdfoundation/open-images-dataset#download-images-with-bounding-boxes-annotations) and unzip them to the directory `dataset/open-images/images`.
- Download bbox annotations of Open-Images dataset from [Open-Images official site](https://storage.googleapis.com/openimages/web/download_v7.html#download-manually) and save them to the directory `dataset/open-images/annotations`.
- Generate bbox annotations of each image in txt format.
    ```
    python scripts/read_bbox.py
    ```

The data structure is like this:
```
dataset
├── open-images
│  ├── annotations
│  │  ├── class-descriptions-boxable.csv
│  │  ├── oidv6-train-annotations-bbox.csv
│  │  ├── test-annotations-bbox.csv
│  │  ├── validation-annotations-bbox.csv
│  ├── images
│  │  ├── train_0
│  │  │  ├── xxx.jpg
│  │  │  ├── ...
│  │  ├── train_1
│  │  ├── ...
│  │  ├── validation
│  │  ├── test
│  ├── bbox
│  │  ├── train_0
│  │  │  ├── xxx.txt
│  │  │  ├── ...
│  │  ├── train_1
│  │  ├── ...
│  │  ├── validation
│  │  ├── test
```

### Download the pretrained model of Stable Diffusion
We utilize the pretrained Stable Diffusion v1-4 as initialization, please download the pretrained models from [Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) and save the model to directory `pretrained_models`. Then run the following script to add zero-initialized weights for 5 additional input channels of the UNet (4 for the encoded masked-image and 1 for the mask itself).
```
python scripts/modify_checkpoints.py
```

### Training Paint by Example
To train a new model on Open-Images, you can use `main.py`. For example,
```
python -u main.py \
--logdir models/Paint-by-Example \
--pretrained_model pretrained_models/sd-v1-4-modified-9channel.ckpt \
--base configs/v1.yaml \
--scale_lr False
```
or simply run:
```
sh train.sh
```

## Citing Paint by Example

```
@article{yang2022paint,
  title={Paint by Example: Exemplar-based Image Editing with Diffusion Models},
  author={Binxin Yang and Shuyang Gu and Bo Zhang and Ting Zhang and Xuejin Chen and Xiaoyan Sun and Dong Chen and Fang Wen},
  journal={arXiv preprint arXiv:2211.13227},
  year={2022}
}
```

## Acknowledgements

This code is mainly based on [Paint by Example](https://github.com/Fantasy-Studio/Paint-by-Example). We thank the authors for their great work.

## Maintenance

Please open a GitHub issue for any help. If you have any questions regarding the technical details, feel free to contact us.
