<p align="center">
    <img src="assets/logo.png" width="300">
</p>

## MambaIR: A Simple Baseline for Image Restoration with State-Space Model

[[Paper](https://arxiv.org/abs/2402.15648)] [Zhihu(知乎)]


[Hang Guo](https://github.com/csguoh)\*, Jinmin Li\*, [Tao Dai](https://cstaodai.com/), Zhihao Ouyang, Xudong Ren, and [Shu-Tao Xia](https://scholar.google.com/citations?hl=zh-CN&user=koAXTXgAAAAJ)

(\*) equal contribution

> **Abstract:**  Recent years have witnessed great progress in image restoration thanks to the advancements in modern deep neural networks \textit{e.g.} Convolutional Neural Network and Transformer. However, existing restoration backbones are usually limited due to the inherent local reductive bias or quadratic computational complexity. Recently, Selective Structured State Space Model \textit{e.g.}, Mamba, have shown great potential for long-range dependencies modeling with linear complexity, but it is still under-explored in low-level computer vision. In this work, we introduce a simple but strong benchmark model, named MambaIR, for image restoration. In detail, we propose the Residual State Space Block as the core component, which employs convolution and channel attention to enhance capabilities of the vanilla Mamba. In this way, our MambaIR takes advantages of local patch recurrence prior as well as channel interaction to produce restoration-specific feature representation. Extensive experiments demonstrate the superiority of our method, for example, MambaIR outperforms Transformer-based baseline SwinIR by up to 0.36dB, using similar computational cost but with global receptive field. 


<p align="center">
    <img src="assets/pipeline.png" style="border-radius: 15px">
</p>

⭐If this work is helpful for you, please help star this repo. Thanks!🤗



## 📑 Contents

- [Visual Results](#visual_results)
- [News](#news)
- [TODO](#todo)
- [Model Summary](#model_summary)
- [Results](#results)
- [Installation](#installation)
- [Training](#training)
- [Testing](#testing)
- [Citation](#cite)


## <a name="visual_results"></a>:eyes:Visual Results On Classic Image SR

<p align="center">
  <img width="800" src="assets/visual.png">
</p>


## <a name="news"></a> 🆕 News

- **2024-2-27:** This repo is released.
- **2024-2-27:** arXiv paper available.




## <a name="todo"></a> ☑️ TODO

- [x] Build the repo
- [x] arXiv version
- [ ] Release code
- [ ] Pretrained weights
- [ ] Real-world SR
- [ ] JPEG Compression Artifact Redection
- [ ] More Tasks
 

## <a name="model_summary"></a> :page_with_curl: Model Summary

| Model          | Task                 | Test_dataset | PSNR | SSIM | ckpt_link | log_file |
| -------------- | -------------------- | ------------ | ---- | ---- | --------- | -------- |
| MambaIR_SR2    | Classic SR x2        | Urban100     | 34.15 |   0.9446   | link      | link     |
| MambaIR_SR3    | Classic SR x3        | Urban100     | 29.93 |  0.8841    | link      | link     |
| MambaIR_SR4    | Classic SR x4        | Urban100     | 27.68 |  0.8287    | link      | link     |
| MambaIR_light2 | Lightweight SR x2    | Urban100     | 32.86 |  0.9343   | link      | link     |
| MambaIR_light3 | Lightweight SR x3    | Urban100     | 28.73 |   0.8635  | link      | link     |
| MambaIR_light4 | Lightweight SR x4    | Urban100     | 26.53| 0.7983     | link      | link     |
| MambaIR_realDN | Real image Denoising | SIDD         | 39.89|   0.960   | link      | link     |


## <a name="results"></a> 🥇 Results

We achieve state-of-the-art performance on various image restoration tasks. Detailed results can be found in the paper.


<details>
<summary>Evaluation on Classic SR (click to expand)</summary>

<p align="center">
  <img width="500" src="assets/classicSR.png">
</p>
</details>



<details>
<summary>Evaluation on Lightweight SR (click to expand)</summary>

<p align="center">
  <img width="500" src="assets/lightSR.png">
</p>
</details>


<details>
<summary>Evaluation on Real Image Denoising (click to expand)</summary>

<p align="center">
  <img width="500" src="assets/real-dn.png">
</p>

</details>


<details>
<summary>Evaluation on Effective Receptive Filed (click to expand)</summary>

<p align="center">
  <img width="600" src="assets/erf.png">
</p>

</details>


## <a name="installation"></a> :wrench: Installation

This codebase was tested with the following environment configurations. It may work with other versions.

- Ubuntu 20.04
- CUDA 11.7
- Python 3.9
- PyTorch 1.13.1 + cu117

To use the selective scan with efficient hard-ware design, the `mamba_ssm` library is advised to install with the folllowing command.

```
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1
```


## <a name="training"></a>  :hourglass: Training

### Train on SR

1. Please download the corresponding training datasets and put them in the folder datasets/DF2K. Download the testing datasets and put them in the folder datasets/SR.

2. Follow the instructions below to begin training our model.

3.

```
# Claissc SR task, cropped input=64×64, 8 GPUs, batch size=4 per GPU
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 basicsr/train.py -opt options/train/train_MambaIR_SR_x2.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 basicsr/train.py -opt options/train/train_MambaIR_SR_x3.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 basicsr/train.py -opt options/train/train_MambaIR_SR_x4.yml --launcher pytorch

# Lightweight SR task, cropped input=64×64, 8 GPUs, batch size=8 per GPU
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 basicsr/train.py -opt options/train/train_MambaIR_lightSR_x2.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 basicsr/train.py -opt options/train/train_MambaIR_lightSR_x3.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 basicsr/train.py -opt options/train/train_MambaIR_lightSR_x4.yml --launcher pytorch
```

3. Run the script then you can find the generated experimental logs in the folder experiments.

### Train on Real Denoising

1. Please download the corresponding training datasets and put them in the folder datasets/SIDD. Note that we provide both training and validating files, which are already processed.
2. Go to folder 'realDenoising'. Follow the instructions below to train our ART model.

``` 
# go to the folder
cd realDenoising
# set the new environment (BasicSRv1.2.0), which is the same with Restormer for training.
python setup.py develop --no_cuda_extgf
# train for RealDN task, 8 GPUs
python -m torch.distributed.launch --nproc_per_node=8 --master_port=2414 basicsr/train.py -opt options/train_MambaIR_RealDN.yml --launcher pytorch
Run the script then you can find the generated experimental logs in the folder realDenoising/experiments.
```

Remember to go back to the original environment if you finish all the training or testing about real image denoising task. This is a friendly hint in order to prevent confusion in the training environment.
```
# Tips here. Go back to the original environment (BasicSRv1.3.5) after finishing all the training or testing about real image denoising. 
cd ..
python setup.py develop
```


## <a name="testing"></a> :smile: Testing

### Test on SR


### Test on Real Denoising


## <a name="cite"></a> 🥰 Citation

Please cite us if our work is useful for your research.

```
@article{guo202mambair,
  title={MambaIR: A Simple Baseline for Image Restoration with State-Space Model},
  author={Hang Guo, Jinmin Li, Tao Dai, Zhihao Ouyang, Xudong Ren, and Shu-Tao Xia},
  journal={arXiv preprint arXiv:2402.15648},
  year={2024}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

This code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR), [ART](https://github.com/gladzhang/ART) ,and [VMamba](https://github.com/MzeroMiko/VMamba). Thanks for their awesome work.

## Contact

If you have any questions, feel free to approach me at cshguo@gmail.com

