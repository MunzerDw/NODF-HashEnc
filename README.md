# Estimating Neural Orientation Distribution Fields for High Resolution Diffusion MRI Scans

This code repository is an implementation of the paper arXiv:2307.08138v2 for 2 different kinds of implicit networks.

<p align="center"><img width="750" src="./docs/method.png"></p>

## Abstract

The Orientation Distribution Function (ODF) characterizes key brain microstructural
properties and plays an important role in understanding brain structural connectivity.
Recent works introduced Implicit Neural Representation (INR) based approaches
to form a spatially aware continuous estimate of the ODF field and demonstrated
promising results in key tasks of interest when compared to conventional discrete
approaches. However, traditional INR methods face difficulties when scaling to large-
scale images, such as modern ultra-high-resolution MRI scans, posing challenges in
learning fine structures as well as inefficiencies in training and inference speed. In this
work, we propose HashEnc, a grid-hash-encoding-based estimation of the ODF field
and demonstrate its effectiveness in retaining structural and textural features. We show
that HashEnc achieves a 10 % enhancement in image quality while requiring 3x less
computational resources than current methods.

From: [Technical University of Munich](https://www.tum.de/en/) and [Harvard Medical School](https://hms.harvard.edu/)

## Setup 

**Environment requirements**
- CUDA 11.X
- Python 3.8

Instal the requirements using conda:

```shell
    conda env create --name nodf --file=environment.yml
    conda activate nodf
```

## Dataset

You require the following files:
- signal.nii.gz: (X, Y, Z) raw MRI signal
- bval.txt: b-values written in one line separated by a space
- bvec.txt: b-vectors each written vertically
- mask.nii.gz: (X, Y, Z) [0,1] mask to select the whole brain region/ region of interest

Put these files under the <code>data</code> folder.

## Usage

If you want to use pre-trained models, please download them from the section below.

### Predicting

```shell
    python predict.py --ckpt_path <path to pytorch lightning .ckpt file>
```

The predictions can be found under `output/<experiment>/predictions`

### Evaluation

To get GFA and DTI images as well as calculate the FSIM and ODF L2-Norm scores:

```shell
    python evaluate.py --device cpu
```

This will directly use the ```predictions.pt``` file under `output/<experiment>/predictions`. The evaluations output can be found under `output/<experiment>/evaluations`.

### Training

To train HashEnc:

```shell
    python train.py
```

To train a large SIREN network:

```shell
    python train.py --experiment_name baseline --depth 10 --r 1024 --nu 1.5 --lambda_c 6.36e-06 --learning_rate 1e-6 --use_baseline
```

### Pretrained Models

| Model                                                                                            |Comment   
|--------------------------------------------------------------------------------------------------|---------|
|[HashEnc](https://drive.google.com/file/d/1MpWNUOTNujwesz5ewNRj7fAe4p2UFsYA/view?usp=drive_link)|HashEnc trained as above with default network configuration|
|[SIREN](https://drive.google.com/file/d/1HypX_L33UgpHp_Eo4WgKS2OjzajW6fJF/view?usp=sharing)|Large SIREN network trained as above|

## Visualization

To visualize the deconvolved ODFs:

```shell
python visualize.py
```

The deconvolved ODFs can be found under `output/<experiment>/visualization`

## Logging

You can use tensorboard to check losses and accuracies by visiting <b>localhost:6006</b> after running:
```shell
tensorboard --logdir output
```

## Citation

If you found our work helpful, please kindly cite our paper:

```shell
TODO
```