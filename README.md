[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/nvlabs/SPADE/master/LICENSE.md) ![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

## [IJCAI-2020] Real-World Automatic Makeup via Identity Preservation Makeup Net

</h1>

  <p align="center">
    Zhikun Huang</a> •
    <a href="https://dblp.uni-trier.de/pers/hd/z/Zheng:Zhedong">Zhedong Zheng</a> •
    <a href="https://dblp.uni-trier.de/pers/hd/y/Yan:Chenggang_Clarence">Chenggang Yan </a> •
    <a href="https://dblp.uni-trier.de/pers/hd/x/Xie:Hongtao">Hongtao Xie</a> •
  <p align="center">
    <a href="https://dblp.uni-trier.de/pers/hd/s/Sun:Yaoqi">Yaoqi Sun</a> •
    <a href="https://www.researchgate.net/scientific-contributions/10713410_Jianzhong_Wang">Jianzhong Wang</a> •
    <a href="https://dblp.uni-trier.de/pers/hd/z/Zhang:Jiyong">Jiyong Zhang</a>
  </p>




![](README.assets/results.jpg#pic_center)

<img src="README.assets/controlable.jpg#pic_center"  />

### IPM-Net

> **[Real-World Automatic Makeup via Identity Preservation Makeup Net](https://github.com/huangzhikun1995/IPM-Net/blob/master/Real_World_Automatic_Makeup_via_Identity_Preservation_Makeup_Net.pdf)**<br>
> Zhikun Huang, Zhedong Zheng, Chenggang Yan, Hongtao Xie, Yaoqi Sun, 
Jianzhong Wang, Jiyong Zhang<br>
>
> **Abstract:** *This paper focuses on the real-world automatic makeup problem. Given one non-makeup target image and one reference image, the automatic makeup is to generate one face image, which maintains the original identity with the makeup style in the reference image. In the real-world scenario, face makeup task demands a robust system against the environmental variants. The two main challenges in real-world face makeup could be summarized as follow: first, the background in real-world images is complicated. The previous methods are prone to change the style of background as well; second, the foreground faces are also easy to be affected. For instance, the “heavy” makeup may lose the discriminative information of the original identity. To address these two challenges, we introduce a new makeup model, called Identity Preservation Makeup Net (IPM-Net), which preserves not only the background but the critical patterns of the original identity. Specifically, we disentangle the face images to two different information codes, i.e., identity content code and makeup style code. When inference, we only need to change the makeup style code to generate various makeup images of the target person. In the experiment, we show the proposed method achieves not only better accuracy in both realism (FID) and diversity (LPIPS) in the test set, but also works well on the real-world images collected from the Internet.*

### Dataset Preparation

We train and test our model on the widely-used [Makeup Transfer dataset](http://liusi-group.com/projects/BeautyGAN).

### Training
You can train your own model after downloading the [dataset](http://liusi-group.com/projects/BeautyGAN) and preprocessing the data.
#### Images processing
To train your own model, you must processing the dataset first. You can use MATLAB and the code we provide in the `preprocessing` folder to preprocess the data. `highcontract_texture.m` provides a  differential filter to extract the texture of the face in the picture which is same to our model , and `sobel_texture.m` provides a Sobel operator to extract the texture. 
In addition, if you collect some makeup and non-mekeup image from the internet to train a model or test our model, you have to [parse the faces](https://github.com/switchablenorms/CelebAMask-HQ/tree/master/face_parsing) in the new images before images preprocessing. 

#### Train IPM-Net
1. Setup the `yaml` file. Check out `config/***.yaml`. Change the data_root field to the path of your prepared folder-based dataset.

2. Start training, and you can use `tensorboard` to visualize your loss log.

```
python train.py --config config/***.yaml
```

### Testing
To test the model, you will need to have a CUDA capable GPU, PyTorch, cuda/cuDNN  drivers, tensorboardX and pyyaml installed. 
#### Download the trained model
We provide our trained model. You can download it from [Google Drive](https://drive.google.com/drive/folders/1dTmg0SWGkqu2NnzWgcI4BlxIy476HuL1?usp=sharing) (or [Baidu Disk](https://pan.baidu.com/s/1_S1F9D7YdamlMvFg7wTRDg) password: yuxv). You can download and move it to the `outputs` folder.
#### Testing the model
You may test our trained model use `test.py` and the few images in the `dataset` folder. Or you can collect some makeup images and non-makeup images from the Internet to test our model. 

```
python test.py 
```
If you want to test your trained model, remember to change the parameters `name` and `num` in the `test.py`. 


### Image generation evaluation
You may use generate lots of images and then do the evaluation using [FID](https://github.com/layumi/TTUR) and [LPIPS](https://github.com/layumi/PerceptualSimilarity) . 

### Related Work
We compared with [BeautyGAN](https://github.com/Honlan/BeautyGAN) , which is also GAN-based and open sourced the trained model. We forked the code and made some changes for evaluatation, thank the authors for their great work. We would also like to thank to the great projects in [CycleGAN(https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [MUNIT](https://github.com/NVlabs/MUNIT) and [DG-Net](https://github.com/NVlabs/DG-Net).

### License
Copyright (C) 2020 Hangzhou Dianzi University. All rights reserved. Licensed under the CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International). The code is released for academic research use only.

