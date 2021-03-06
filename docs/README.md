# Sky Optimization: Semantically aware image processing of skies in low-light photography

NTIRE CVPRW 2020

Authors: [Orly Liba](https://sites.google.com/corp/view/orly-liba/),
[Longqi Cai](https://www.linkedin.com/in/longqicai/en-us),
[Yun-Ta Tsai](https://ai.google/research/people/105312/),
[Elad Eban](https://research.google/people/EladEban/),
[Yair Movshovitz-Attias](https://research.google/people/YairMovshovitzAttias/),
[Yael Pritch](https://scholar.google.com/citations?user=2jXxOYQAAAAJ),
[Huizhong Chen](https://www.linkedin.com/in/huizhong-chen-00776432),
[Jonathan T. Barron](https://jonbarron.info/)

![figure1](sky-optimization-system-examples.png)

## Abstract

The sky is a major component of the appearance of a photograph, and its color and tone can strongly influence the mood of a picture. In nighttime photography, the sky can also suffer from noise and color artifacts. For this reason, there is a strong desire to process the sky in isolation from the rest of the scene to achieve an optimal look. 
In this work, we propose an automated method, which can run as a part of a camera pipeline, for creating accurate sky alpha-masks and using them to improve the appearance of the sky.
Our method performs end-to-end sky optimization in less than half a second per image on a mobile device.
We introduce a method for creating an accurate sky-mask dataset that is based on partially annotated images that are inpainted and refined by our modified weighted guided filter. We use this dataset to train a neural network for semantic sky segmentation.
Due to the compute and power constraints of mobile devices, sky segmentation is performed at a low image resolution. Our modified weighted guided filter is used for edge-aware upsampling to resize the alpha-mask to a higher resolution.
With this detailed mask we automatically apply post-processing steps to the sky in isolation, such as automatic spatially varying white-balance, brightness adjustments, contrast enhancement, and noise reduction.

#### Downloads

[Paper on CVF website](http://openaccess.thecvf.com/content_CVPRW_2020/html/w31/Liba_Sky_Optimization_Semantically_Aware_Image_Processing_of_Skies_in_Low-Light_CVPRW_2020_paper.html)

[Paper on arXiv](https://arxiv.org/abs/2006.10172)

[Full resolution images from the paper](https://github.com/google/sky-optimization/tree/master/full-resolution-images)

[Refined and non-refined sky-mask datsets](https://github.com/google/sky-optimization/tree/master/sky-mask-datasets), corresponding to the Results section of the paper.
