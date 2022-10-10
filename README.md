# FreGAN: Exploiting Frequency Components for Training GANs under Limited Data

![image](./assets/teaser.png)

> **Exploiting Frequency Components for Training GANs under Limited Data**
> Mengping Yang, Zhe Wang, Ziqiu Chi, Yanbing Zhang


[[Paper]()]

## Abstract 
Training GANs under limited data often leads to discriminator overfitting and memorization issues, causing divergent training. Existing approaches mitigate the overfitting by employing data augmentations, model regularization, or attention mechanisms. However, they ignore the frequency bias of GANs and take poor consideration towards frequency information, especially high-frequency signals that contain rich details. To fully utilize the frequency information of limited data, this paper proposes FreGAN, which raises the model's frequency awareness and draws more attention to producing high-frequency signals, facilitating high-quality generation. In addition to exploiting both real and generated images' frequency information, we also involve the frequency signals of real images as a self-supervised constraint, which alleviates the GAN disequilibrium and encourages the generator to synthesize adequate rather than arbitrary frequency signals. Extensive results demonstrate the superiority and effectiveness of our FreGAN in ameliorating generation quality in the low-data regime (especially when training data is less than 100). Besides, FreGAN can be seamlessly applied to existing regularization and attention mechanism models to further boost the performance.

## Qualitative results
Here we provide the qualitative comparison results of our FreGAN and baseline [[FastGAN](https://github.com/odegeasslbc/FastGAN-pytorch)].
The images from left to right are generated images, low-frequency, and high-frequency components, respectively.
Our FreGAN improves the overall quality of generated images and raises the model's frequency awareness, encouraging the generator to produce precise high-frequency signals with fine details.
![image](./assets/Visall.png)

## Usage 
Use ./run.sh to run the code.

The results and models will be automatically saved in /train_resutls folder.

The results of FID, KID, Precision, Recall, Density, and Coverage will be automatically in FID.txt in the results folder, and IS results will be saved in IS.txt in the same folder.

FYI, you need to install the clean-fid (use pip install clean-fid) and prdc (use pip install prdc) for calculating our adopted clean-fid and precision/recall metrics.

## Contact
Fell free to contact me at kobeshegu@gmail.com if you have any questions or advices, thanks!

## BibTeX
<!-- ```bibtex
@article{yang2022FreGAN,
title   = {FreGAN: Exploiting Frequency Components for Training GANs under Limited Data},
author  = {Yang, Mengping and Wang, Zhe and Chi, Ziqiu and Zhang, Yanbing},
article = {},
year    = {2022} 
}
``` -->

