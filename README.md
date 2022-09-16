| Method | PSNR    | SSIM    |
| :---:   | :---: | :---: |
| VINet   | 28.47 | 0.9222 |
| DFGVI | 30.28   | 0.925   |
| CPN | 31.59    | 0.933    |
| OPN   | 32.40| 0.944|
|3DGC  | 31.69   | 0.940   |
| STTN | 32.83   | 0.943    |
| TSAM   | 31.50|0.934 |
| OURS | 32.98   | 0.94   |


|  | Frame 50    | Frame 100    | Frame 150|
| :---:   | :---: | :---: | :---: |
| Video 1   | 0.9377 | 0.9358 |  0.9383 |
| Video 2 | 0.9365   | 0.9389   | 0.9374 |


Frame 50 Frame 100 Frame150
Video 1 0.9377 0.9358 0.9383
Video 2 0.9365 0.9389 0.9374
Table 3: Here the frame difference of 50, 100 and 150 frames
of input video with previous frame of inpainted video is
shown.
frame difference-SSIM
frame 50 Frame 100 Frame150
Video 1 0.4504 0.4445 0.4603
Video 2 0.4386 0.4465 0.4532
FUTURE WORK
We will address the problem of handing complex textures
by adjusting the network structure and also try to introduce
a deeper network for this model to capture the training image
distribution better. We will also improve it for other manip-
ulation tasks.
References
SupplementaryMaterials. 2022. :[Online] Available.
https://github.com/Rishabh-Shuklaa/suplimentry. Accessed:
16/09/2022.
Shaham, T. R., Dekel, T., Michaeli, T.Singan: Learning a
generative model from a single natural image. In Proceed-
ings of the IEEE International Conference on Computer
Vision (pp. 4570-4580) (2019).



# REFERENCES
1:Kim, D., Woo, S., Lee, J.Y., Kweon, I.S.: Deep video inpainting. In: CVPR. pp. 5792–5801 (2019).<br /><br />
2:Lee, S., Oh, S.W., Won, D., Kim, S.J.: Copy-and-paste networks for deep video inpainting. In: ICCV. pp. 4413–4421 (2019).<br /><br />
3:Xu, R., Li, X., Zhou, B., Loy, C.C.: Deep flow-guided video inpainting. In: CVPR. pp. 3723–3732 (2019).<br /><br />
4:Oh, S.W., Lee, S., Lee, J.Y., Kim, S.J.: Onion-peel networks for deep video completion. In: ICCV. pp. 4403–4412 (2019).<br /><br />
5:Chang, Y.L., Liu, Z.Y., Lee, K.Y., Hsu, W.: Free-form video inpainting with 3D gated convolution and temporal PatchGAN. In: ICCV. pp. 9066–9075 (2019).<br /><br />
6:Zeng, Y., Fu, J., Chao, H.: Learning joint spatial-temporal transformations for video inpainting. In: ECCV. pp. 528–543 (2020).<br /><br />
7:Zou, X., Yang, L., Liu, D., Lee, Y.J.: Progressive temporal feature alignment network for video inpainting. In: CVPR (2021).<br /><br />
