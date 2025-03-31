typhoon photo prediction_using Diffusion  
dataset V2 git( 有两个dataset， only use WP (west pacific) ,师兄让我选的 ，数据很大50G ):  
https://github.com/kitamoto-lab/digital-typhoon/blob/main/README.md  

V2 refer paper:https://arxiv.org/abs/2411.16421  
code devided into 2 parts：we only run CDDPM/CNN model:  
paper refered:https://click.endnote.com/viewer?doi=10.48550%2Farxiv.2409.07961&token=WzQ0NzE3MDYsIjEwLjQ4NTUwL2FyeGl2LjI0MDkuMDc5NjEiXQ.2bxOluD3P7Xrs-Za7tT2ZR9RSFY  
github for the paper:https://github.com/TammyLing/Typhoon-forecasting?utm_source=catalyzex.com  

NVIDIA(donot use but we can refer the writing stracture)-Generative Correction Diffusion Model (CorrDiff) for Km-scale Atmospheric Downscaling):https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/examples/generative/corrdiff/readme.html#:~:text=To%20improve%20weather%20hazard%20predictions%20without%20expensive%20simulations%2C,using%20high-resolution%20weather%20data%20and%20coarser%20ERA5%20reanalysis.  

my Google drive:  
code for CDDPM and dataset(dataset is included in the CDDPM,named:WP):https://drive.google.com/drive/folders/11GDacHP5BNizN6NCXyCFe6yYizED8Tau?usp=drive_link  
code for CNN：https://drive.google.com/drive/folders/11CLKDupchDoeVjavgMSBa2Tq1tN3AIv_?usp=drive_link

1.data discription:  
It is a track data, location of which is Western North Pacific basin, and the Japan Meteorological Agency (JMA) is designated as the regional center and collected from 1842. With ’annotation’ for TC (tropical cyclones) including: location, intensity, and wind circles, based on the interpretation of meteorological experts following the established procedure (e.g.
Dvorak Technique), utilized Lambert azimuthalequal-area projection referring to the best track data which means it recorded from the start to the end of life of TC to review and evaluate it so as to get the "best estimation". 
![image](https://github.com/user-attachments/assets/93491308-8bae-474c-a129-dcf4533f87a4)


