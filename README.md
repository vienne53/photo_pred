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

我们为所有代卫星开发
了⾃⼰的解析器，并参照最佳路径创建了以台⻛为中⼼的图像的⼯作流程，如图 1 所⽰。
通过这种⼯作流程，我们整合了元数据和图像，创建了“数字台⻛”数据集。元数据包含每⼩
时的最佳路径数据以及有关⽂件名和每张图像质量的附加信息。最佳路径数据的格式与⽇本⽓
象厅提供的原始最佳路径数据格式⼀致。另⼀⽅⾯，图像以 HDF5 格式呈现台⻛中⼼周围⼆维
亮度温度阵列。当前数据集仅包含红外通道（IR1）（波⻓约为 11 微⽶），但不包含数字台⻛⽹站上
提供的其他任何通道IR1 是唯⼀⼀条最⻓且数据质量问题最少的通道，这也是我们在数据集的第⼀个版本中仅
包含 IR1 通道的原因。未来纳⼊多光谱数据可能会提供诸如多光谱分类和回归等更多任务。
空间分辨率 约每像素 5 千⽶的空间分辨率反映了第⼀代“向⽇葵”1 号⾄ 5 号卫星的红外 1 通道的
空间分辨率。第⼆代卫星的空间分辨率已提⾼到 4 千⽶，第三代则提⾼到了 2 千⽶。尽管技术有了
这些进步，但我们仍选择 5 千⽶的分辨率，因为这是创建⻓期同质数据集的最佳选择。未来的⼀个
有趣任务是将机器学习模型从⻓期低分辨率数据集转移到短期⾼分辨率数据集，以便利⽤最新技术
实现更准确的预测。
时间分辨率 ⼀⼩时的时间分辨率反映了第⼀代中部分卫星（从“向⽇葵 3 号”开始）的时间分辨率。
从“向⽇葵 1 号”到“向⽇葵 2 号”，时间分辨率超过⼀⼩时，通常为每三⼩时⼀次。因此，1987
年以前的数据作为⼀⼩时的数据集存在⼤量缺失值。第⼆代卫星的时间分辨率已提⾼到 30 分钟，第
三代卫星则提⾼到 10 分钟。尽管技术有了这些进步，我们仍选择⼀⼩时作为时间间隔，因为它能代
表许多⽓象观测的典型间隔。
空间覆盖范围 当前的数据集仅涵盖北半球（NH）的西北太平洋盆地，但“数字台⻛”⽹站提供了南
半球（SH）澳⼤利亚盆地的相同类型的图像，其最佳路径数据来⾃澳⼤利亚⽓象局。这⾥有⼀个有
趣的问题是，如何将北半球训练的模型转移到南半球。从⽓象学⻆度来看，不同盆地的热带⽓旋被认
为是相同的⽓象现象，因此理论上可以创建类似的数据库，机器学习的结果也是可迁移的。然⽽，我
们还需要考虑许多可能影响实际结果的细节，例如最佳路径数据的质量不同，以及不同卫星的传感器
特性和校准⽅法不同。我们未来版本的数据集和基准可能会解决这些问题。(待会看看
作者的思路
1. 分析任务：利用当前和过去的数据来估算当前值，例如估算台风强度，可细分为分类任务或回归任务，取决于目标变量的类型。
2. 预测任务：基于当前和过去的数据对未来进行预测，包括短期预测的“临近预报”，但受限于大气混沌性质。
3. 术语区分：在机器学习领域，“预测”一词与气象学中的用法不同，为避免混淆，论文统一使用“预测”。
4. 重新分析任务：在所有可获取的数据基础上得出最佳估计，对生成跨越长时间段的统一数据集尤其重要，机器学习有助于评估标注数据的质量。
预计要推测的结果：
1. 推论目标：任务旨在推断台风的强度、规模和位置，使用分类等级对热带气旋进行强度和类型分类，并通过中心气压和最大持续风速进行强度数值测量。
2. 强度与规模：热带气旋的强度通过气压或风速进行回归任务，同时考虑强风圈半径作为规模回归任务的目标变量。
3. 位置推断：预测台风位置的回归任务以人类专家估算的气旋中心位置（纬度和经度坐标）作为目标变量，精度为0.1度。
4. 形成过程：任务包括推断热带气旋的生成，这在热带地区众多活跃云团中识别潜在的热带气旋形成过程。
5. 过渡推断：任务还涉及热带气旋向温带气旋转变的推断，这一过程通常发生在中纬度地区，机器学习模型用于数据驱动建模这一连续转变过程。
他做了：
 
1. 分类与回归任务：研究中包含分类任务和回归任务，分类任务将图像分为热带气旋等级，回归任务则预测气压或风速值。
2. 模型与数据处理比较：比较了VGG、ResNet和视觉变换器三种架构，并分析了全分辨率、调整大小和裁剪图像三种训练方式对模型性能的影响。
3. 实验设置与评估：使用TorchVision中的ResNet18和ResNet50模型进行训练，采用均方根误差（RMSE）和标准差评估预测准确性。
4. 结果分析：裁剪图像的RMSE较低，表明此方法更能保留台风中心特征；全图训练效果不佳；对弱台风回归效果好，强台风效果差。
5. 强度预测方法：采用卷积长短期记忆网络（ConvLSTM）预测台风强度，通过分析预测图像中的气压进行强度预测。
强度再预测！
划分了三个组， 目的是识别数据集中因技术进步等因素引起的偏差和不一致性
实践过程：
1. 图像预处理与模型训练：输入图像被调整为224×224大小，每个数据集随机抽取208个序列，分为80/20的训练集和测试集。每个数据集上训练一个ResNet18模型，共训练101个周期，批量大小为16，学习率为10^-4。
2. 模型测试与性能评估：三个模型在三个数据集上的表现相当，未显示出数据集偏差对模型质量有明显影响。详细实验描述见附录。
总结：研究引入了“数字台风”数据集，为机器学习模型提供了一个独特的基准测试平台，有助于推动热带气旋研究，增进科学认知，并对灾害减少和气候变化等社会问题提供帮助。

他的一些修改建议：
1. 额外基准测试结果：在第5节的基础上，提供了更多分类任务的基准测试结果，包括VGG、ResNet18和视觉变换器（ViT）三种架构在不同输入图像上的性能比较。
2. 强度分析：VGG在分类准确率上略优于ResNet18，而ViT表现最差。结果表明数据集规模可能不足以充分发挥模型性能，建议探索更深的网络架构和增加训练轮数。
3. 分类准确率问题：分类准确率最高为70%，可能低估了实际性能。分类任务的等级定义存在连续性和模糊性，建议引入更符合实际的类别转换损失函数。
4. 回归任务的优势：回归任务（特别是对中心气压的回归）定义更明确，因此在正文中主要介绍回归任务的结果。
5. 数据集规模的影响：通过使用每一代卫星数据的全部序列作为训练集，发现更大规模的数据集可能抵消了代际偏差的负面影响，基于最大数据集训练的模型表现最佳。



数据集描述在附录B
他们写了一个库？
pyphoon2 提供了⼀个过滤选项，可根据遮挡像素的百分⽐从机器学习任务
中排除某些图像，⽤⼾有责任根据其机器学习任务的需要适当使⽤过滤功能。
![image](https://translate.google.com/saved?sl=auto&tl=zh-CN&op=translate&hl=zh-cn)
