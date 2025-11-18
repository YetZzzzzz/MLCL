# MLCL: Multi-Level Contrastive Learning for Multimodal Sentiment Analysis
The implementation of the paper "Multi-Level Contrastive Learning for Multimodal Sentiment Analysis", which has been accepted by IEEE Transactions on Multimedia ([TMM](https://ieeexplore.ieee.org/document/11175557)). 

### The Framework of MLCL:
![image](https://github.com/YetZzzzzz/MLCL/blob/main/framework.png)
Figure 1: Overview of the Multi-Level Contrastive Learning (MLCL) framework. The MLCL module is structured around three contrastive learning sub-modules: Uni-Modal Contrastive Learning (UMCL), Bi-Modal Contrastive Learning (BMCL), and Tri-Modal Contrastive Learning (TMCL). The UMCL aims to enhance the uni-modal representations. The BMCL focuses on synthesizing and aligning bi-modal representations, while the TMCL seeks to achieve more alignment between all three modalities. Here `FF' denotes the Feed-forward layer.

### Datasets:
**Please move the following datasets into directory ./datasets/.**

The CMU-MOSI and CMU-MOSEI datasets can be downloaded according to [MIB](https://github.com/TmacMai/Multimodal-Information-Bottleneck) and [MAG](https://github.com/WasifurRahman/BERT_multimodal_transformer) through the following link: 
```
pip install gdown
gdown https://drive.google.com/uc?id=12HbavGOtoVCqicvSYWl3zImli5Jz0Nou
gdown https://drive.google.com/uc?id=1VJhSc2TGrPU8zJSVTYwn5kfuG47VaNQ3
```
For UR-FUNNY and MUStARD, the dataset can be downloaded according to [HKT](https://github.com/matalvepu/HKT/blob/main/dataset/download.txt) through:
```
Download Link of UR-FUNNY:  https://www.dropbox.com/s/5y8q52vj3jklwmm/ur_funny.pkl?dl=1
Download Link of MUsTARD: https://www.dropbox.com/s/w566pkeo63odcj5/mustard.pkl?dl=1
```
Please rename the files as ur_funny.pkl and mustard.pkl, and move them into the directory ./datasets/.

### Prerequisites:
```
* Python 3.8.10
* CUDA 11.5
* pytorch 1.12.1+cu113
* sentence-transformers 3.1.1
* transformers 4.30.2
```
**Note that the torch version can be changed to your cuda version, but please keep the transformers==4.30.2 as some functions will change in later versions**

### Pretrained model:
Downlaod the [BERT-base](https://huggingface.co/google-bert/bert-base-uncased/tree/main) , and put into directory ./BERT-EN/.

### Citation:
Please cite our paper if you find our work useful for your research:
```
@ARTICLE{Zhuang2025multi,
  author={Zhuang, Yan and Bai, Wei and Zhang, Yanru and Deng, Jiawen and Hu, Zheng and Zhang, Xiaoyue and Ren, Fuji},
  journal={IEEE Transactions on Multimedia}, 
  title={Multi-Level Contrastive Learning for Multimodal Sentiment Analysis}, 
  year={2025},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TMM.2025.3613116}}
```

### Acknowledgement
Thanks to  [MIB](https://github.com/TmacMai/Multimodal-Information-Bottleneck) , [MAG](https://github.com/WasifurRahman/BERT_multimodal_transformer),  [MCL](https://github.com/TmacMai/Multimodal-Correlation-Learning), [HKT](https://github.com/matalvepu/HKT), [LFMIM](https://github.com/sunjunaimer/LFMIM) for their great help to our codes and research. 

