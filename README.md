# fast-person-search-baseline
Re-implement of the baseline of paper "Making person search enjoy the merits of person re-identification".
## Prerequisites
- Python 3.9.7
- Pytorch 1.10
- torchvision 0.11.2
- tqdm 4.15.0
- scikit-learn 1.0.1
- pandas 1.4.2
## Features
- AMP training
- Large batch size
## Get Started
1. Download this repository by running `git clone https://github.com/Niculuse/fast-person-search-baseline.git` or clicking the button `clone or download`.

2. Prepare datasets

    You may download the person search datasets [PRW](https://github.com/liangzheng06/PRW-baseline) and [CUHK-SYSU](https://github.com/ShuangLI59/person_search) first, and then prepare the datasets via following commands:
    
    ```bash
    cd fast-person-search-baseline
    mkdir data
    ```
    
    (1) Market1501 and DukeMTMC
    
    * Extract the datasets and rename them to `prw` and `sysu`, respectively.
    * Copy the folder `prw` and `sysu` to the folder `data`. The data structure should be like:
    
    ```bash
    data
        prw
            frames/
            annotations/
            ......
        sysu
            frames/
            annotation/
            ...... 
    ```
   
## Train the Baseline
You may train a baseline model by following command:
```bash
sh train.sh
```
With the default settings. You may get results as follows:
```
PRW:  mAP:48.6%, top-1 86.7%
CUHK-SYSU: mAP:86.7%, top-1 88.4%
```
You can edit the file `train.sh` with training settings you want.
## Citation
If you find this repository useful for your research, please cite the paper:

```
@article{liu2022making,
  title={Making person search enjoy the merits of person re-identification},
  author={Liu, Chuang and Yang, Hua and Zhou, Qin and Zheng, Shibao},
  journal={Pattern Recognition},
  volume={127},
  pages={108654},
  year={2022},
  publisher={Elsevier}
}
```

