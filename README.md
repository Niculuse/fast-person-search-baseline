# fast-person-search-baseline
Re-implement of the baseline of paper "Making person search enjoy the merits of person re-identification".
## Prerequisites
- Python 3.9.7
- Pytorch 1.10
- torchvision 0.11.2
- tqdm 4.15.0
- sklearn 0.24.2

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
   
## Train Baseline CNN Model
You may train a baseline cnn model by following command:
```bash
sh train_baseline_cnn.sh
```
You can edit the file `train_baseline_cnn.sh` with training settings you want.
## Extracting CNN Features (Preparing training set for GCN Model)
1. You may extract cnn features by following command:

```bash
sh extract_baseline_cnn.sh
```

Before runing the command above, you should edit the corresponding settings in the file `extract_baseline_cnn.sh`, for example, the model path and the dataset name. Then, you should obtain a file `feats.pkl` under the model path folder.

2. Make `feats` directory under this repository root folder. Copy the file `feats.pkl` to corresponding folder under the folder `feats`. Data structure should look like:

```bash
datasets/
feats
    market1501
        feats.pkl
    dukemtmc
        feats.pkl
    cuhk03
        feats.pkl
```
## Training GCN Model
 You may train our GCN mdel via following command:
 
 ```bash
 sh train_graph.sh
 ```
 
 You should edit the file `train_graph.sh` before you running it to make sure that training settings are what you want.
## Citation

