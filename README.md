# Contents

* [Other Info](#other-info)
* [Prerequisites](#prerequisites)
* [Code and Data Preparation](#Code_and_Data_Preparation)
* [Training and Testing  of BSN](#Training_and_Testing_of_BSN)




# Prerequisites
These code is  implemented in Pytorch 0.4.1 + Python2 + tensorboardX. Thus please install Pytorch first.

# Code and Data Preparation

## Get the code

Clone this repo with git, please use:

```
git clone https://github.com/wzmsltw/BSN-boundary-sensitive-network.pytorch.git
```



## Download Datasets

We support experiments with publicly available dataset ActivityNet 1.3 for temporal action proposal generation now. To download this dataset, please use [official ActivityNet downloader](https://github.com/activitynet/ActivityNet/tree/master/Crawler) to download videos from the YouTube.

To extract visual feature, we adopt TSN model pretrained on the training set of ActivityNet, which is the challenge solution of CUHK&ETH&SIAT team in ActivityNet challenge 2016. Please refer this repo [TSN-yjxiong](https://github.com/yjxiong/temporal-segment-networks) to extract frames and optical flow
refer this repo [anet2016-cuhk](https://github.com/yjxiong/anet2016-cuhk) to find pretrained TSN model.

For convenience of training and testing, we rescale the feature length of all videos to same length 100, and we provide the rescaled feature at here [Google Cloud](https://drive.google.com/file/d/1ISemndlSDS2FtqQOKL0t3Cjj9yk2yznF/view?usp=sharing) or [Baidu Yun](). If you download features using BaiduYun, please use `cat zip_csv_mean_100.z* > csv_mean_100.zip` before unzip.
After download and unzip, please put `csv_mean_100` directory to `./data/activitynet_feature_cuhk/` .

# Training and Testing  of BSN

All configurations of BSN are saved in opts.py, where you can modify training and model parameter.


#### 1. Training of temporal evaluation module


```
python main.py --module TEM --mode train
```

We also provide trained TEM model in `./checkpoint/`

#### 2. Testing of temporal evaluation module

```
python main.py --module TEM --mode inference
```

#### 3. Proposals generation and BSP feature generation

```
python main.py --module PGM
```

#### 4. Training of proposal evaluation module

```
python main.py --module PEM --mode train
```

We also provide trained PEM model in `./checkpoint` .

#### 6. Testing of proposal evaluation module

```
python main.py --module PEM --mode inference --pem_batch_size 1
```

#### 7. Post processing and generate final results

```
python main.py --module Post_processing
```

#### 8. Eval the performance of proposals

```
python main.py --module Evaluation
```
You can find evaluation figure in `./output`

You can also simply run all above commands using:
```
sh bsn.sh
```
