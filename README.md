# [ECCV 2024] OneNIP
Official PyTorch Implementation of [Learning to Detect Multi-class Anomalies with Just One Normal Image Prompt](https://csgaobb.github.io/Pub_files/ECCV2024_OneNIP_CR_Full_0725_Mobile.pdf), Accepted by ECCV 2024.

![Image text](docs/OneNIP-Framework.png)

OneNIP mainly consists of unsupervised reconstruction, unsupervised restoration, and supervised refiner.

## 1. Comparsions of OneNIP and UniAD

![Alt text](docs/OneNIPvsUniAD.png)

## 2. Results and Checkpoints. 
All pre-trained model weights are stored in Google Drive.

| Dataset |  Input-Reslution | I-AUROC | P-AUROC | P-AUAP | Checkpoints | Test-Log|
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | 
| MVTec |  224 $\times$ 224  |  97.9  |  97.9  |  63.7  |  [model weight](https://drive.google.com/file/d/1q6gMbBKrF-sM1822KlFhmj-jCbMEdBMa/view?usp=sharing) | [testlog](./checkpoints/onenip-mvtec-4-4-224/log/dec_20240921_215951.log)  | 
| MVTec |  256 $\times$ 256  |  97.6  |  97.9  |  64.8  |  [model weight](https://drive.google.com/file/d/1eVXrncc7iRtaNQpyHk3cQ1QQlPBhXpuF/view?usp=sharing) | [testlog](./checkpoints/onenip-mvtec-4-4-256/log/dec_20240921_220207.log)  | 
| MVTec |  320 $\times$ 320  |  97.9  |  97.9  |  65.9  |  [model weight](https://drive.google.com/file/d/19xK8nksu1uBG-Affbcu6cZmaj10cMeSL/view?usp=sharing) | [testlog](./checkpoints/onenip-mvtec-4-4-320/log/dec_20240921_220430.log)  | 
||
| VisA  |  224 $\times$ 224  |  92.5  |  98.7  |  43.3  |  [model weight](https://drive.google.com/file/d/16r5pq5CBVPgu2jMizVJW83K0oB_xdmFl/view?usp=sharing) | [testlog](./checkpoints/onenip-visa-4-4-224/log/dec_20240921_221901.log)  | 
| VisA  |  256 $\times$ 256  |  93.1  |  98.8  |  44.9  |  [model weight](https://drive.google.com/file/d/1ZV2Hh5oniMW1cePsRQ_RPgkCBIOQHoIi/view?usp=sharing) | [testlog](./checkpoints/onenip-visa-4-4-256/log/dec_20240921_225047.log)  |
| VisA  |  320 $\times$ 320  |  94.2  |  98.8  |  46.1  |  [model weight](https://drive.google.com/file/d/17DX4ukJIzMAKYfLPMu1yp3VbvFfXNCvo/view?usp=sharing) | [testlog](./checkpoints/onenip-visa-4-4-320/log/dec_20240921_220825.log)  | 
||
| BTAD  |  224 $\times$ 224  |  92.6  |  97.4  |  56.8  |  [model weight](https://drive.google.com/file/d/1drMQZubI3dFz0yNXJuyTOU4DmmFkGNEc/view?usp=sharing) | [testlog](./checkpoints/onenip-btad-4-4-224/log/dec_20240921_221227.log)  | 
| BTAD  |  256 $\times$ 256  |  94.6  |  97.6  |  57.0  |  [model weight](https://drive.google.com/file/d/1avzuJQLd2Xd_7hUEG25s1ev7cMqNUNJz/view?usp=sharing) | [testlog](./checkpoints/onenip-btad-4-4-256/log/dec_20240921_221334.log)  | 
| BTAD  |  320 $\times$ 320  |  95.3  |  97.8  |  57.6  |  [model weight](https://drive.google.com/file/d/1jRyIrwR96tAgjvdvLJ8346Hylugr0rmu/view?usp=sharing) | [testlog](./checkpoints/onenip-btad-4-4-320/log/dec_20240921_235736.log)  |
|| 
|MVTec+VisA+BTAD| 224 $\times$ 224 |  94.5  |  98.0  |  52.4  |   [model weight](https://drive.google.com/file/d/17sccEGFcFYFOwDp6e3Mh0a5QK8iOUeFT/view?usp=sharing) | [testlog](./checkpoints/onenip-mvtec+btad+visa-4-4-224/log/dec_20240921_230615.log)  | 
||

## 3. Evaluation and Training

### 3.1 Prepare data
Download [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad), [BTAD](https://avires.dimi.uniud.it/papers/btad/btad.zip), [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar) and [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz) datasets. Unzip and move them to `./data`. The data directory should be as follows.
```
├── data
│   ├── btad
│   │   ├── 01
│   │   ├── 02
│   │   ├── 03
│   │   ├── test.json
│   │   ├── train.json
│   ├── dtd
│   │   ├── images
│   │   ├── imdb
│   │   ├── labels
│   ├── mvtec
│   │   ├── bottle
│   │   ├── cable
│   │   ├── ...
│   │   └── zipper
│   │   ├── test.json
│   │   ├── train.json
│   ├── mvtec+btad+visa
│   │   ├── 01
│   │   ├── bottle
│   │   ├── ...
│   │   └── zipper
│   │   ├── test.json
│   │   ├── train.json
│   ├── visa
│   │   ├── candle
│   │   ├── capsules
│   │   ├── ...
│   │   ├── pipe_fryum
│   │   ├── test.json
│   │   ├── train.json
```


### 3.2 Evaluation with pre-trained checkpoints
Download pre-trained checkpoints to `./checkpoints`
```
cd ./exps
bash eval_onenip.sh 8 0,1,2,3,4,5,6,7
```


### 3.3 Training OneNIP

```
cd ./exps
bash train_onenip.sh 8 0,1,2,3,4,5,6,7
```

## Citing

If you find this code useful in your research, please consider citing us:
```
@inproceedings{gao2024onenip,
  title={Learning to Detect Multi-class Anomalies with Just One Normal Image Prompt},
  author={Gao, Bin-Bin},
  booktitle={18th European Conference on Computer Vision (ECCV 2024)},
  pages={-},
  year={2024}
}
```



## Acknowledgement

Our OneNIP is build on [UniAD](https://github.com/zhiyuanyou/UniAD). Thank the authors of UniAD for open-sourcing their implementation codes!
