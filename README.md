# ViT_Classification_Sample

## 動作環境
<details>
<summary>ライブラリのバージョン</summary>
 
* Ubuntu 18.04
* Geforce RTX 4090
* driver 530.30.02
* cuda 12.1
* python 3.6.9
* torch 1.8.1+cu111
* torchaudio  0.8.1
* torchinfo 1.5.4
* torchmetrics  0.8.2
* torchsummary  1.5.1
* torchvision 0.9.1+cu111
* timm  0.5.4
* tlt  0.1.0
* numpy  1.19.5
* Pillow  8.4.0
* scikit-image  0.17.2
* scikit-learn  0.24.2
* tqdm  4.64.0
* opencv-python  4.5.1.48
* opencv-python-headless  4.6.0.66
* scipy  1.5.4
* matplotlib  3.3.4
* mmcv  1.7.1
</details>

## ファイル＆フォルダ一覧

<details>
<summary>学習用コード等</summary>
 
|ファイル名|説明|
|----|----|
|vit_train.py|ViTを学習するコード．|
|mae_train.py|ViTを学習するコード(Masked Autoencoder(MAE)で事前学習したTransformer Encoderを使用)．|
|cait_train.py|CaiTを学習するコード．|
|deit_train.py|DeiTを学習するコード．|
|trainer.py|学習ループのコード．|
|vis_att.py|Attention Mapを可視化するコード．|
|vis_class_att.py|Class AttentionのAttention Mapを可視化するコード．|
|attention_rollout.py|Attention RolloutでAttention Mapを可視化するコード．|
|make_graph.py|学習曲線を可視化するコード．|
</details>

## 実行手順

### 環境設定

[先述の環境](https://github.com/cu-milab/ra-takase-2020/tree/master/Code/ViT_sample#%E5%8B%95%E4%BD%9C%E7%92%B0%E5%A2%83)を整えてください．

### 学習済みモデルのダウンロード
MAEで学習したTransformer Encoderをファインチューニングする場合は学習済みのモデルをダウンロードしてください．
<details>
<summary>学習済みのモデル</summary>
 
MAE：http://gofile.me/77OyG/Ez4XwKwo8

</details>

### 学習
ハイパーパラメータは適宜調整してください．

※ ImageNetなどの大きなデータセットで学習する場合はRandAugment，MixUp，CutMix，Random ErasingなどのData Augmentationの追加やWarmUp Epoch，Label Smoothing，Stochastic Depthなどを導入してください．

<details>
<summary>ViT，MAE，DeiT，CaiTのファインチューニング(CIFAR-10)</summary>
 
```
python3 vit_train.py --epoch 10 --batch_size 128 --amp --dataset cifar10 --warmup_t 0 --warmup_lr_init 0
```
```
python3 mae_train.py --epoch 10 --batch_size 128 --amp --dataset cifar10 --warmup_t 0 --warmup_lr_init 0
```
```
python3 deit_train.py --epoch 10 --batch_size 128 --amp --dataset cifar10 --warmup_t 0 --warmup_lr_init 0
```
```
python3 cait_train.py --epoch 10 --batch_size 128 --amp --dataset cifar10 --warmup_t 0 --warmup_lr_init 0
```
</details>

<details>
<summary>ViT，MAE，DeiT，CaiTのファインチューニング(CIFAR-100)</summary>
 
```
python3 vit_train.py --epoch 10 --batch_size 128 --amp --dataset cifar100 --warmup_t 0 --warmup_lr_init 0
```
```
python3 mae_train.py --epoch 10 --batch_size 128 --amp --dataset cifar100 --warmup_t 0 --warmup_lr_init 0
```
```
python3 deit_train.py --epoch 10 --batch_size 128 --amp --dataset cifar100 --warmup_t 0 --warmup_lr_init 0
```
```
python3 cait_train.py --epoch 10 --batch_size 128 --amp --dataset cifar100 --warmup_t 0 --warmup_lr_init 0
```
</details>

### アテンションマップの可視化

<details>
<summary>ViT，MAE，DeiTの場合</summary>
 
```
python3 vis_att.py 
```
</details>

<details>
<summary>Attention Rolloutの場合</summary>
 
```
python3 attention_rollout.py 
```
</details>

<details>
<summary>CaiTの場合</summary>
 
```
python3 vis_class_att.py 
```
</details>

## 参考文献
* 参考にした論文
  * ViT
    * An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
  * DeiT
    * Training data-efficient image transformers & distillation through attention
  * CaiT
    * Going deeper with Image Transformers
  * MAE
    * Masked Autoencoders Are Scalable Vision Learners
  * RandAugment
    * RandAugment: Practical automated data augmentation with a reduced search space
  * MixUp
    * mixup: Beyond Empirical Risk Minimization
  * CutMix
    * CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
  * Random Erasing
    * Random Erasing Data Augmentation
  * Attention Rollout
    * Quantifying Attention Flow in Transformers

* 参考にしたリポジトリ 
  * timm
    * https://github.com/huggingface/pytorch-image-models
  * ViT
    * https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
  * DeiT
    * https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/deit.py
  * CaiT
    * https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/cait.py
  * MAE
    * https://github.com/facebookresearch/mae
  * RandAugment
    * https://github.com/ildoonet/pytorch-randaugment
  * MixUp
    * https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/mixup.py
  * CutMix
    * https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/mixup.py
  * Random Erasing
    * https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/random_erasing.py
  * Attention Rollout
    * https://gihyo.jp/book/2022/978-4-297-13058-9/support