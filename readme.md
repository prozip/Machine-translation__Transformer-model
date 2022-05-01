## Transformer?
- Using multiple self-attention layer intead of RNNs

## Model
- Building from scratch (for learning)
- Tensorflow & Keras low-level api

## Use

#### Create Converter model & Train

```
python train.py -t {tfds|local} -e {epoch} {path_to_dataset} 
```

#### Run Translator

```
python translator.py -m {model} {input}
```

#### Export to tfmodel

```
python translator.py -m {model} export
```

## Pretrain Sample

#### Model Info

- Name: en_to_vi & en_to_vi_normal (tiếng việt không dấu)
- Dataset: Ted, IWSLT, PhoMT
- GPU: P100, Training time per epoch: 40'

#### Sample output

##### en_to_vi (full version)

```
Input:         : A group of about 20 civilians has left the Azovstal steelworks in Mariupol, the final part of the southern city still in the hands of Ukrainian troops.
Prediction     : Một nhóm khoảng 20 người dân sự đã rời khỏi các nhà máy thép Azovastal ở Mariupol , phần cuối cùng của thành phố phía nam vẫn nằm trong tay của quân Ukraine .
```

##### en_to_vi_normal (tiếng việt không dấu)

```
Input:         : A group of about 20 civilians has left the Azovstal steelworks in Mariupol, the final part of the southern city still in the hands of Ukrainian troops.
Prediction     : mot nhom cua khoang 20 dan su đa roi khoi cac vu thep azovst o mariupol , phan cuoi cung cua thanh pho phia nam van nam trong tay quan đoi ukraine .
```

