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

##### en_to_vi & en_to_vi_normal (tiếng việt không dấu)

- Dataset: Ted, IWSLT, PhoMT
- GPU: P100, Training time per epoch: 40'

##### zh_to_vi => zh_to_en & en_to_vi

- Dataset: BaiduAI, ParaCrawl_v9

- GPU: T4, Traning time per epoch: 2,5h

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

##### zh_to_vi (zh_to_en & en_to_vi)

```
Input:		   : 文章指出，要强化国家战略科技力量，提升国家创新体系整体效能。国家实验室、国家科研机构、高水平研究型大学、科技领军企业都是国家战略科技力量的重要组成部分，要自觉履行高水平科技自立自强的使命担当。	
Prediction1    : The article points out that we should strengthen the national strategic scientific and technological force and improve the overall efficiency of the national innovation system. National laboratories, national scientific research institutions, high-level research universities and science and technology leading enterprises are important components of the national strategic science and technology force. They should consciously fulfill the mission of self-reliance and self-improvement of high-level science and technology.
Prediction2    : Bài báo chỉ ra rằng chúng ta nên tăng cường lực lượng khoa học chiến lược và công nghệ và cải thiện hiệu quả toàn diện của hệ thống đổi mới quốc gia . Các phòng thí nghiệm quốc gia , các tổ chức nghiên cứu khoa học khoa học quốc gia , các trường đại học và khoa học cấp cao và công nghệ dẫn đến các thành phần quan trọng của khoa học chiến lược và công nghệ quốc gia . Họ cần hoàn thành nhiệm vụ tự cải thiện về khoa học và công nghệ cao .
```

