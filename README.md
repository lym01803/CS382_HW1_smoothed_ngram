# README
**刘彦铭 ID: 518030910393**
## 实现了两种算法

1. 基于狄利克雷分布的语言模型 (Section 1.1 in report)
2. 简单插值方法 (Section 1.2 in report)

## 复现方法
运行```python```脚本```ngram_model.py```, 参数格式:
```
usage: ngram_model.py [-h] --train TRAIN [--dev DEV] --test TEST
                      [--method METHOD] [--n N]
                      [--unk_threshold UNK_THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  --train TRAIN
  --dev DEV
  --test TEST
  --method METHOD (default: method in section 1.2)
  --n N (default: 2)
  --unk_threshold UNK_THRESHOLD (default: 0)
```

**Example**

**复现 1.1 中方法**

bigram
```
python .\ngram_model.py --train .\hw1_dataset\train_set.txt,.\hw1_dataset\dev_set.txt --test .\hw1_dataset\test_set.txt --method MacKay --n 2 --unk_threshold 0
```

**复现 1.2 中方法**

trigram
```
python .\ngram_model.py --train .\hw1_dataset\train_set.txt --dev .\hw1_dataset\dev_set.txt --test .\hw1_dataset\test_set.txt --n 3 
```

**注**

本地实测笔记本i7约几分钟能跑完训练+测试。方法1在n=3时迭代收敛略慢。不要被进度条所蒙骗，训练中的一些操作是均摊时间复杂度的，进度条的预计剩余时间会不准。