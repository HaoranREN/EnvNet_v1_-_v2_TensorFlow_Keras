# EnvNet_v1_v2_TensorFlow_Keras
 
 An implementation of [EnvNet_v1](https://ieeexplore.ieee.org/document/7952651) and [EnvNet_v2](https://arxiv.org/abs/1711.10282) in Python with TensorFlow Keras.
 
 Train an example with [ESC-50](https://github.com/karolpiczak/ESC-50) dataset.


## Requirements
 - Numpy
 - Scipy
 - librosa (0.7+)
 - TensorFlow (1.14+)

## Description

 The [`EnvNet_v1.py`](EnvNet_v1.py) and [`EnvNet_v2.py`](EnvNet_v2.py) contain the model definition and the train/val/test methods.
 
 The [`EnvNet_v1_data_utils.py`](EnvNet_v1_data_utils.py) and [`EnvNet_v2_data_utils.py`](EnvNet_v2_data_utils.py) prepare the ESC-50 dataset for the model.

 Data preparation follows the [Envnet_v1 paper](https://ieeexplore.ieee.org/document/7952651):
 - Train and validate with a random selected window from each audio recoding
 - Test on sliding windows and predict at audio recording level with probability voting
 - Normalize data between -1 to 1
 - Remove silent window when maximum amplitude is smaller than 0.2
 
## Reference

[EnvNet_v1](https://ieeexplore.ieee.org/document/7952651):
```
@inproceedings{tokozume2017learning,
  title={Learning environmental sounds with end-to-end convolutional neural network},
  author={Tokozume, Yuji and Harada, Tatsuya},
  booktitle={2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={2721--2725},
  year={2017},
  organization={IEEE}
}
```

[EnvNet_v2](https://arxiv.org/abs/1711.10282):
```
@inproceedings{tokozume2017learning,
  title={Learning from between-class examples for deep sound recognition},
  author={Tokozume, Yuji and Ushiku, Yoshitaka and Harada, Tatsuya},
  journal={arXiv preprint arXiv:1711.10282},
  year={2017}
}
```

[ESC-50](https://github.com/karolpiczak/ESC-50):
```
@inproceedings{piczak2015esc,
  title={ESC: Dataset for environmental sound classification},
  author={Piczak, Karol J},
  booktitle={Proceedings of the 23rd ACM international conference on Multimedia},
  pages={1015--1018},
  year={2015}
}
```
