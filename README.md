# Welcome to the repository of Just Another (Pytorch) Wrapper!

## What is JAW?

JAW (Just Another (Pytorch) Wrapper) is, like is name speaks for itself, a little Pytorch wrapper and above all a proposal of project architecture for Machine Learning. The main idea is to show how write is own little ML framework based on Pytorch, but with an architecture flexible enought to permit the use of another library (such as keras or Tensorflow).

## Installation

For the installation of JAW and Example modules , use:
```python3 setup.py sdist```
for generate the archive, then install it via pip:
```pip install dist/jaw-0.1.tar.gz```

## Example

For run the training script given in example, use the following command:

```python3 src/example/fashionMNIST/simple_MLP_classifier.py --epochs 5 --loss RelativeL2 --model FCReg --logdir logs```

## Tutorial

[Here](https://yvregon.github.io/JAW/) you will find a tutorial for how use JAW for create its own ML project. You will also find the package's documentation as well as that of the example generated during the tutorial.
