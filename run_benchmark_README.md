run_benchmark.py能跑出文章中提到的结果，则其中的KSigma奏效，由于KSigma变换贯穿训练和推理，所以值得研究清楚

# 推理时

class KSigma的__call__中，输入的归一化mosaic图像，经过如下操作：

    a. 乘以标定域的 WhiteLevel - BlackLevel，**黑白电平**也仅在归一化时使用，没有显示的减去**黑电平**的操作

```python
img = img_01 * self.V
```

    b. 用当前ISO进行KSigma变换，再用anchor ISO进行反KSigma变换。anchor ISO疑似训练时用的ISO设置。

```python
img = img * cvt_k + cvt_b
```

    c. 除以标定域的WhiteLevel-BlackLevel，回到归一化域  

```python
return img / self.V
```

# 训练时

raw图训练应该遵循一样的方法，除了在步骤a之前还要加上噪声之外，其他应该没有变化

而加噪声这一步应该就在上一段的a和b之间



# bayer转rgb对PSNR的影响

根据 run_benchmark.py的测试，bayer的PSNR统一都比RGB高。这一结论也符合文章 Unprocessing Images for Learned Raw Denoising 中，table 1 的结果。



通过控制变量，发现，wb_gain， CCM和gamma都会拉低rgb域的PSNR水平

定性的理解如下：

1. wb_gain: 给三通道都乘以了>1.0的数，而图像的dynamic range(DR)定义没变，噪声相对DR下降了。

2. CCM：三通道的噪声互相影响了

3. gamma：给三通道都乘以了>1.0的数，原理同1. wb_gain








