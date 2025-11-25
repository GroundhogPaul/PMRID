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


