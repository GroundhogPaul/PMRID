# 1. 罗文标定出的Jn1镜头的噪声：

## 1.1 标定结果

```python
    assert AGain >= 1.0
    assert AGain <= 1024
    if AGain >= 4.0:
        line = lambda x: 4.675e-05 * x + 0.0003418
        sig_read = line(AGain)
        sig_read = np.clip(sig_read, 0.0, 65535.0)
        var_read = sig_read*sig_read
    else:
        line = lambda x: 1.822e-4 * x + 0.0003203
        sig_read = line(AGain)
        sig_read = np.clip(sig_read, 0.0, 65535.0)
        var_read = sig_read*sig_read

    line = lambda x: 3.674e-05 * x
    # shot_noise = line(AGain) + np.random_normal((), stddev=5.007*1.021e-05)
    var_shot = line(AGain) # + np.random_normal((), stddev=5.007*1.021e-05)
var_shot = np.clip(var_shot, 0.0, 65535.0)
```

 代入Again=64，得到

sig_read = 0.00334

var_read = 1.111422244e-05

var_shot = 0.00235136

## 1.2 与Google训练范围的关系

![1.2_GoogleNoiseRange.jpg](.\ImgSet\1.2_GoogleNoiseRange.jpg)

这里的λread和λshot对应的是1.1代码中的var_read和var_shot

**var_shot的范围**

根据官方提供的代码，这里的log指的是10对数，代码中使用的所谓tf.log和tf.exp只是手段，真正起作用的是0.0001和0.012这两个数，对应的是10-4.0和10-1.921

则google训练框架中，

var_shot的范围为[10-4.0,
10-1.921] = [0.0001, 0.012]

**var_read的范围**

       google文章的(3)式写了：

log (λread ) ∼ N (µ = 2.18 log (λshot ) + 1.2, σ = 0.26)

取3σ为合理上下界范围，

当var_shot取下界-4.0时，log (λread )的范围为：

[2.18*-4+1.2-3*0.26, 2.18*-4+1.2+3*0.26] = [-8.3, -6.74]

              则 var_shot = 1e-4时，var_read ∈ [10^-8, 10^-6.74] = [1e-8,
1.82e-7]

当var_shot取上界-1.921时，log (λread )的范围为：

[2.18*-1.921+1.2-3*0.26, 2.18*-1.921+1.2+3*0.26] = [-3.77, -2.21]

              则 var_shot = 1e-4时，var_read ∈ [10^-3.77, 10^-2.21] = [1.70e-4, 6.16e-3]

对应罗文的var_shot = 0.00235

       log(var_shot)
= -2.629

log (λread )的范围为 [2.18*-2.629+1.2-3*0.26, 2.18*-2.629+1.2+3*0.26] =
[-5.31, -3.75]

var_read ∈ [10^-5.31, 10^-3.75] = [4.9e-6, 1.8e-4]

罗文标定的var_read为 1.111422244e-05

结论：如果认为罗文的标定是准确的，则Jn1在64倍Again下的var_read和shot_read在Google文章Unprocessing Images for Learned Raw Denoising图3的范围内
