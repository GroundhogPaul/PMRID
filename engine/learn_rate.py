import numpy as np
import matplotlib.pyplot as plt

def lr_official(epoch):
    assert epoch >= 0
    assert epoch < 8000
    lrMax = 1e-3
    lrMin = 1e-5
    if epoch < 4000:
        epoch = epoch % (100)
        if epoch < 50:
            lr = lrMax - (lrMax - lrMin) * (epoch / 50)
        else:
            lr = lrMin + (lrMax - lrMin) * ((epoch - 50) / 50)
    elif epoch < 8000:
        epoch = epoch - 4000
        lr = lrMax - (lrMax - lrMin) / 4000 * epoch 
    return lr

if __name__ == "__main__":
    lstEpochs = list(range(0, 8000, 1))
    lstLrs = [lr_official(epoch) for epoch in lstEpochs]
    plt.plot(lstEpochs, lstLrs)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Official Learning Rate Schedule')
    plt.grid()
    plt.show()

