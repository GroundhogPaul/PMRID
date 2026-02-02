import numpy as np
import matplotlib.pyplot as plt

def lr_official(epoch, epochMax):
    assert epoch >= 0
    assert epoch < epochMax

    epochMaxHalf = epochMax // 2
    Ntriangle = 40
    Ttriangle = epochMaxHalf // Ntriangle
    TtriangleHalf = Ttriangle / 2
    lrMax = 1e-3
    lrMin = 1e-5
    if epoch < epochMaxHalf:

        epoch = epoch % Ttriangle
        if epoch < TtriangleHalf:
            lr = lrMax - (lrMax - lrMin) * (epoch /TtriangleHalf)
        else:
            lr = lrMin + (lrMax - lrMin) * ((epoch - TtriangleHalf) / TtriangleHalf)
    elif epoch < epochMax:
        epoch = epoch - epochMaxHalf
        lr = lrMax - (lrMax - lrMin) / epochMaxHalf * epoch 
    return lr

if __name__ == "__main__":
    epochMax = 8000
    lstEpochs = list(range(0, epochMax, 1))
    lstLrs = [lr_official(epoch, epochMax=epochMax) for epoch in lstEpochs]
    plt.plot(lstEpochs, lstLrs)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Official Learning Rate Schedule')
    plt.grid()
    plt.show()

