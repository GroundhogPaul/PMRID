import numpy as np
from typing import Tuple

print(" ------ using Offical K Sigma params from PMRID ------ ")
Official_Ksigma_params = {
    "K_coeff" : [0.0005995267, 0.00868861],
    "B_coeff" : [7.11772e-7, 6.514934e-4, 0.11492713],
    "anchor" : 1600
    }   #  PMRID sensor

print(" ------ using K Sigma params from Jn1 caliberation from LuoWen: Again > 4.0 ------ ")
Official_Ksigma_params = {
    #  LuoWen Jn1, Again >= 4.0, 
    #  sigRead = 4.675e-05 * x + 0.0003418
    #  varShot = 3.674e-05 * x
    "K_coeff" : [3.758e-04, 0.0000],
    "B_coeff" : [2.287231e-07, 3.3444979e-4, 0.12226],
    "anchor" : 1600
    }

# Official_Ksigma_params = {
#     "K_coeff" : [0.0005995267, 0.00868861],
#     "B_coeff" : [7.11772e-7, 6.514934e-4, 0.11492713],
#     "anchor" : 1600 # TODO TODO
#     }   #  LuoWen Jn1, Again > ?? # TODO TODO

class KSigma:

    def __init__(self, K_coeff: Tuple[float, float], B_coeff: Tuple[float, float, float], anchor: float, V: float = 959.0, k = None, sigma = None):
        self.K = np.poly1d(K_coeff)
        self.Sigma = np.poly1d(B_coeff)
        self.anchor = anchor
        self.V = V

        self.k_default = k
        self.sigma_default = sigma

        self.iso_last = -1.0
    
    def __call__(self, img_01, iso: float, inverse=False):
        if self.k_default is None or self.sigma_default is None:
            k, sigma = self.K(iso), self.Sigma(iso)
        else:
            k, sigma = self.k_default, self.sigma_default
        assert iso > 0, "ISO should be larger than 0, current ISO: {}".format(iso)
        assert iso <= 12800, "ISO should be less than 12800, current ISO: {}".format(iso)
        k_a, sigma_a = self.K(self.anchor), self.Sigma(self.anchor)

        cvt_k = k_a / k
        cvt_b = (sigma / (k ** 2) - sigma_a / (k_a ** 2)) * k_a

        img = img_01 * self.V

        if not inverse:
            img = img * cvt_k + cvt_b
        else:
            img = (img - cvt_b) / cvt_k

        return img / self.V

    def GetKSigma(self, iso: float):
        if self.k_default is not None and self.sigma_default is not None:
            self.iso_last = iso
            return self.k_default, self.sigma_default
        self.iso_last = iso
        k = self.K(iso)
        sigma = self.Sigma(iso)

        return k, sigma


if __name__ == '__main__':
    kSigmaCur = KSigma(K_coeff = Official_Ksigma_params['K_coeff'], 
                       B_coeff = Official_Ksigma_params['B_coeff'], 
                       anchor = Official_Ksigma_params['anchor'])

    iso = 6400
    k, sigma = kSigmaCur.GetKSigma(iso)

    print(f"k = {k:0.2f}")
    print(f"sigma = {sigma:0.2f}")

    # ----- read noise  from LuoWen to PMRID ----- #
    sig_read = 4.675e-05* 64 + 0.0003418
    Again = 6400.0
    sig_read = 4.675e-07 * Again + 0.0003418
    var_read = (1023*(4.675e-07 * Again + 0.0003418))**2 
    var_read = (Again*4.7825e-4 + 0.34966) **2
    var_read = (Again*4.7825e-4)**2.0 + 2.0*(Again*0.34966*4.7825e-4) + 0.34966**2
    var_read = (Again**2)*2.287231e-07 + Again*3.3445e-4 + 0.12226

    # ----- shot noise  from LuoWen to PMRID ----- #
    sig_shot = 3.674e-05 * 64 * 1023
    print(sig_shot)
    Again = 6400.0
    print(3.674e-07 * Again * 1023)
    print(3.758e-04 * Again)