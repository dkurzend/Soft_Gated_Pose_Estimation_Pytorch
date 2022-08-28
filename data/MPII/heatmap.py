import numpy as np


class GenerateHeatmap():
    def __init__(self, output_res, num_parts):
        self.output_res = output_res
        self.num_parts = num_parts
        sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3

        # x,y are width, height of heatmap
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]   # shape (size, 1)
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    ## keypoints are already down scaled to 64x64 resolution
    def __call__(self, keypoints):
        # hms shape (num_keypoints, output_res, output_res) = (16, 64, 64)
        hms = np.zeros(shape = (self.num_parts, self.output_res, self.output_res), dtype = np.float32)
        sigma = self.sigma

        for p in keypoints:
            for idx, pt in enumerate(p):
                if pt[0] > 0:
                    # get key point coordinates x, y
                    x, y = int(pt[0]), int(pt[1])
                    # if key point coordinates invalid continue
                    if x<0 or y<0 or x>=self.output_res or y>=self.output_res:
                        continue

                    # upper left and bottom right coordinates
                    ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
                    br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

                    c,d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a,b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc,dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa,bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb,cc:dd] = np.maximum(hms[idx, aa:bb,cc:dd], self.g[a:b,c:d])

        return hms # shape [16, 64, 64]
