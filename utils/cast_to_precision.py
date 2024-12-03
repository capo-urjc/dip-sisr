import numpy as np


class CastToPrecision:
    def __init__(self, precision: int = 16):
        self.precision = precision

    def __call__(self, image: np.ndarray):
        if self.precision == 16:
            return image.astype(np.float16)

        elif self.precision == 32:
            return image.astype(np.float32)

        elif self.precision == 64:
            return image.astype(np.float64)


# if __name__ == "__main__":
# image = np.random.randint(low=0, high=2**16, size=(256, 256, 35))
# image = {'x': image, 'y': image}
# normaliza = Normalize(param=5000)
# output = normaliza(image)
# print(1)
# # output = normaliza.forward(image)
