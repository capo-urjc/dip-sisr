import numpy as np


class Normalize:
    def __init__(self, param: float=5000):
        self.param = param

    def __call__(self, image: np.ndarray):
        return image/self.param


if __name__ == "__main__":
    image = np.random.randint(low=0, high=2**16, size=(256, 256, 35))
    image = {'x': image, 'y': image}
    normaliza = Normalize(param=5000)
    output = normaliza(image)
    print(1)
    # output = normaliza.forward(image)
