import os
import numpy as np
import cupy as cp
from PIL import Image
import chainer
import chainer.backends.cuda
from chainer import Variable, serializers
from net import Generator, Discriminator

def read_image_as_array(path, dtype):
    f = Image.open(path)
    try:
        image = np.asarray(f, dtype=dtype)
    finally:
        # Only pillow >= 3.0 has 'close' method
        if hasattr(f, 'close'):
            f.close()
    return image

def discriminate():
    dis = Discriminator()
    chainer.backends.cuda.get_device_from_id(0).use()
    serializers.load_npz("./result/dis_iter_25000.npz", dis)
    dis.to_gpu()  # Copy the model to the GPU
    paths = ["./data/70.png", "./data/35.png"]
    for path in paths:
        img = read_image_as_array(path, dtype=np.float32)
        img = img.reshape(-1, img.shape[0], img.shape[1], img.shape[2]).transpose(0,3,1,2)
        img = Variable(chainer.backends.cuda.to_gpu(img))
        with chainer.using_config('train', False):
            x = dis(img)
        x = chainer.backends.cuda.to_cpu(x.data)
        print(x)

def main():
    gen = Generator(n_hidden=100)
    chainer.backends.cuda.get_device_from_id(0).use()
    serializers.load_npz("./result/gen_iter_25000.npz", gen)
    gen.to_gpu()  # Copy the model to the GPU
    start = np.random.uniform(-1, 1, (100, 1, 1)).astype(np.float32)
    end = np.random.uniform(-1, 1, (100, 1, 1)).astype(np.float32)
    diff = end - start
    for i in range(50):
        arr = start + i*diff/50
        z = Variable(chainer.backends.cuda.to_gpu(arr.reshape(1,100,1,1)))
        with chainer.using_config('train', False):
            x = gen(z)
        x = chainer.backends.cuda.to_cpu(x.data)
        x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
        x = x.reshape(3,96,96).transpose(1,2,0)
        Image.fromarray(x).save("./continuous/" + str(i) + ".png")

if __name__=="__main__":
    main()
