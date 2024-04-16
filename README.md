# unet-easy: an easy implementation of unet in diffusion models

UNet  gets $x_t$ (noisy image) and $t$ (time step) as input, and outputs $\epsilon_t$ (the noise in the image). 

<!-- smaller image -->
![alt text](imgs/u1.png)

So we can define the UNet as follows:

```python
import torch.nn as nn
import torch


class UNET(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, t):
        return x
```

Here are the inputs and outputs of the UNet:
- x `[batch_size, channels, height, width]` is the noisy image.
- t `[batch_size]` is the time step.
- x `[batch_size, channels, height, width]` in `return x` is the noise in the image.

We will use the following random input to test the UNet:

```python
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 3, 224, 224).to(device)
    t = torch.randint(0, 100, (2,)).to(device)
    model = UNET().to(device)
    out = model(x, t)
    print(out.shape)
```

The current output is `torch.Size([2, 3, 224, 224])`, which is the same shape as the input.

> `to(device)` will bring obvious speedup todiffusion models, so I suggest you to add it even you are just testing the code.

Then we need to design the shape change in the UNet.

First, we use a `conv_in` to make [2, 3, 224, 224] to [2, 64, 224, 224].

Then, we use `downs` to downsample the feature map.

Specifically, the change of channels in downsample is: 64 -> 64 -> 128 -> 256 -> 512, while the change of size in downsample is: 224 -> 112 -> 56 -> 28 -> 28.

Next, we use `conv_mid` to make [2, 512, 28, 28] to [2, 512, 28, 28], does not change any shape.

Then, we use `ups` to upsample the feature map.

Here is very important and not easy to align the shape change.

You can think like this:

We save the feature map of down layer 1, 2, 3.

Then the output of conv_mid is is actually the output of last down layer.

The output of down layer 3 is [256, 28, 28], while the output of down layer 4 is [512, 28, 28].

When in upsample, we need to concat 512+ 256 -> 512, 512 + 256 -> 512 -> [256, 56, 56] is the output of up layer 1.


output of down layer 2 is [128, 56, 56], we concat it with the output of up layer 1 -> [256, 56, 56].

output of down layer 1 is [64, 112, 112], we concat it with the output of up layer 2 -> [256, 112, 112].

output of conv_in is [64, 224, 224], we concat it with the output of up layer 3 -> [64, 224, 224].

output of conv_in is [64, 224, 224], we concat it with the output of up layer 4 -> [64, 224, 224].




Specifically, the change of channels in upsample is: 512 -> 256 -> 128 -> 64 -> 64, while the change of size in upsample is: 28 -> 56 -> 112 -> 224 -> 224.

Finally, we use `conv_out` to make [2, 64, 224, 224] to [2, 3, 224, 224], which is the same shape as the input.



We will use a architecture like this:
1. conv_in: 2, 3, 224, 224 -> 2, 64, 224, 224
2. downs: 2, 64, 224, 224 -> 2, 512, 28, 28
3. conv_mid: 2, 512, 28, 28 -> 2, 512, 28, 28
4. ups: 2, 512, 28, 28 -> 2, 64, 224, 224
5. conv_out: 2, 64, 224, 224 -> 2, 3, 224, 224

In detail, the change of channels in downsample and upsample is:
- downs: 64 -> 64 -> 128 -> 256 -> 512
- ups: 512 -> 256 -> 128 -> 64 -> 64

The change of size in downsample and upsample is:
- downs: 224 -> 112 -> 56 -> 28 -> 28
- ups: 28 -> 56 -> 112 -> 224 -> 224

The illustration of the architecture is shown below:


![alt text](imgs/u2.png)






















![unet](imgs/unet.png)