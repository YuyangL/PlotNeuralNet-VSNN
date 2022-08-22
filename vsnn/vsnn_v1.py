import sys

sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import *

# User inputs
cl0, cl1, cl2, el2, el1, el0 = ({},) * 6
# Contraction
cl0['conv'] = 64
cl0['incept'] = 128

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # Input
    to_input('../vsnn/vsnn_v1_input.png', width=10, height=8, name='input'),

    # CL0
    to_Conv(name='cl0-c', s_filer="", n_filer=cl0['conv'], offset="(1,0,0)", to="(0,0,0)",
            width=3, height=40, depth=50, caption='C0-1'),
    to_Inception(name='cl0-i', s_filer="200x160", n_filer=cl0['incept'], offset="(0,0,0)", to="(cl0-c-east)",
                 width=5, height=40, depth=50, caption="I0-1"),
    to_Pool(name="cl0-pool", offset="(0,0,0)", to="(cl0-i-east)", width=1, height=20, depth=25, opacity=0.5),

    # CL1
    *block_ConvInceptionPool(name='cl1', bottom='cl0-pool', top='cl1-pool', s_filer="100x80", n_filer=(128, 256),
                             offset="(2, -5, 0)", size=((20, 25, 6), (20, 25, 8)), opacity=0.5, label='1-2'),

    # CL2
    *block_ConvInceptionPool(name='cl2', bottom='cl1-pool', top='cl2-pool', s_filer="50x40", n_filer=(384, 512),
                             offset="(2, -5, 0)", size=((10, 13, 9), (10, 13, 11)), opacity=0.5, label='2-3'),

    # CL3
    *block_ConvInceptionPool(name='cl3', bottom='cl2-pool', top='cl3-pool', s_filer="25x20", n_filer=(768, 1024),
                             offset="(2, -5, 0)", size=((5, 7, 12), (5, 7, 14)), label='3-4'),

    # CL4
    *block_ConvInception(name='cl4', bottom='cl3-pool', s_filer="13x10", n_filer=(1280, 1536),
                         offset="(2, -5, 0)", size=((3, 4, 15), (3, 4, 17)), label='4-5'),

    # #Bottleneck
    # #block-005
    # to_ConvConvRelu( name='ccr_b5', s_filer=32, n_filer=(1024,1024), offset="(2,0,0)", to="(cl3-pool-east)", width=(8,8), height=8, depth=8, caption="Bottleneck"  ),
    # to_connection("cl3-pool", "ccr_b5"),

    # Decoder
    # EL3
    *block_UnconvSkipConvInception(name="el3", bottom="cl4-i", s_filer="25x20", n_filer=(2048, 768, 768),
                                   offset="(2, 5, 0)",
                                   size=((5, 7, 28), (5, 7, 12), (5, 7, 12)), label=("3-1", "3-6", "3-6")),
    to_skip(of='cl3-i', to='el3-res', pos=5),

    # EL2
    *block_UnconvSkipConvInception(name="el2", bottom="el3-i", s_filer="50x40", n_filer=(1024, 384, 384),
                                   offset="(3, 5, 0)",
                                   size=((10, 13, 14), (10, 13, 9), (10, 13, 9)), label=("2-2", "2-7", "2-7")),
    to_skip(of='cl2-i', to='el2-res', pos=4),

    # EL1
    *block_UnconvSkipConvInception(name="el1", bottom="el2-i", s_filer="100x80", n_filer=(512, 128, 128),
                                   offset="(4, 5, 0)",
                                   size=((20, 25, 11), (20, 25, 6), (20, 25, 6)), label=("1-3", "1-8", "1-8")),
    to_skip(of='cl1-i', to='el1-res', pos=3),

    # EL0
    *block_UnconvSkipConv(name="el0", bottom="el1-i", s_filer="200x160", n_filer=(256, 64),
                          offset="(5, 5, 0)",
                          size=((40, 50, 10), (40, 50, 3)), label=("0-4", "0-9")),
    to_skip(of='cl0-i', to='el0-res', pos=2),
    to_Conv(name='el0-c2', s_filer="200x160", n_filer=3, offset="(6,0,0)", to="(el0-c-east)",
            width=1, height=40, depth=50, caption='C0-10'),
    to_connection(
        "el0-c",
        "el0-c2"
    ),

    # *block_Unconv(name="b7", bottom="end_b6", top='end_b7', s_filer=128, n_filer=256, offset="(2.1,5,0)", size=(25, 25, 4.5), opacity=0.5),
    # to_skip( of='cl2-i', to='ccr_res_b7', pos=1.25),
    # *block_Unconv(name="b8", bottom="end_b7", top='end_b8', s_filer=256, n_filer=128, offset="(2.1,5,0)", size=(32, 32, 3.5), opacity=0.5),
    # to_skip(of='cl1-i', to='ccr_res_b8', pos=1.25),
    #
    # *block_Unconv(name="b9", bottom="end_b8", top='end_b9', s_filer=512, n_filer=64, offset="(2.1,5,0)", size=(40, 40, 2.5), opacity=0.5),
    # to_skip(of='cl0-i', to='ccr_res_b9', pos=1.25),
    #
    # to_ConvSoftMax( name="soft1", s_filer=512, offset="(0.75,0,0)", to="(end_b9-east)", width=1, height=40, depth=40, caption="SOFT" ),
    # to_connection( "end_b9", "soft1"),

    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')


if __name__ == '__main__':
    main()
