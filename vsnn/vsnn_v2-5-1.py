import sys

sys.path.append('../')
from pycore.blocks import *

# User inputs
cl0, cl1, cl2, cl3, bottleneck, el3, el2, el1, el0 = {}, {}, {}, {}, {}, {}, {}, {}, {}
# Contraction
cl0['conv'] = 32
cl0['incept'] = 144
cl1['conv1'] = 160
cl1['conv2'] = 288
cl2['conv1'] = 304
cl2['conv2'] = 432
cl3['conv'] = 448
cl3['incept'] = 720
bottleneck['conv'] = 736
bottleneck['incept'] = 1024
el3['conv'] = 448
el3['incept'] = 432
el2['conv'] = 304
el2['incept'] = 288
el1['conv'] = 160
el1['incept'] = 144
el0['conv'] = 32
el0['conv 1x1'] = 4

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # Input
    to_input('../vsnn/vsnn_v1_input.png', width=16, height=8, name='input'),

    # Encoder
    # CL0
    to_Conv(name='cl0-c', s_filer="320x160", n_filer=cl0['conv'], offset="(1,0,0)", to="(0,0,0)",
            width=3, height=40, depth=80, caption='C0-1'),

    # CL1
    to_Conv(name='cl1-c1', s_filer="", n_filer=cl1['conv1'], offset="(2, -5, 0)", to="(0,0,0)",
            width=3, height=20, depth=40, caption='C1-2'),
    to_Conv(name='cl1-c2', s_filer="160x80", n_filer=cl1['conv2'], offset="(0,0,0)", to="(0,0,0)",
            width=4, height=20, depth=40, caption='C1-3'),
    to_Pool(name="cl1-pool", offset="(0,0,0)", to="(cl1-c2-east)", width=1, height=10, depth=20, opacity=0.5),
    to_connection(
        "{}".format("cl0-c"),
        "{}".format("cl1-c1")),

    # CL2
    to_Conv(name='cl2-c1', s_filer="80x40", n_filer=cl2['conv1'], offset="(2, -5, 0)", to="(0,0,0)",
            width=4, height=10, depth=20, caption='C2-4'),
    to_Conv(name='cl2-c2', s_filer="", n_filer=cl2['conv2'], offset="(0,0,0)", to="(0,0,0)",
            width=5, height=5, depth=10, caption='C2-5'),
    to_connection(
        "{}".format("cl1-pool"),
        "{}".format("cl2-c1")),

    # CL3
    *block_ConvInceptionPool(name='cl3', bottom='cl2-c2', top='cl3-pool', s_filer="40x20",
                             n_filer=(cl3['conv'], cl3['incept']),
                             offset="(2, -5, 0)", size=((5, 10, 7), (5, 10, 9)), label=('3-6', '3-1')),

    # Bottleneck
    *block_ConvInception(name='cl4', bottom='cl3-pool', s_filer="20x10",
                         n_filer=(bottleneck['conv'], bottleneck['incept']),
                         offset="(2, -5, 0)", size=((2.5, 5, 11), (2.5, 5, 13)), label='4-2'),

    # Decoder
    # EL3
    *block_UnconvSkipConvInception(name="el3", bottom="cl4-i", s_filer="40x20",
                                   n_filer=(cl3['incept'], cl3['incept']*2, el3['conv'], el3['incept']),
                                   offset="(2, 5, 0)",
                                   size=((5, 10, 9), (5, 10, 18), (5, 10, 9), (5, 10, 7)),
                                   label=("3-1", "3-1", "3-6", "3-6")),
    to_skip(of='cl3-i', to='el3-skip', pos=5),

    # EL2
    # TransposeConv
    to_UnPool(name='{}-unpool'.format("el2"), offset="(3, 5, 0)", to="({}-east)".format("el3-i"), width=4,
              height=10,
              depth=20, opacity=.5, caption="TC{}".format("2-2"), n_filer=str(cl2['conv1'])),
    # Skip concat
    to_ConvRes(name='{}-skip'.format("el2"), offset="(0,0,0)", to="({}-unpool-east)".format("el2"),
               s_filer="", n_filer=str(cl2['conv1'] * 2), width=8, height=10, depth=20,
               opacity=.5, caption="SC{}".format("2-2")),
    to_Conv(name='{}-c'.format("el2"), offset="(0,0,0)", to="({}-skip-east)".format("el2"), s_filer="80x40",
            n_filer=str(el2['conv']), width=4, height=10, depth=20,
            caption="C{}".format("2-7")),
    to_connection(
        "{}".format("el3-i"),
        "{}".format("el2-unpool")),
    to_skip(of='cl2-c1', to='el2-skip', pos=4),

    # EL1
    # TransposeConv
    to_UnPool(name='{}-unpool'.format("el1"), offset="(4, 5, 0)", to="({}-east)".format("el2-c"), width=40,
              height=20,
              depth=3, opacity=.5, caption="TC{}".format("1-3"), n_filer=str(cl1['conv2'])),
    # Skip concat
    to_ConvRes(name='{}-skip'.format("el1"), offset="(0,0,0)", to="({}-unpool-east)".format("el1"),
               s_filer="", n_filer=str(cl1['conv2'] * 2), width=20, height=10, depth=6,
               opacity=.5, caption="SC{}".format("1-3")),
    to_Conv(name='{}-c'.format("el1"), offset="(0,0,0)", to="({}-skip-east)".format("el1"), s_filer="160x80",
            n_filer=str(el2['conv']), width=40, height=20, depth=3,
            caption="C{}".format("1-8")),
    to_connection(
        "{}".format("el1-c"),
        "{}".format("el2-unpool")),
    to_skip(of='cl1-c2', to='el1-skip', pos=3),

    # EL0
    *block_UnconvSkipConv(name="el0", bottom="el1-c", s_filer="320x160",
                          n_filer=(cl0['conv'], cl0['conv']*2, el0['conv']),
                          offset="(5, 5, 0)",
                          size=((40, 80, 5), (40, 80, 10), (40, 80, 3)), label=("0-4", "0-4", "0-9")),
    to_skip(of='cl0-c', to='el0-skip', pos=2),
    to_Conv(name='el0-c2', s_filer="320x160", n_filer=el0['conv 1x1'], offset="(6,0,0)", to="(el0-c-east)",
            width=1, height=40, depth=80, caption='C0-10'),
    to_connection(
        "el0-c",
        "el0-c2"
    ),

    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')


if __name__ == '__main__':
    main()
