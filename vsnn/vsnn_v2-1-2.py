import sys

sys.path.append('../')
from pycore.blocks import *

# User inputs
cl0, cl1, cl2, cl3, cl4, bottleneck, el4, el3, el2, el1, el0 = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
# Contraction
cl0['conv'] = 32
cl0['incept'] = 144
cl1['conv'] = 160
cl1['incept'] = 288
cl2['conv'] = 304
cl2['incept'] = 432
cl3['conv'] = 448
cl3['incept'] = 720
cl4['conv'] = 736
cl4['incept'] = 1024
bottleneck['conv'] = 1280
bottleneck['incept'] = 1536
el4['conv'] = 736
el4['incept'] = 720
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
    to_Conv(name='cl0-c', s_filer="", n_filer=cl0['conv'], offset="(1,0,0)", to="(0,0,0)",
            width=3, height=40, depth=80, caption='C0-1'),
    to_Inception(name='cl0-i', s_filer="320x160", n_filer=cl0['incept'], offset="(0,0,0)", to="(cl0-c-east)",
                 width=5, height=40, depth=80, caption="I0-1"),
    to_Pool(name="cl0-pool", offset="(0,0,0)", to="(cl0-i-east)", width=1, height=20, depth=40, opacity=0.5),

    # CL1
    *block_ConvInceptionPool(name='cl1', bottom='cl0-pool', top='cl1-pool', s_filer="160x80",
                             n_filer=(cl1['conv'], cl1['incept']),
                             offset="(2, -5, 0)", size=((20, 40, 6), (20, 40, 8)), opacity=0.5, label='1-2'),

    # CL2
    *block_ConvInceptionPool(name='cl2', bottom='cl1-pool', top='cl2-pool', s_filer="80x40",
                             n_filer=(cl2['conv'], cl2['incept']),
                             offset="(2, -5, 0)", size=((10, 20, 9), (10, 20, 11)), opacity=0.5, label='2-3'),

    # CL3
    *block_ConvInceptionPool(name='cl3', bottom='cl2-pool', top='cl3-pool', s_filer="40x20",
                             n_filer=(cl3['conv'], cl3['incept']),
                             offset="(2, -5, 0)", size=((5, 10, 12), (5, 10, 14)), label='3-4'),

    # CL4
    *block_ConvInceptionPool(name='cl4', bottom='cl3-pool', top='cl4-pool', s_filer="20x10",
                             n_filer=(cl4['conv'], cl4['incept']),
                             offset="(2, -5, 0)", size=((2.5, 5, 15), (2.5, 5, 17)), label=('4-5', '4-5.1')),

    # Bottleneck
    *block_ConvInception(name='cl5', bottom='cl4-pool', s_filer="10x5",
                         n_filer=(bottleneck['conv'], bottleneck['incept']),
                         offset="(2, -5, 0)", size=((1.25, 2.5, 18), (1.25, 2.5, 20)), label=('5-9', '5-10')),

    # Decoder
    # EL4
    *block_UnconvSkipConvInception(name="el4", bottom="cl5-i", s_filer="20x10",
                                   n_filer=(cl4['incept'], cl4['incept'] * 2, el4['conv'], el4['incept']),
                                   offset="(2, 5, 0)",
                                   size=((2.5, 5, 17), (2.5, 5, 34), (2.5, 5, 15), (2.5, 5, 14)),
                                   label=("4-5", "4-5", "4-10", "4-11")),
    to_skip(of='cl4-i', to='el4-skip', pos=5),

    # EL3
    *block_UnconvSkipConvInception(name="el3", bottom="el4-i", s_filer="40x20",
                                   n_filer=(cl3['incept'], cl3['incept']*2, el3['conv'], el3['incept']),
                                   offset="(3, 5, 0)",
                                   size=((5, 10, 14), (5, 10, 28), (5, 10, 12), (5, 10, 11)),
                                   label=("3-1", "3-1", "3-6", "3-6")),
    to_skip(of='cl3-i', to='el3-skip', pos=5),

    # EL2
    *block_UnconvSkipConvInception(name="el2", bottom="el3-i", s_filer="80x40",
                                   n_filer=(cl2['incept'], cl2['incept']*2, el2['conv'], el2['incept']),
                                   offset="(4, 5, 0)",
                                   size=((10, 20, 11), (10, 20, 22), (10, 20, 9), (10, 20, 8)),
                                   label=("2-2", "2-2", "2-7", "2-7")),
    to_skip(of='cl2-i', to='el2-skip', pos=4),

    # EL1
    *block_UnconvSkipConvInception(name="el1", bottom="el2-i", s_filer="160x80",
                                   n_filer=(cl1['incept'], cl1['incept']*2, el1['conv'], el1['incept']),
                                   offset="(5, 5, 0)",
                                   size=((20, 40, 8), (20, 40, 16), (20, 40, 6), (20, 40, 5)),
                                   label=("1-3", "1-3", "1-8", "1-8")),
    to_skip(of='cl1-i', to='el1-skip', pos=3),

    # EL0
    *block_UnconvSkipConv(name="el0", bottom="el1-i", s_filer="320x160",
                          n_filer=(cl0['incept'], cl0['incept']*2, el0['conv']),
                          offset="(6, 5, 0)",
                          size=((40, 80, 5), (40, 80, 10), (40, 80, 3)), label=("0-4", "0-4", "0-9")),
    to_skip(of='cl0-i', to='el0-skip', pos=2),
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
