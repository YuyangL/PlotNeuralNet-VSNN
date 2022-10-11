import sys

sys.path.append('../')
from pycore.blocks import *

# User inputs
cl0, cl1, cl2, cl3, bottleneck, el3, el2, el1, el0 = {}, {}, {}, {}, {}, {}, {}, {}, {}
# Contraction
cl0['conv1'] = 32
cl0['conv2'] = 32
cl0['conv3'] = 32
cl1['conv1'] = 32
cl1['conv2'] = 64
cl2['conv1'] = 80
cl2['conv2'] = 192
cl3['incept1'] = 256
cl3['incept2'] = 288
cl3['incept3'] = 288
# Bottleneck
bottleneck['incept1'] = 768  # Deep variant
bottleneck['incept2'] = 768
bottleneck['incept3'] = 768
bottleneck['incept4'] = 768
# Expansion
el3['incept1'] = 288
el3['incept2'] = 288
el3['incept3'] = 256
el2['conv1'] = 192
el2['conv2'] = 80
el1['conv1'] = 64
el1['conv2'] = 32
el0['conv1'] = 32
el0['conv2'] = 32
el0['conv 1x1'] = 2

base_width = 4

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # Input
    to_input('../vsnn/vsnn_v1_input.png', width=16, height=8, name='input'),

    # Encoder
    # CL0
    to_Conv(name='cl0-c1', s_filer="", n_filer=cl0['conv1'], offset="(1,0,0)", to="(0,0,0)",
            width=base_width, height=40, depth=80, caption='C0-1'),
    to_Conv(name='cl0-c2', s_filer="", n_filer=cl0['conv2'], offset="(0,0,0)", to="(cl0-c1-east)",
            width=base_width, height=40, depth=80, caption='C0-2'),
    to_Conv(name='cl0-c3', s_filer="319x159", n_filer=cl0['conv3'], offset="(0,0,0)", to="(cl0-c2-east)",
            width=base_width, height=40, depth=80, caption='C0-3'),

    # CL1
    to_Conv(name='cl1-c1', s_filer="", n_filer=cl1['conv1'], offset="(3, -5, 0)", to="(cl0-c2-east)",
            width=base_width, height=20, depth=40, caption='C1-4'),
    to_Conv(name='cl1-c2', s_filer="159x79", n_filer=cl1['conv2'], offset="(0,0,0)", to="(cl1-c1-east)",
            width=base_width + 1, height=20, depth=40, caption='C1-5'),
    to_Pool(name="cl1-pool", offset="(0,0,0)", to="(cl1-c2-east)", width=1, height=10, depth=20, opacity=0.5),
    to_connection(
        "{}".format("cl0-c2"),
        "{}".format("cl1-c1")),

    # CL2
    to_Conv(name='cl2-c1', s_filer="", n_filer=cl2['conv1'], offset="(2, -5, 0)", to="(cl1-pool-east)",
            width=base_width + 2, height=10, depth=20, caption='C2-6'),
    to_Conv(name='cl2-c2', s_filer="79x39", n_filer=cl2['conv2'], offset="(0,0,0)", to="(cl2-c1-east)",
            width=base_width + 3, height=10, depth=20, caption='C2-7'),
    to_Pool(name="cl2-pool", offset="(0,0,0)", to="(cl2-c2-east)", width=1, height=5, depth=10, opacity=0.5),
    to_connection(
        "{}".format("cl1-pool"),
        "{}".format("cl2-c1")),

    # CL3
    *block_Inceptionx3Pool(name='cl3', bottom='cl2-pool', top='cl3-pool', s_filer="39x19",
                           n_filer=(cl3['incept1'], cl3['incept2'], cl3['incept3']),
                           offset="(2, -5, 0)",
                           size=((5, 10, base_width + 4), (5, 10, base_width + 5), (5, 10, base_width + 5)),
                           label=('3-1', '3-2', '3-3')),

    # Bottleneck
    *block_Inceptionx4(name='cl4', bottom='cl3-pool', s_filer="19x9",
                       n_filer=(
                           bottleneck['incept1'], bottleneck['incept2'], bottleneck['incept3'], bottleneck['incept4']),
                       offset="(2, -5, 0)",
                       size=((2.5, 5, base_width + 7), (2.5, 5, base_width + 7),
                             (2.5, 5, base_width + 7), (2.5, 5, base_width + 7)),
                       label=('4-4', '4-5', '4-6', '4-7')),

    # Decoder
    # EL3
    *block_UnconvSkipInceptionx3(name="el3", bottom="cl4-i4", s_filer="39x19",
                                 n_filer=(
                                     cl3['incept3'], cl3['incept3'] * 2, el3['incept1'], el3['incept2'],
                                     el3['incept3']),
                                 offset="(2, 5, 0)",
                                 size=((5, 10, base_width + 5), (5, 10, 2*(base_width + 5)),
                                       (5, 10, base_width + 5), (5, 10, base_width + 5), (5, 10, base_width + 4)),
                                 label=("3-1", "3-1", "3-8", "3-9", "3-10")),
    to_skip(of='cl3-i3', to='el3-skip', pos=5),

    # EL2
    *block_UnconvSkipConvx2(name="el2", bottom="el3-i3", s_filer="79x39",
                            n_filer=(cl2['conv2'], cl2['conv2'] * 2, el2['conv1'], el2['conv2']),
                            offset="(3, 5, 0)",
                            size=((10, 20, base_width + 3), (10, 20, 2*(base_width + 3)),
                                  (10, 20, base_width + 3), (10, 20, base_width + 2)),
                            label=("2-2", "2-2", "2-8", "2-9")),
    to_skip(of='cl2-c2', to='el2-skip', pos=4),

    # EL1
    *block_UnconvSkipConvx2(name="el1", bottom="el2-c2", s_filer="159x79",
                            n_filer=(cl1['conv2'], cl1['conv2'] * 2, el1['conv1'], el1['conv2']),
                            offset="(4, 5, 0)",
                            size=((20, 40, base_width + 1), (20, 40, 2*(base_width + 1)),
                                  (20, 40, base_width + 1), (20, 40, base_width)),
                            label=("1-3", "1-3", "1-10", "1-11")),
    to_skip(of='cl1-c2', to='el1-skip', pos=3),

    # EL0
    *block_UnconvSkipConvx2(name="el0", bottom="el1-c2", s_filer="319x159",
                            n_filer=(cl0['conv1'], cl0['conv1'] * 2, el0['conv1'], el0['conv2']),
                            offset="(5, 5, 0)",
                            size=((40, 80, base_width), (40, 80, 2 * base_width),
                                  (40, 80, base_width), (40, 80, base_width)),
                            label=("0-4", "0-4", "0-12", '0-13')),
    to_skip(of='cl0-c2', to='el0-skip', pos=2),
    to_Conv(name='el0-c3', s_filer="319x159", n_filer=el0['conv 1x1'], offset="(6,0,0)", to="(el0-c2-east)",
            width=1, height=40, depth=80, caption='C0-14'),
    to_connection(
        "el0-c2",
        "el0-c3"
    ),

    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')


if __name__ == '__main__':
    main()
