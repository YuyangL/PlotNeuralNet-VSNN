from .tikzeng import *


# define new block
def block_2ConvPool(name, bottom, top, s_filer=256, n_filer=64, offset="(1,0,0)", size=(32, 32, 3.5), opacity=0.5):
    return [
        to_ConvConvRelu(
            name="ccr_{}".format(name),
            s_filer=str(s_filer),
            n_filer=(n_filer, n_filer),
            offset=offset,
            to="({}-east)".format(bottom),
            width=(size[2], size[2]),
            height=size[0],
            depth=size[1],
        ),
        to_Pool(
            name="{}".format(top),
            offset="(0,0,0)",
            to="(ccr_{}-east)".format(name),
            width=1,
            height=size[0] - int(size[0] / 4),
            depth=size[1] - int(size[0] / 4),
            opacity=opacity, ),
        to_connection(
            "{}".format(bottom),
            "ccr_{}".format(name)
        )
    ]


def block_Unconv(name, bottom, top, s_filer=256, n_filer=64, offset="(1,0,0)", size=(32, 32, 3.5), opacity=0.5):
    return [
        to_UnPool(name='unpool_{}'.format(name), offset=offset, to="({}-east)".format(bottom), width=1, height=size[0],
                  depth=size[1], opacity=opacity),
        to_ConvRes(name='ccr_res_{}'.format(name), offset="(0,0,0)", to="(unpool_{}-east)".format(name),
                   s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1],
                   opacity=opacity),
        to_Conv(name='ccr_{}'.format(name), offset="(0,0,0)", to="(ccr_res_{}-east)".format(name), s_filer=str(s_filer),
                n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1]),
        to_ConvRes(name='ccr_res_c_{}'.format(name), offset="(0,0,0)", to="(ccr_{}-east)".format(name),
                   s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1],
                   opacity=opacity),
        to_Conv(name='{}'.format(top), offset="(0,0,0)", to="(ccr_res_c_{}-east)".format(name), s_filer=str(s_filer),
                n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1]),
        to_connection(
            "{}".format(bottom),
            "unpool_{}".format(name)
        )
    ]


def block_Res(num, name, botton, top, s_filer=256, n_filer=64, offset="(0,0,0)", size=(32, 32, 3.5), opacity=0.5):
    lys = []
    layers = [*['{}_{}'.format(name, i) for i in range(num - 1)], top]
    for name in layers:
        ly = [to_Conv(
            name='{}'.format(name),
            offset=offset,
            to="({}-east)".format(botton),
            s_filer=str(s_filer),
            n_filer=str(n_filer),
            width=size[2],
            height=size[0],
            depth=size[1]
        ),
            to_connection(
                "{}".format(botton),
                "{}".format(name)
            )
        ]
        botton = name
        lys += ly

    lys += [
        to_skip(of=layers[1], to=layers[-2], pos=1.25),
    ]
    return lys


def block_ConvReluPool(name, bottom, top, s_filer=256, n_filer=64, offset="(1,0,0)", size=(32, 32, 3.5), opacity=0.5):
    return [
        to_ConvRelu(
            name="ccr_{}".format(name),
            s_filer=str(s_filer),
            n_filer=n_filer,
            offset=offset,
            to="({}-east)".format(bottom),
            width=size[2],
            height=size[0],
            depth=size[1],
        ),
        to_Pool(
            name="{}".format(top),
            offset="(0,0,0)",
            to="(ccr_{}-east)".format(name),
            width=1,
            height=size[0] - int(size[0] / 4),
            depth=size[1] - int(size[0] / 4),
            opacity=opacity, ),
        to_connection(
            "{}".format(bottom),
            "ccr_{}".format(name)
        )
    ]


def block_ConvInceptionPool(name, bottom, top, s_filer=256, n_filer=(64, 128), offset="(1,0,0)",
                            size=((32, 32, 3.5), (32, 32, 5)), opacity=0.5, label='1'):
    if isinstance(label, str):
        label = (label,) * 2

    return [
        to_Conv(
            name="{}-c".format(name),
            s_filer="",
            n_filer=n_filer[0],
            offset=offset,
            to="({}-east)".format(bottom),
            width=size[0][2],
            height=size[0][0],
            depth=size[0][1],
            caption="C{}".format(label[0])
        ),
        to_Inception(
            name="{}-i".format(name),
            s_filer=str(s_filer),
            n_filer=n_filer[1],
            to="({}-c-east)".format(name),
            width=size[1][2],
            height=size[0][0],
            depth=size[0][1],
            caption="I{}".format(label[1])
        ),
        to_Pool(
            name="{}".format(top),
            offset="(0,0,0)",
            to="({}-i-east)".format(name),
            width=1,
            height=size[1][0] - int(size[1][0] / 2),
            depth=size[1][1] - int(size[1][0] / 2),
            opacity=opacity,
        ),
        to_connection(
            "{}".format(bottom),
            "{}-c".format(name)
        )
    ]


def block_ConvInception(name, bottom, s_filer=256, n_filer=(64, 128), offset="(1,0,0)",
                        size=((32, 32, 3.5), (32, 32, 5)), label='1'):
    if isinstance(label, str):
        label = (label,) * 2

    return [
        to_Conv(
            name="{}-c".format(name),
            s_filer="",
            n_filer=n_filer[0],
            offset=offset,
            to="({}-east)".format(bottom),
            width=size[0][2],
            height=size[0][0],
            depth=size[0][1],
            caption="C{}".format(label[0])
        ),
        to_Inception(
            name="{}-i".format(name),
            s_filer=str(s_filer),
            n_filer=n_filer[1],
            to="({}-c-east)".format(name),
            width=size[1][2],
            height=size[0][0],
            depth=size[0][1],
            caption="I{}".format(label[1])
        ),
        to_connection(
            "{}".format(bottom),
            "{}-c".format(name)
        )
    ]


def block_Inceptionx3Pool(name, bottom, top, s_filer=256, n_filer=(64, 128, 128), offset="(1,0,0)",
                          size=((32, 32, 3.5), (32, 32, 5), (32, 32, 5)), label='1', pool_width=1,
                          gap_pool=False):
    if isinstance(label, str):
        label = (label,) * 99

    return [
        to_Inception(
            name="{}-i1".format(name),
            s_filer="",
            n_filer=n_filer[0],
            offset=offset,
            to="({}-east)".format(bottom),
            width=size[0][2],
            height=size[0][0],
            depth=size[0][1],
            caption="I{}".format(label[0])
        ),
        to_Inception(
            name="{}-i2".format(name),
            s_filer="",
            n_filer=n_filer[1],
            to="({}-i1-east)".format(name),
            width=size[1][2],
            height=size[1][0],
            depth=size[1][1],
            caption="I{}".format(label[1])
        ),
        to_Inception(
            name="{}-i3".format(name),
            s_filer=str(s_filer),
            n_filer=n_filer[2],
            to="({}-i2-east)".format(name),
            width=size[2][2],
            height=size[2][0],
            depth=size[2][1],
            caption="I{}".format(label[2])
        ),
        to_Pool(
            name="{}".format(top),
            offset="(0,0,0)",
            to="({}-i3-east)".format(name),
            width=pool_width,
            height=size[2][0] - int(size[2][0] / 2) if not gap_pool else 2,
            depth=size[2][1] - int(size[2][0] / 2) if not gap_pool else 2,
            opacity=.5,
        ),
        to_connection(
            "{}".format(bottom),
            "{}-i1".format(name)
        )
    ]


def block_Inceptionx4Pool(name, bottom, top, s_filer=256, n_filer=(64, 128, 128, 128), offset="(1,0,0)",
                          size=((32, 32, 3.5), (32, 32, 5), (32, 32, 5), (32, 32, 5)), label='1', pool_width=1,
                          gap_pool=False):
    if isinstance(label, str):
        label = (label,) * 99

    return [
        to_Inception(
            name="{}-i1".format(name),
            s_filer="",
            n_filer=n_filer[0],
            offset=offset,
            to="({}-east)".format(bottom),
            width=size[0][2],
            height=size[0][0],
            depth=size[0][1],
            caption="I{}".format(label[0])
        ),
        to_Inception(
            name="{}-i2".format(name),
            s_filer="",
            n_filer=n_filer[1],
            to="({}-i1-east)".format(name),
            width=size[1][2],
            height=size[1][0],
            depth=size[1][1],
            caption="I{}".format(label[1])
        ),
        to_Inception(
            name="{}-i3".format(name),
            s_filer="",
            n_filer=n_filer[2],
            to="({}-i2-east)".format(name),
            width=size[2][2],
            height=size[2][0],
            depth=size[2][1],
            caption="I{}".format(label[2])
        ),
        to_Inception(
            name="{}-i4".format(name),
            s_filer=str(s_filer),
            n_filer=n_filer[3],
            to="({}-i3-east)".format(name),
            width=size[3][2],
            height=size[3][0],
            depth=size[3][1],
            caption="I{}".format(label[3])
        ),
        to_Pool(
            name="{}".format(top),
            offset="(0,0,0)",
            to="({}-i4-east)".format(name),
            width=pool_width,
            height=size[3][0] - int(size[3][0] / 2) if not gap_pool else 2,
            depth=size[3][1] - int(size[3][0] / 2) if not gap_pool else 2,
            opacity=.5,
        ),
        to_connection(
            "{}".format(bottom),
            "{}-i1".format(name)
        )
    ]


def block_Inceptionx2(name, bottom, s_filer=256, n_filer=(64, 128), offset="(1,0,0)",
                      size=((32, 32, 3.5), (32, 32, 5)),
                      label='1'):
    if isinstance(label, str):
        label = (label,) * 99

    return [
        to_Inception(
            name="{}-i1".format(name),
            s_filer="",
            n_filer=n_filer[0],
            offset=offset,
            to="({}-east)".format(bottom),
            width=size[0][2],
            height=size[0][0],
            depth=size[0][1],
            caption="C{}".format(label[0])
        ),
        to_Inception(
            name="{}-i2".format(name),
            s_filer=str(s_filer),
            n_filer=n_filer[1],
            to="({}-i1-east)".format(name),
            width=size[1][2],
            height=size[1][0],
            depth=size[1][1],
            caption="I{}".format(label[1])
        ),
        to_connection(
            "{}".format(bottom),
            "{}-i1".format(name)
        )
    ]


def block_Inceptionx4(name, bottom, s_filer=256, n_filer=(64, 128, 128, 128), offset="(1,0,0)",
                      size=((32, 32, 3.5), (32, 32, 5), (32, 32, 5), (32, 32, 5)),
                      label='1'):
    if isinstance(label, str):
        label = (label,) * 99

    return [
        to_Inception(
            name="{}-i1".format(name),
            s_filer="",
            n_filer=n_filer[0],
            offset=offset,
            to="({}-east)".format(bottom),
            width=size[0][2],
            height=size[0][0],
            depth=size[0][1],
            caption="I{}".format(label[0])
        ),
        to_Inception(
            name="{}-i2".format(name),
            s_filer="",
            n_filer=n_filer[1],
            to="({}-i1-east)".format(name),
            width=size[1][2],
            height=size[1][0],
            depth=size[1][1],
            caption="I{}".format(label[1])
        ),
        to_Inception(
            name="{}-i3".format(name),
            s_filer="",
            n_filer=n_filer[2],
            to="({}-i2-east)".format(name),
            width=size[2][2],
            height=size[2][0],
            depth=size[2][1],
            caption="I{}".format(label[2])
        ),
        to_Inception(
            name="{}-i4".format(name),
            s_filer=str(s_filer),
            n_filer=n_filer[3],
            to="({}-i3-east)".format(name),
            width=size[3][2],
            height=size[3][0],
            depth=size[3][1],
            caption="I{}".format(label[3])
        ),
        to_connection(
            "{}".format(bottom),
            "{}-i1".format(name)
        )
    ]


def block_Inceptionx5(name, bottom, s_filer=256, n_filer=(64, 128, 128, 128, 128), offset="(1,0,0)",
                      size=((32, 32, 3.5), (32, 32, 5), (32, 32, 5), (32, 32, 5), (32, 32, 5)),
                      label='1'):
    if isinstance(label, str):
        label = (label,) * 99

    return [
        to_Inception(
            name="{}-i1".format(name),
            s_filer="",
            n_filer=n_filer[0],
            offset=offset,
            to="({}-east)".format(bottom),
            width=size[0][2],
            height=size[0][0],
            depth=size[0][1],
            caption="I{}".format(label[0])
        ),
        to_Inception(
            name="{}-i2".format(name),
            s_filer="",
            n_filer=n_filer[1],
            to="({}-i1-east)".format(name),
            width=size[1][2],
            height=size[1][0],
            depth=size[1][1],
            caption="I{}".format(label[1])
        ),
        to_Inception(
            name="{}-i3".format(name),
            s_filer="",
            n_filer=n_filer[2],
            to="({}-i2-east)".format(name),
            width=size[2][2],
            height=size[2][0],
            depth=size[2][1],
            caption="I{}".format(label[2])
        ),
        to_Inception(
            name="{}-i4".format(name),
            s_filer="",
            n_filer=n_filer[3],
            to="({}-i3-east)".format(name),
            width=size[3][2],
            height=size[3][0],
            depth=size[3][1],
            caption="I{}".format(label[3])
        ),
        to_Inception(
            name="{}-i5".format(name),
            s_filer=str(s_filer),
            n_filer=n_filer[4],
            to="({}-i4-east)".format(name),
            width=size[4][2],
            height=size[4][0],
            depth=size[4][1],
            caption="I{}".format(label[4])
        ),
        to_connection(
            "{}".format(bottom),
            "{}-i1".format(name)
        )
    ]


def block_UnconvSkipConvInception(name, bottom, s_filer=256, n_filer=(64, 64, 64), offset="(1,0,0)",
                                  size=((32, 32, 5), (32, 32, 3.5), (32, 32, 3.5), (32, 32, 3.5)),
                                  opacity=0.5, label=('1', '1', '1')):
    return [
        to_UnPool(name='{}-unpool'.format(name), offset=offset, to="({}-east)".format(bottom), width=size[0][2],
                  height=size[0][0],
                  depth=size[0][1], opacity=opacity, caption="TC{}".format(label[0]), n_filer=str(n_filer[0])),
        to_ConvRes(name='{}-skip'.format(name), offset="(0,0,0)", to="({}-unpool-east)".format(name),
                   s_filer="", n_filer=str(n_filer[1]), width=size[1][2], height=size[1][0], depth=size[1][1],
                   opacity=opacity, caption="SC{}".format(label[1])),
        to_Conv(name='{}-c'.format(name), offset="(0,0,0)", to="({}-skip-east)".format(name), s_filer="",
                n_filer=str(n_filer[2]), width=size[2][2], height=size[2][0], depth=size[2][1],
                caption="C{}".format(label[2])),
        to_Inception(
            name="{}-i".format(name),
            s_filer=str(s_filer),
            n_filer=n_filer[3],
            to="({}-c-east)".format(name),
            width=size[3][2],
            height=size[3][0],
            depth=size[3][1],
            caption="I{}".format(label[3])
        ),
        to_connection(
            "{}".format(bottom),
            "{}-unpool".format(name)
        )
    ]


def block_UnconvSkipInceptionx3(name, bottom, s_filer=256, n_filer=(64, 64, 64, 64, 64), offset="(1,0,0)",
                                size=((32, 32, 5), (32, 32, 3.5), (32, 32, 3.5), (32, 32, 3.5), (32, 32, 3.5)),
                                opacity=0.5, label=('1', '1', '1', '1', '1')):
    return [
        to_UnPool(name='{}-unpool'.format(name), offset=offset, to="({}-east)".format(bottom), width=size[0][2],
                  height=size[0][0],
                  depth=size[0][1], opacity=opacity, caption="TC{}".format(label[0]), n_filer=str(n_filer[0])),
        to_ConvRes(name='{}-skip'.format(name), offset="(0,0,0)", to="({}-unpool-east)".format(name),
                   s_filer="", n_filer=str(n_filer[1]), width=size[1][2], height=size[1][0], depth=size[1][1],
                   opacity=opacity, caption="SC{}".format(label[1])),
        to_Inception(
            name="{}-i1".format(name),
            s_filer="",
            n_filer=n_filer[2],
            to="({}-skip-east)".format(name),
            width=size[2][2],
            height=size[2][0],
            depth=size[2][1],
            caption="I{}".format(label[2])
        ),
        to_Inception(
            name="{}-i2".format(name),
            s_filer="",
            n_filer=n_filer[3],
            to="({}-i1-east)".format(name),
            width=size[3][2],
            height=size[3][0],
            depth=size[3][1],
            caption="I{}".format(label[3])
        ),
        to_Inception(
            name="{}-i3".format(name),
            s_filer=str(s_filer),
            n_filer=n_filer[4],
            to="({}-i2-east)".format(name),
            width=size[4][2],
            height=size[4][0],
            depth=size[4][1],
            caption="I{}".format(label[4])
        ),
        to_connection(
            "{}".format(bottom),
            "{}-unpool".format(name)
        )
    ]


def block_ConvSkipInceptionx3(name, bottom, s_filer=256, n_filer=(64, 64, 64, 64, 64), offset="(1,0,0)",
                              size=((32, 32, 5), (32, 32, 3.5), (32, 32, 3.5), (32, 32, 3.5), (32, 32, 3.5)),
                              opacity=0.5, label=('1', '1', '1', '1', '1')):
    return [
        to_Conv(name='{}-c'.format(name), offset=offset, to="({}-east)".format(bottom), s_filer="",
                n_filer=str(n_filer[0]), width=size[0][2], height=size[0][0], depth=size[0][1],
                caption="C{}".format(label[0])),
        to_ConvRes(name='{}-skip'.format(name), offset="(0,0,0)", to="({}-c-east)".format(name),
                   s_filer="", n_filer=str(n_filer[1]), width=size[1][2], height=size[1][0], depth=size[1][1],
                   opacity=opacity, caption="SC{}".format(label[1])),
        to_Inception(
            name="{}-i1".format(name),
            s_filer="",
            n_filer=n_filer[2],
            to="({}-skip-east)".format(name),
            width=size[2][2],
            height=size[2][0],
            depth=size[2][1],
            caption="I{}".format(label[2])
        ),
        to_Inception(
            name="{}-i2".format(name),
            s_filer="",
            n_filer=n_filer[3],
            to="({}-i1-east)".format(name),
            width=size[3][2],
            height=size[3][0],
            depth=size[3][1],
            caption="I{}".format(label[3])
        ),
        to_Inception(
            name="{}-i3".format(name),
            s_filer=str(s_filer),
            n_filer=n_filer[4],
            to="({}-i2-east)".format(name),
            width=size[4][2],
            height=size[4][0],
            depth=size[4][1],
            caption="I{}".format(label[4])
        ),
        to_connection(
            "{}".format(bottom),
            "{}-c".format(name)
        )
    ]


def block_UnconvSkipInceptionx4(name, bottom, s_filer=256, n_filer=(64, 64, 64, 64, 64, 64), offset="(1,0,0)",
                                size=((32, 32, 5), (32, 32, 3.5), (32, 32, 3.5), (32, 32, 3.5), (32, 32, 3.5),
                                      (32, 32, 5)),
                                opacity=0.5, label=('1', '1', '1', '1', '1', '1')):
    return [
        to_UnPool(name='{}-unpool'.format(name), offset=offset, to="({}-east)".format(bottom), width=size[0][2],
                  height=size[0][0],
                  depth=size[0][1], opacity=opacity, caption="TC{}".format(label[0]), n_filer=str(n_filer[0])),
        to_ConvRes(name='{}-skip'.format(name), offset="(0,0,0)", to="({}-unpool-east)".format(name),
                   s_filer="", n_filer=str(n_filer[1]), width=size[1][2], height=size[1][0], depth=size[1][1],
                   opacity=opacity, caption="SC{}".format(label[1])),
        to_Inception(
            name="{}-i1".format(name),
            s_filer="",
            n_filer=n_filer[2],
            to="({}-skip-east)".format(name),
            width=size[2][2],
            height=size[2][0],
            depth=size[2][1],
            caption="I{}".format(label[2])
        ),
        to_Inception(
            name="{}-i2".format(name),
            s_filer="",
            n_filer=n_filer[3],
            to="({}-i1-east)".format(name),
            width=size[3][2],
            height=size[3][0],
            depth=size[3][1],
            caption="I{}".format(label[3])
        ),
        to_Inception(
            name="{}-i3".format(name),
            s_filer='',
            n_filer=n_filer[4],
            to="({}-i2-east)".format(name),
            width=size[4][2],
            height=size[4][0],
            depth=size[4][1],
            caption="I{}".format(label[4])
        ),
        to_Inception(
            name="{}-i4".format(name),
            s_filer=str(s_filer),
            n_filer=n_filer[5],
            to="({}-i3-east)".format(name),
            width=size[5][2],
            height=size[5][0],
            depth=size[5][1],
            caption="I{}".format(label[5])
        ),
        to_connection(
            "{}".format(bottom),
            "{}-unpool".format(name)
        )
    ]


def block_ConvSkipInceptionx4(name, bottom, s_filer=256, n_filer=(64, 64, 64, 64, 64, 64), offset="(1,0,0)",
                              size=((32, 32, 5), (32, 32, 3.5), (32, 32, 3.5), (32, 32, 3.5), (32, 32, 3.5),
                                    (32, 32, 5)),
                              opacity=0.5, label=('1', '1', '1', '1', '1', '1')):
    return [
        to_Conv(name='{}-c'.format(name), offset=offset, to="({}-east)".format(bottom), s_filer="",
                n_filer=str(n_filer[0]), width=size[0][2], height=size[0][0], depth=size[0][1],
                caption="C{}".format(label[0])),
        to_ConvRes(name='{}-skip'.format(name), offset="(0,0,0)", to="({}-c-east)".format(name),
                   s_filer="", n_filer=str(n_filer[1]), width=size[1][2], height=size[1][0], depth=size[1][1],
                   opacity=opacity, caption="SC{}".format(label[1])),
        to_Inception(
            name="{}-i1".format(name),
            s_filer="",
            n_filer=n_filer[2],
            to="({}-skip-east)".format(name),
            width=size[2][2],
            height=size[2][0],
            depth=size[2][1],
            caption="I{}".format(label[2])
        ),
        to_Inception(
            name="{}-i2".format(name),
            s_filer="",
            n_filer=n_filer[3],
            to="({}-i1-east)".format(name),
            width=size[3][2],
            height=size[3][0],
            depth=size[3][1],
            caption="I{}".format(label[3])
        ),
        to_Inception(
            name="{}-i3".format(name),
            s_filer='',
            n_filer=n_filer[4],
            to="({}-i2-east)".format(name),
            width=size[4][2],
            height=size[4][0],
            depth=size[4][1],
            caption="I{}".format(label[4])
        ),
        to_Inception(
            name="{}-i4".format(name),
            s_filer=str(s_filer),
            n_filer=n_filer[5],
            to="({}-i3-east)".format(name),
            width=size[5][2],
            height=size[5][0],
            depth=size[5][1],
            caption="I{}".format(label[5])
        ),
        to_connection(
            "{}".format(bottom),
            "{}-c".format(name)
        )
    ]


def block_UnconvSkipConv(name, bottom, s_filer=256, n_filer=(64, 64, 64), offset="(1,0,0)",
                         size=((32, 32, 5), (32, 32, 3.5), (32, 32, 3.5)),
                         opacity=0.5, label=('1', '1', '1')):
    return [
        to_UnPool(name='{}-unpool'.format(name), offset=offset, to="({}-east)".format(bottom), width=size[0][2],
                  height=size[0][0],
                  depth=size[0][1], opacity=opacity, n_filer=str(n_filer[0]),
                  caption="TC{}".format(label[0])),
        to_ConvRes(name='{}-skip'.format(name), offset="(0,0,0)", to="({}-unpool-east)".format(name),
                   s_filer="", n_filer=str(n_filer[1]), width=size[1][2], height=size[1][0], depth=size[1][1],
                   opacity=opacity, caption="SC{}".format(label[1])),
        to_Conv(name='{}-c'.format(name), offset="(0,0,0)", to="({}-skip-east)".format(name), s_filer=str(s_filer),
                n_filer=str(n_filer[2]), width=size[2][2], height=size[2][0], depth=size[2][1],
                caption="C{}".format(label[2])),
        to_connection(
            "{}".format(bottom),
            "{}-unpool".format(name)
        )
    ]


def block_UnconvSkipConvx2(name, bottom, s_filer=256, n_filer=(64, 64, 64, 64), offset="(1,0,0)",
                           size=((32, 32, 5), (32, 32, 3.5), (32, 32, 3.5), (32, 32, 3.5)),
                           opacity=0.5, label=('1', '1', '1', '1')):
    return [
        to_UnPool(name='{}-unpool'.format(name), offset=offset, to="({}-east)".format(bottom), width=size[0][2],
                  height=size[0][0],
                  depth=size[0][1], opacity=opacity, n_filer=str(n_filer[0]),
                  caption="TC{}".format(label[0])),
        to_ConvRes(name='{}-skip'.format(name), offset="(0,0,0)", to="({}-unpool-east)".format(name),
                   s_filer="", n_filer=str(n_filer[1]), width=size[1][2], height=size[1][0], depth=size[1][1],
                   opacity=opacity, caption="SC{}".format(label[1])),
        to_Conv(name='{}-c1'.format(name), offset="(0,0,0)", to="({}-skip-east)".format(name), s_filer="",
                n_filer=str(n_filer[2]), width=size[2][2], height=size[2][0], depth=size[2][1],
                caption="C{}".format(label[2])),
        to_Conv(name='{}-c2'.format(name), offset="(0,0,0)", to="({}-c1-east)".format(name), s_filer=str(s_filer),
                n_filer=str(n_filer[3]), width=size[3][2], height=size[3][0], depth=size[3][1],
                caption="C{}".format(label[3])),
        to_connection(
            "{}".format(bottom),
            "{}-unpool".format(name)
        )
    ]
