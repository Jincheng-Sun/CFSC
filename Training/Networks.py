from Training.Resnet_block import ResBlock_original, ResBlock_3layers,global_average_pooling
from keras.layers import Input, Conv2D, BatchNormalization, Activation, AvgPool2D, Flatten, Dense,GlobalAveragePooling2D
from keras import Model

def Res50(num):
    input = Input(shape=[100, 100, 1])
    conv1 = Conv2D(filters=64, kernel_size=[7,7], strides=[2,2], padding='same')(input)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)

    block1 = ResBlock_3layers(layer=act1, filters=(64, 64, 256), kernels=(1, 3, 1),
                              activation='relu', channel_change=True, dimention_change=False)
    block2 = ResBlock_3layers(layer=block1, filters=(64, 64, 256), kernels=(1, 3, 1),
                              activation='relu', channel_change=False, dimention_change=False)
    block3 = ResBlock_3layers(layer=block2, filters=(64, 64, 256), kernels=(1, 3, 1),
                              activation='relu', channel_change=False, dimention_change=False)

    block4 = ResBlock_3layers(layer=block3, filters=(128, 128, 512), kernels=(1, 3, 1),
                              activation='relu', channel_change=True, dimention_change=True)
    block5 = ResBlock_3layers(layer=block4, filters=(128, 128, 512), kernels=(1, 3, 1),
                              activation='relu', channel_change=False, dimention_change=False)
    block6 = ResBlock_3layers(layer=block5, filters=(128, 128, 512), kernels=(1, 3, 1),
                              activation='relu', channel_change=False, dimention_change=False)
    block7 = ResBlock_3layers(layer=block6, filters=(128, 128, 512), kernels=(1, 3, 1),
                              activation='relu', channel_change=False, dimention_change=False)

    block8 = ResBlock_3layers(layer=block7, filters=(256, 256, 1024), kernels=(1, 3, 1),
                              activation='relu', channel_change=True, dimention_change=True)
    block9 = ResBlock_3layers(layer=block8, filters=(256, 256, 1024), kernels=(1, 3, 1),
                              activation='relu', channel_change=False, dimention_change=False)
    block10 = ResBlock_3layers(layer=block9, filters=(256, 256, 1024), kernels=(1, 3, 1),
                               activation='relu', channel_change=False, dimention_change=False)
    block11 = ResBlock_3layers(layer=block10, filters=(256, 256, 1024), kernels=(1, 3, 1),
                               activation='relu', channel_change=False, dimention_change=False)
    block12 = ResBlock_3layers(layer=block11, filters=(256, 256, 1024), kernels=(1, 3, 1),
                               activation='relu', channel_change=False, dimention_change=False)
    block13 = ResBlock_3layers(layer=block12, filters=(256, 256, 1024), kernels=(1, 3, 1),
                               activation='relu', channel_change=False, dimention_change=False)

    block14 = ResBlock_3layers(layer=block13, filters=(512, 512, 2048), kernels=(1, 3, 1),
                               activation='relu', channel_change=True, dimention_change=True)
    block15 = ResBlock_3layers(layer=block14, filters=(512, 512, 2048), kernels=(1, 3, 1),
                               activation='relu', channel_change=False, dimention_change=False)
    block16 = ResBlock_3layers(layer=block15, filters=(512, 512, 2048), kernels=(1, 3, 1),
                               activation='relu', channel_change=False, dimention_change=False)

    output = global_average_pooling(block16,5)
    model = Model(inputs=[input], outputs=[output])
    model.summary()
    return model

# Res50(5)
