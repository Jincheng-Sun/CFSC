from Training.origin_resnet import ResBlock_type1, ResBlock_type2
from keras.layers import Input, Conv2D, BatchNormalization, Activation, AvgPool2D, Flatten, Dense
from keras import Model

def Res50(num):
    input = Input(shape=[100, 100, 1])
    conv1 = Conv2D(filters=64, kernel_size=7, strides=2, padding='same')(input)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)

    block1 = ResBlock_type2(layer=act1, filters=(64, 64, 256), kernels=(1, 3, 1),
                            activation='relu', shift=True, shrink=False)
    block2 = ResBlock_type2(layer=block1, filters=(64, 64, 256), kernels=(1, 3, 1),
                            activation='relu', shift=False, shrink=False)
    block3 = ResBlock_type2(layer=block2, filters=(64, 64, 256), kernels=(1, 3, 1),
                            activation='relu', shift=False, shrink=False)

    block4 = ResBlock_type2(layer=block3, filters=(128, 128, 512), kernels=(1, 3, 1),
                            activation='relu', shift=True, shrink=True)
    block5 = ResBlock_type2(layer=block4, filters=(128, 128, 512), kernels=(1, 3, 1),
                            activation='relu', shift=False, shrink=False)
    block6 = ResBlock_type2(layer=block5, filters=(128, 128, 512), kernels=(1, 3, 1),
                            activation='relu', shift=False, shrink=False)
    block7 = ResBlock_type2(layer=block6, filters=(128, 128, 512), kernels=(1, 3, 1),
                            activation='relu', shift=False, shrink=False)

    block8 = ResBlock_type2(layer=block7, filters=(256, 256, 1024), kernels=(1, 3, 1),
                            activation='relu', shift=True, shrink=True)
    block9 = ResBlock_type2(layer=block8, filters=(256, 256, 1024), kernels=(1, 3, 1),
                            activation='relu', shift=False, shrink=False)
    block10 = ResBlock_type2(layer=block9, filters=(256, 256, 1024), kernels=(1, 3, 1),
                             activation='relu', shift=False, shrink=False)
    block11 = ResBlock_type2(layer=block10, filters=(256, 256, 1024), kernels=(1, 3, 1),
                             activation='relu', shift=False, shrink=False)
    block12 = ResBlock_type2(layer=block11, filters=(256, 256, 1024), kernels=(1, 3, 1),
                             activation='relu', shift=False, shrink=False)
    block13 = ResBlock_type2(layer=block12, filters=(256, 256, 1024), kernels=(1, 3, 1),
                             activation='relu', shift=False, shrink=False)

    block14 = ResBlock_type2(layer=block13, filters=(512, 512, 2048), kernels=(1, 3, 1),
                             activation='relu', shift=True, shrink=True)
    block15 = ResBlock_type2(layer=block14, filters=(512, 512, 2048), kernels=(1, 3, 1),
                             activation='relu', shift=False, shrink=False)
    block16 = ResBlock_type2(layer=block15, filters=(512, 512, 2048), kernels=(1, 3, 1),
                             activation='relu', shift=False, shrink=False)

    pool = AvgPool2D(pool_size=7)(block16)
    flt = Flatten()(pool)
    fc = Dense(num, activation='softmax', kernel_initializer='random_uniform')(flt)
    model = Model(inputs=[input], outputs=[fc])
    model.summary()
    return model

Res50(5)
