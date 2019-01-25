from keras.layers import BatchNormalization, Conv2D, Activation, Dropout, GlobalAveragePooling2D, add, Input, Dense, \
    MaxPool2D

def bn_relu(layer, dropout=0, **params):
    layer = BatchNormalization()(layer)
    layer = Activation(params['conv_activation'])(layer)

    if dropout > 0:
        layer = Dropout(dropout)(layer)
    return layer

def global_average_pooling(layer, cls):
    layer = Conv2D(cls, [1, 1])(layer)
    layer = GlobalAveragePooling2D()(layer)
    layer = Activation(activation='softmax')(layer)
    return layer

def ResBlock_original(layer, filters, kernels, strides, dropout, activation, channels_change=False):
    # ↱--------------------↴
    # --Conv-BN-Act-Conv-BN-Act-
    # ↳-Conv-BN------------↑

    filter1, filter2, filter3 = filters
    kernel1, kernel2, kernel3 = kernels
    stride1, stride2, stride3 = strides

    if channels_change:
        shortcut = Conv2D(filters=filter1,
                          kernel_size=kernel1,
                          kernel_initializer='random_uniform',
                          # kernel_regularizer=regularizers.l2(0.01),
                          strides=stride1,
                          padding='same')(layer)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = layer

    layer = Conv2D(filters=filter2,
                   kernel_size=kernel2,
                   kernel_initializer='random_uniform',
                   # kernel_regularizer=regularizers.l2(0.01),
                   strides=stride2,
                   padding='same')(layer)
    layer = bn_relu(layer, dropout=dropout, conv_activation=activation)

    layer = Conv2D(filters=filter3,
                   kernel_size=kernel3,
                   kernel_initializer='random_uniform',
                   # kernel_regularizer=regularizers.l2(0.01),
                   strides=stride3,
                   padding='same')(layer)

    layer = BatchNormalization()(layer)

    layer = add([shortcut, layer])

    layer = Activation(activation=activation)(layer)

    return layer


def ResBlock_3layers(layer, filters, kernels,strides, activation, dropout = 0, channel_change=False):
    # ↱--------------------------------↴
    # -Conv-BN-Act-Conv-BN-Act-conv-BN--Act-
    # ↳------------------------Conv-BN-↑

    filter1, filter2, filter3 = filters
    kernel1, kernel2, kernel3 = kernels
    stride1, stride2, stride3 = strides

    if channel_change:
        shortcut = Conv2D(filters=filter3,
                          kernel_size=kernel1,
                          kernel_initializer='random_uniform',
                          # kernel_regularizer=regularizers.l2(0.01),
                          strides=stride1,
                          padding='same')(layer)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = layer

    layer = Conv2D(filters=filter1,
                   kernel_size=kernel1,
                   kernel_initializer='random_uniform',
                   # kernel_regularizer=regularizers.l2(0.01),
                   strides=stride1,
                   padding='same')(layer)
    layer = bn_relu(layer, dropout=dropout, conv_activation=activation)

    layer = Conv2D(filters=filter2,
                   kernel_size=kernel2,
                   kernel_initializer='random_uniform',
                   # kernel_regularizer=regularizers.l2(0.01),
                   strides=stride2,
                   padding='same')(layer)
    layer = bn_relu(layer, dropout=dropout, conv_activation=activation)

    layer = Conv2D(filters=filter3,
                   kernel_size=kernel3,
                   kernel_initializer='random_uniform',
                   # kernel_regularizer=regularizers.l2(0.01),
                   strides=stride3,
                   padding='same')(layer)

    layer = BatchNormalization()(layer)

    output = add([shortcut, layer])

    output = Activation(activation=activation)(output)

    return output


def ResBlock_preAct(layer, filters, kernels, strides, activation, dropout=0, channel_change=False):
    #             ↱-----------↴
    # -BN-Act-Conv-BN-Act-Conv-
    # ↳-----------------------↑

    filter1, filter2 = filters
    kernel1, kernel2 = kernels
    stride1, stride2 = strides

    shortcut = layer

    layer = bn_relu(layer, dropout=dropout, conv_activation=activation)

    layer = Conv2D(filters=filter1,
                   kernel_size=kernel1,
                   kernel_initializer='random_uniform',
                   # kernel_regularizer=regularizers.l2(0.01),
                   strides=stride1,
                   padding='same')(layer)

    if channel_change:
        shortcut = layer

    layer = bn_relu(layer, dropout=dropout, conv_activation=activation)

    layer = Conv2D(filters=filter2,
                   kernel_size=kernel2,
                   kernel_initializer='random_uniform',
                   # kernel_regularizer=regularizers.l2(0.01),
                   strides=stride2,
                   padding='same')(layer)

    output = add([shortcut, layer])

    return output


def ResBlock_preAct_3layers(layer, filters, kernels, strides, activation, dropout=0, channel_change=False):
    # ↱------------------------------------↴
    # -BN-Act-Conv-BN-Act-Conv-BN-Act-Conv--
    #        ↳------------------------Conv-↑

    filter1, filter2, filter3 = filters
    kernel1, kernel2, kernel3 = kernels
    stride1, stride2, stride3 = strides

    shortcut = layer
    # BN1
    layer = bn_relu(layer, dropout=dropout, conv_activation=activation)

    if channel_change:

        shortcut = Conv2D(filters=filter3,
                       kernel_size=kernel1,
                       kernel_initializer='random_uniform',
                       # kernel_regularizer=regularizers.l2(0.01),
                       strides=stride1,
                       padding='same')(layer)

    layer = Conv2D(filters=filter1,
                   kernel_size=kernel1,
                   kernel_initializer='random_uniform',
                   # kernel_regularizer=regularizers.l2(0.01),
                   strides=stride1,
                   padding='same')(layer)

    layer = bn_relu(layer, dropout=dropout, conv_activation=activation)

    layer = Conv2D(filters=filter2,
                   kernel_size=kernel2,
                   kernel_initializer='random_uniform',
                   # kernel_regularizer=regularizers.l2(0.01),
                   strides=stride2,
                   padding='same')(layer)

    layer = bn_relu(layer, dropout=dropout, conv_activation=activation)

    layer = Conv2D(filters=filter3,
                   kernel_size=kernel3,
                   kernel_initializer='random_uniform',
                   # kernel_regularizer=regularizers.l2(0.01),
                   strides=stride3,
                   padding='same')(layer)

    output = add([shortcut, layer])

    return output