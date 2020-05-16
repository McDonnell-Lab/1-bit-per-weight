import numpy as np

import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.layers import InputSpec, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, GlobalAveragePooling2D, Lambda, concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

    
#***************************************************************************************************
#Definition of the binary 2D conv layer as in https://arxiv.org/abs/1802.08530
#M. D. McDonnell, Training wide residual networks for deployment using a single bit for each weight
#ICLR, 2018
#
#Adapated by M.D. McDonnell from https://github.com/DingKe/nn_playground/blob/master/binarynet/binary_layers.py
#
#See also: https://stackoverflow.com/questions/36456436/how-can-i-define-only-the-gradient-for-a-tensorflow-subgraph/36480182
#
#***************************************************************************************************
class BinaryConv2D(Conv2D):

    def __init__(self, filters, **kwargs):
        super(BinaryConv2D, self).__init__(filters, **kwargs)
        
        
    def build(self, input_shape):
        channel_axis = -1
        if self.data_format == 'channels_first':
            channel_axis = 1

        input_dim = int(input_shape[channel_axis])
        if input_dim is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')

        #***************************************************************************************************
        #Binary layer multiplier as in https://arxiv.org/abs/1802.08530
        #M. D. McDonnell, Training wide residual networks for deployment using a single bit for each weight
        #ICLR, 2018
        self.multiplier=np.sqrt(2.0/float(self.kernel_size[0])/float(self.kernel_size[1])/float(input_dim))
        #***************************************************************************************************
        
        self.kernel = self.add_weight(shape=self.kernel_size + (input_dim, self.filters),
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        
        #***************************************************************************************************
        #Binary layer as in https://arxiv.org/abs/1802.08530
        #M. D. McDonnell, Training wide residual networks for deployment using a single bit for each weight
        #ICLR, 2018
        #
        #This code sets the full precsion weights to binary for forward and bacjkward propagation
        #but enables gradients to update the full precision weights that ar used only during training 
        #
        binary_kernel = self.kernel + K.stop_gradient(K.sign(self.kernel) - self.kernel)
        binary_kernel=binary_kernel+K.stop_gradient(binary_kernel*self.multiplier-binary_kernel)
        #***************************************************************************************************
        
        outputs = K.conv2d( inputs,
                            binary_kernel,
                            strides=self.strides,
                            padding=self.padding,
                            data_format=self.data_format,
                            dilation_rate=self.dilation_rate)


        return outputs
        
    def get_config(self):
        config = {'multiplier': self.multiplier}
        base_config = super(BinaryConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
#***************************************************************************************************
#Definition of the  resnet variant in https://arxiv.org/abs/1802.08530
#M. D. McDonnell, Training wide residual networks for deployment using a single bit for each weight
#ICLR, 2018
#***************************************************************************************************
def resnet_layer(inputs,num_filters=16,kernel_size=3,strides=1,bn_moments_momentum=0.99,
                 learn_bn = True,wd=1e-4,UseRelu=True,UseBN=True,UseBinaryWeights=False):
    x = inputs
    if UseBN:
        #epsilon=1e-3 is keras default
        x = BatchNormalization(epsilon=1e-2,momentum=bn_moments_momentum,center=learn_bn,scale=learn_bn)(x)
    if UseRelu:
        x = Activation('relu')(x)
    if UseBinaryWeights:
        x = BinaryConv2D(num_filters,
                     kernel_size=kernel_size,
                     strides=strides,
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(wd),
                     use_bias=False)(x)
    else:
        x = Conv2D(num_filters,
                     kernel_size=kernel_size,
                     strides=strides,
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(wd),
                     use_bias=False)(x)
    return x



#***************************************************************************************************
#Definition of the resnet variant in https://arxiv.org/abs/1802.08530
#M. D. McDonnell, Training wide residual networks for deployment using a single bit for each weight
#ICLR, 2018
#***************************************************************************************************
def resnet(UseBinaryWeights,input_shape, depth, num_classes=10, width=1,wd=0.0):

    # Start model definition.
    base_filters = 16
    num_filters = base_filters*width
    
    bn_moments_momentum = 0.99 #this is keras default
    My_wd = wd
    
    num_res_blocks = int((depth - 2) / 6)

    #input layers prior to first branching
    inputs = Input(shape=input_shape)     
    
    
    ResidualPath = resnet_layer(inputs=inputs,
                     num_filters=num_filters,
                     kernel_size=3,
                     strides=1,
                     bn_moments_momentum=bn_moments_momentum,
                     learn_bn = True,
                     wd=My_wd,
                     UseRelu=False,
                     UseBN=True,
                     UseBinaryWeights=UseBinaryWeights)
    
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            ConvPath = resnet_layer(inputs=ResidualPath,
                             num_filters=num_filters,
                             kernel_size=3,
                             strides=strides, #sometimes this is 2
                             bn_moments_momentum=bn_moments_momentum,
                             learn_bn = False,
                             wd=My_wd,
                             UseBN=True,
                             UseBinaryWeights=UseBinaryWeights)
            ConvPath = resnet_layer(inputs=ConvPath,
                             num_filters=num_filters,
                             kernel_size=3,
                             strides=1,
                             bn_moments_momentum=bn_moments_momentum,
                             learn_bn = False,
                             wd=My_wd,
                             UseBN=True,
                             UseBinaryWeights=UseBinaryWeights)
            if stack > 0 and res_block == 0:  
                # first layer but not first stack: this is where we have gone up in channels and down in feature map size
                #so need to account for this in the residual path
                #average pool and downsample the residual path
                ResidualPath = AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(ResidualPath)
                
                #zero pad to increase channels
                ResidualPath=concatenate([ResidualPath,Lambda(K.zeros_like)(ResidualPath)])

            ResidualPath = tensorflow.keras.layers.add([ConvPath,ResidualPath])
            
        #when we are here, we double the number of filters    
        num_filters *= 2

    #output layers after last sum
    OutputPath = resnet_layer(inputs=ResidualPath,
                     num_filters=num_classes,
                     strides = 1,
                     kernel_size=1,
                     bn_moments_momentum=bn_moments_momentum,
                     learn_bn = False,
                     wd=My_wd,
                     UseBN=True,
                     UseBinaryWeights=UseBinaryWeights)
    OutputPath = BatchNormalization(epsilon=1e-2,momentum=bn_moments_momentum,center=False, scale=False)(OutputPath)
    OutputPath = GlobalAveragePooling2D()(OutputPath)
    OutputPath = Activation('softmax')(OutputPath)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=OutputPath)
    return model

  