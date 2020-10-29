import numpy as np
import scipy.signal

def fn_conv(input, params, hyper_params, backprop, dv_output=None):
    """
    Args:
        input: The input data to the layer function. [in_height] x [in_width] x [num_channels] x [batch_size] array
        params: Weight and bias information for the layer.
            params['W']: layer weights, [filter_height] x [filter_width] x [filter_depth] x [num_filters] array
            params['b']: layer bias, [num_filters] x 1 array
        hyper_params: Optional, could include information such as stride and padding.
        backprop: Boolean stating whether or not to compute the output terms for backpropagation.
        dv_output: The partial derivative of the loss with respect to each element in the output matrix. Only passed in when backprop is set to true. Same size as output.

    Returns:
        output: Output of layer, [out_height] x [out_width] x [num_filters] x [batch_size] array
        dv_input: The derivative of the loss with respect to the input. Same size as input.
        grad: The gradient term that you will use to update the weights defined in params and train your network. Dictionary with same structure as params.
            grad['W']: gradient wrt weights, same size as params['W']
            grad['b']: gradient wrt bias, same size as params['b']
    """

    in_height, in_width, num_channels, batch_size = input.shape
    _, _, filter_depth, num_filters = params['W'].shape
    out_height = in_height - params['W'].shape[0] + 1
    out_width = in_width - params['W'].shape[1] + 1

    assert params['W'].shape[2] == input.shape[2], 'Filter depth does not match number of input channels'

    # Initialize
    output = np.zeros((out_height, out_width, num_filters, batch_size))
    dv_input = np.zeros(0)
    grad = {'W': np.zeros(0),
            'b': np.zeros(0)}
    
    # TODO: FORWARD CODE
    #       Update output with values
    # for i in range(batch_size):
    #     for j in range(num_filters):
    #         for k in range(num_channels):
    #             output[:,:,j,i] += scipy.signal.convolve(input[:,:,k,i], params['W'][:,:,k,j], mode='valid') + params['b'][j]/num_channels
    for i in range(batch_size):
        for j in range(num_filters):
            output[:,:,j,i] = scipy.signal.convolve(input[:,:,:,i], np.flip(params['W'][:,:,:,j]), mode='valid')[...,0] + params['b'][j]
    # for j in range(num_filters):
    #     Wj_ = np.tile(np.expand_dims(params['W'][:,:,:,j], axis=-1), [1,batch_size])#[...,::-1]
    #     # print(Wj_.shape)
    #     output[:,:,j,:] = scipy.signal.convolve(input[:,:,:,:], Wj_, mode='valid')[...,0] + params['b'][j]
    #     # print(output[:,:,j,:].shape)
    # # print(output.shape)


    if backprop:
        assert dv_output is not None
        dv_input = np.zeros(input.shape)
        grad['W'] = np.zeros(params['W'].shape)
        grad['b'] = np.zeros(params['b'].shape)
        
        # TODO: BACKPROP CODE
        #       Update dv_input and grad with values
        # for i in range(batch_size):
        #     for k in range(num_channels):
        #         for j in range(num_filters):
        #             dv_input[:,:,k,i] += scipy.signal.convolve(dv_output[:,:,j,i], np.flip(params['W'][:,:,k,j]), mode='full')
        #             grad['W'][:,:,k,j] += np.flip(scipy.signal.convolve(input[:,:,k,i], np.flip(dv_output[:,:,j,i]), mode='valid'))
        for i in range(batch_size):
            for k in range(num_channels):
                for j in range(num_filters):
                    dv_input[:,:,k,i] += scipy.signal.convolve(dv_output[:,:,j,i], params['W'][:,:,k,j], mode='full')
                    grad['W'][:,:,k,j] += scipy.signal.convolve(input[:,:,k,i], np.flip(dv_output[:,:,j,i]), mode='valid')
        # for i in range(batch_size):
        #     for j in range(num_filters):
        #         dv_input[:,:,:,i] += scipy.signal.convolve(np.tile(np.expand_dims(dv_output[:,:,j,i], axis=-1), [1,num_channels]), params['W'][:,:,:,j], mode='full')[:,:,5]

        grad['b'] = np.sum(dv_output, axis=(0,1,3)).reshape(-1,1) / batch_size
        grad['W'] = grad['W'] / batch_size


    return output, dv_input, grad
