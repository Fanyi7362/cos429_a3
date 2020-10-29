import numpy as np

def calc_gradient(model, input, layer_acts, dv_output):
    '''
    Calculate the gradient at each layer, to do this you need dv_output
    determined by your loss function and the activations of each layer.
    The loop of this function will look very similar to the code from
    inference, just looping in reverse.
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [batch_size]
        layer_acts: A list of activations of each layer in model["layers"]
        dv_output: The partial derivative of the loss with respect to each element in the output matrix of the last layer.
    Returns: 
        grads:  A list of gradients of each layer in model["layers"]
    '''
    num_layers = len(model["layers"])
    grads = [None,] * num_layers

    # TODO: Determine the gradient at each layer.
    #       Remember that back-propagation traverses 
    #       the model in the reverse order.
    dv_output_ = dv_output
    for i in range(num_layers)[::-1]:
        layer = model['layers'][i]
        fn = layer['fwd_fn']
        if i == 0:
            input_ = input
        else:
            input_ = layer_acts[i-1]

        _, dv_output_, grad = fn(input_, layer['params'], layer['hyper_params'], True, dv_output_)
        grads[i] = grad

    return grads