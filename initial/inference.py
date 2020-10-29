import numpy as np

def inference(model, input):
    """
    Do forward propagation through the network to get the activation
    at each layer, and the final output
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [batch_size]
    Returns:
        output: The final output of the model
        activations: A list of activations for each layer in model["layers"]
    """

    num_layers = len(model['layers'])
    activations = [None,] * num_layers

    # TODO: FORWARD PROPAGATION CODE
    for i in range(num_layers):
        layer = model['layers'][i]
        fwd_fn = layer['fwd_fn']
        if i == 0:
            input_ = input
        else:
            input_ = activations[i-1]

        output_, _, _ = fwd_fn(input_, layer['params'], layer['hyper_params'], False)
        activations[i] = output_

    output = activations[-1]
    return output, activations
