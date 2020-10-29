import numpy as np

def update_weights(model, grads, hyper_params):
    '''
    Update the weights of each layer in your model based on the calculated gradients
    Args:
        model: Dictionary holding the model
        grads: A list of gradients of each layer in model["layers"]
        hyper_params: 
            hyper_params['learning_rate']
            hyper_params['weight_decay']: Should be applied to W only.
    Returns: 
        updated_model:  Dictionary holding the updated model
    '''
    num_layers = len(grads)
    a = hyper_params["learning_rate"]
    lmd = hyper_params["weight_decay"]
    updated_model = model

    # TODO: Update the weights of each layer in your model based on the calculated gradients
    for i in range(num_layers):
        layer = updated_model['layers'][i]
        if layer['type'] == 'linear' or layer['type'] == 'conv':
            assert layer['params']['W'] is not None
            assert layer['params']['b'] is not None
            assert grads[i]['W'].shape == layer['params']['W'].shape
            assert grads[i]['b'].shape == layer['params']['b'].shape

            # layer['params']['W'] -= a * (grads[i]['W']) + lmd*layer['params']['W']
            layer['params']['W'] -= a * (grads[i]['W'] + 2*lmd*layer['params']['W'])
            layer['params']['b'] -= a * grads[i]['b']

    return updated_model