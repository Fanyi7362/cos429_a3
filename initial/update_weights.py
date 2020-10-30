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
    rho = hyper_params.get("momentum_rho", 0)
    iter_n = hyper_params.get("iter_n", 0)
    updated_model = model

    # TODO: Update the weights of each layer in your model based on the calculated gradients
    for i in range(num_layers):
        layer = updated_model['layers'][i]
        if layer['type'] == 'linear' or layer['type'] == 'conv':
            # assert layer['params']['W'] is not None
            # assert layer['params']['b'] is not None
            # assert grads[i]['W'].shape == layer['params']['W'].shape
            # assert grads[i]['b'].shape == layer['params']['b'].shape

            if iter_n==0:
                layer['params']['W_m'] = np.zeros(grads[i]['W'].shape)
                layer['params']['b_m'] = np.zeros(grads[i]['b'].shape)

                layer['params']['W_m'] = np.copy(grads[i]['W'])
                layer['params']['b_m'] = np.copy(grads[i]['b'])
            else:
                layer['params']['W_m'] = rho*layer['params']['W_m'] + (1-rho)*grads[i]['W']
                layer['params']['b_m'] = rho*layer['params']['b_m'] + (1-rho)*grads[i]['b']

            # layer['params']['W'] -= a * (grads[i]['W_m'] + 2*lmd*layer['params']['W'])
            layer['params']['W'] -= a * layer['params']['W_m'] + lmd*layer['params']['W']
            layer['params']['b'] -= a * layer['params']['b_m']

    return updated_model