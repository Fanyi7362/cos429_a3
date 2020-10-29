import numpy as np

def loss_crossentropy(input, labels, hyper_params, backprop):
    """
    Args:
        input: [num_nodes] x [batch_size] array
        labels: [batch_size] array
        hyper_params: Dummy input. This is included to maintain consistency across all layer and loss functions, but the input argument is not used.
        backprop: Boolean stating whether or not to compute the output terms for backpropagation.

    Returns:
        loss: scalar value, the loss averaged over the input batch
        dv_input: The derivative of the loss with respect to the input. Same size as input.
    """

    assert labels.max() <= input.shape[0]
    loss = 0

    # TODO: CALCULATE LOSS
    num_nodes, batch_size= input.shape
    for i in range(batch_size):
        loss += -np.log(input[int(labels[i]),i])
    loss /= batch_size


    eps = 0.00001
    dv_input = np.zeros(0)
    if backprop:
        dv_input = np.zeros(input.shape)
        
        # TODO: BACKPROP CODE
        #       Add a small eps to the denominator to avoid numerical instability
        for i in range(batch_size):
            dv_input[int(labels[i]),i] = -1 / (input[int(labels[i]),i] + eps)


    return loss, dv_input
