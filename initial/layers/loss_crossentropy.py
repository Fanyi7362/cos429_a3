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
    num_nodes, batch_size = input.shape
    label_onehot = np.zeros([num_nodes, batch_size])
    for ii in range(batch_size):
        label_onehot[int(labels[ii]),ii] = 1

    loss = - np.sum(np.multiply(label_onehot, np.log(input)))/batch_size


    eps = 0.00001
    dv_input = np.zeros(0)
    if backprop:
        dv_input = np.zeros(input.shape)
        
        # TODO: BACKPROP CODE
        #       Add a small eps to the denominator to avoid numerical instability
        for ii in range(batch_size):
            dv_input[int(labels[ii]),ii] = - 1/(input[int(labels[ii]),ii]+eps)


    return loss, dv_input
