import sys
sys.path += ['layers']
import numpy as np

######################################################
# Set use_pcode to True to use the provided pyc code
# for inference, calc_gradient, loss_crossentropy and update_weights
use_pcode = False

# You can modify the imports of this section to indicate
# whether to use the provided pyc or your own code for each of the four functions.
if use_pcode:
    # import the provided pyc implementation
    sys.path += ['pyc_code']
    from inference_ import inference
    from calc_gradient_ import calc_gradient
    from loss_crossentropy_ import loss_crossentropy
    from update_weights_ import update_weights
else:
    # import your own implementation
    from inference import inference
    from calc_gradient import calc_gradient
    from loss_crossentropy import loss_crossentropy
    from update_weights import update_weights
######################################################

def train(model, input, label, params, numIters):
    '''
    This training function is written specifically for classification,
    since it uses crossentropy loss and tests accuracy assuming the final output
    layer is a softmax layer. These can be changed for more general use.
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [num_inputs]
        label: [num_inputs]
        params: Paramters for configuring training
            params["learning_rate"]
            params["weight_decay"]
            params["batch_size"]
            params["save_file"]
            Free to add more parameters to this dictionary for your convenience of training.
        numIters: Number of training iterations
    '''
    # Initialize training parameters
    # Learning rate
    lr = params.get("learning_rate", .01)
    # Weight decay
    wd = params.get("weight_decay", .0005)
    # Batch size
    batch_size = params.get("batch_size", 128)
    # There is a good chance you will want to save your network model during/after
    # training. It is up to you where you save and how often you choose to back up
    # your model. By default the code saves the model in 'model.npz'.
    save_file = params.get("save_file", 'model.npz')

    lr_decay = params.get("lr_decay", .95)
    lr_decay_step = params.get("lr_decay_step", 10)
    momentum_rho = params.get("momentum_rho", 0)

    early_stop_ratio = params.get("early_stop_ratio", 1e-5)
    save_step = params.get("save_step", 10)

    # update_params will be passed to your update_weights function.
    # This allows flexibility in case you want to implement extra features like momentum.
    update_params = {"learning_rate": lr,
                     "weight_decay": wd,
                     "momentum_rho": momentum_rho,
                     "iter_n": 0}

    val_data = np.load('val_data.npy')
    val_label = np.load('val_label.npy')

    num_inputs = input.shape[-1]
    train_loss = np.zeros((numIters,))
    train_accuracy = np.zeros((numIters,))
    val_loss, val_accuracy = [], []

    for i in range(numIters):
        # TODO: One training iteration
        # Steps:
        #   (1) Select a subset of the input to use as a batch
        #   (2) Run inference on the batch
        #   (3) Calculate loss and determine accuracy
        #   (4) Calculate gradients
        #   (5) Update the weights of the model
        # Optionally,
        #   (1) Monitor the progress of training
        #   (2) Save your learnt model, using ``np.savez(save_file, **model)``

        index = np.random.choice(num_inputs, batch_size, replace=False)  
        input_ = input[..., index]
        label_ = label[..., index]

        output_, activations_ = inference(model, input_)
        prediction_ = np.argmax(output_, axis=0)
        train_accuracy[i] = np.sum(prediction_==label_) / batch_size

        train_loss_, dv_output_ = loss_crossentropy(output_, label_, {}, True)
        train_loss[i] = train_loss_

        grads_ = calc_gradient(model, input_, activations_, dv_output_)

        update_params['iter_n'] = i
        model = update_weights(model, grads_, update_params)

        if (i+1)%lr_decay_step == 0:
            update_params["learning_rate"] = lr_decay * update_params["learning_rate"]

        if (i+1)%save_step == 0:
            np.savez(save_file, **model)

            start_idx = i + 1 - save_step
            train_loss_ma = np.sum(train_loss[start_idx:])/save_step
            train_acc_ma = np.sum(train_accuracy[start_idx:])/save_step

            # validation test
            val_output_, _ = inference(model, val_data)
            val_prediction_ = np.argmax(val_output_, axis=0)
            val_accuracy_ = np.sum(val_prediction_==val_label) / val_label.shape[-1]
            val_loss_, _ = loss_crossentropy(val_output_, val_label, {}, False)
            val_accuracy.append(val_accuracy_)
            val_loss.append(val_loss_)

            print("iter={:d}, mavg_loss={:0.4f}, mavg_train_acc={:0.4f}, val_loss={:0.4f}, val_acc={:0.4f}"
                .format(i+1, train_loss_ma, train_acc_ma, val_loss_, val_accuracy_))

            # if (i+1)==save_step: loss_ma_prev = loss_ma
            # if i>=save_step:
            #     if np.abs(loss_ma-loss_ma_prev)/(loss_ma_prev+1e-10) < early_stop_ratio:
            #         print('Stop early')
            #         break
            #     loss_ma_prev = loss_ma

    np.savez(save_file, **model)

    return model, train_loss, train_accuracy, np.array(val_loss), np.array(val_accuracy)
