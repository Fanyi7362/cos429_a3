import sys
sys.path += ['../data']
import numpy as np
import time

from load_MNIST_images import load_MNIST_images
from load_MNIST_labels import load_MNIST_labels

from train import train
from init_layers import init_layers
from inference import inference

np.random.seed(0)


def train_model(use_trained, model_name='model.npz', input_mean_name='input_mean.npy'):
    # Load training data
    train_data = load_MNIST_images('../data/train-images.idx3-ubyte')
    train_label = load_MNIST_labels('../data/train-labels.idx1-ubyte')

    im_height, im_width, num_channels, num_train = train_data.shape
    input_mean = np.mean(train_data, axis=-1)
    train_data = (train_data.T - input_mean.T).T
    np.save(input_mean_name, input_mean)
    make_val_data(input_mean_name)
    output_size = int(np.max(train_label) - np.min(train_label) + 1)

    batch_size = 1024
    learning_rate = 0.4
    weight_decay = 0.0001

    lr_decay = 0.995
    lr_decay_step = 1
    momentum_rho = 0.8

    early_stop_ratio = 1e-5
    save_step = 1

    numIters = 300

    # temp = 1
    # index = np.random.choice(num_train, batch_size*temp, replace=False)  
    # train_data = train_data[..., index]; train_label = train_label[..., index]

    if use_trained: 
        model = np.load(model_name, allow_pickle=True)
        model = dict(model)

    else:
      layers = [init_layers('conv', {'filter_size': 5, 
                                     'filter_depth': num_channels, 
                                     'num_filters': 6,
                                     'weight_scale': 1,
                                     'bias_scale': 0.1,
                                     }),
                init_layers('relu', {}),

                init_layers('pool', {'filter_size': 2,
                                     'stride': 2}),              

                init_layers('conv', {'filter_size': 5, 
                                     'filter_depth': 6, 
                                     'num_filters': 16,
                                     'weight_scale': 1,
                                     'bias_scale': 0.1,
                                     }),
                init_layers('relu', {}),

                init_layers('pool', {'filter_size': 2,
                                     'stride': 2}),              

                init_layers('flatten', {}),

                init_layers('linear', {'num_in': 4*4*16,
                                       'num_out': 120,
                                       # 'weight_scale': 0.0,
                                       # 'bias_scale': 0
                                       }),
                init_layers('relu', {}),

                init_layers('linear', {'num_in': 120,
                                       'num_out': 84,
                                       'weight_scale': 1,
                                       # 'bias_scale': 0
                                       }),

                init_layers('relu', {}),              

                init_layers('linear', {'num_in': 84,
                                       'num_out': output_size,
                                       'weight_scale': 1,
                                       # 'bias_scale': 0
                                       }),

                init_layers('softmax', {})]

      model = {'layers': layers,
               'input_size': [im_height, im_width, num_channels],
               'output_size': output_size}

    params = {'learning_rate': learning_rate, 
              'weight_decay': weight_decay, 
              'batch_size': batch_size, 
              'save_file': model_name, 

              'lr_decay': lr_decay,
              'lr_decay_step': lr_decay_step,
              'momentum_rho': momentum_rho,

              'early_stop_ratio': early_stop_ratio,
              'save_step': save_step}

    start = time.time()
    model, train_loss, train_accuracy, val_loss, val_accuracy = train(model, train_data, train_label, params, numIters)
    stop = time.time()

    np.save('training_loss_file', train_loss)
    np.save('training_accuracy_file', train_accuracy)
    np.save('val_loss_file', val_loss)
    np.save('val_accuracy_file', val_accuracy)
    print("Done training, time used: {:0.1f} min".format((stop-start)/60))



def test_model(model_name, input_mean_name):
    print('Testing...')
    # Load testing data
    test_data = load_MNIST_images('../data/t10k-images.idx3-ubyte')
    test_label = load_MNIST_labels('../data/t10k-labels.idx1-ubyte')
    # index = np.random.choice(100, batch_size*temp, replace=False)  
    # test_data = test_data[..., index]; test_label = test_label[..., index]
    num_test = test_data.shape[-1]

    model = np.load(model_name, allow_pickle=True)
    model = dict(model)

    input_mean = np.load(input_mean_name)
    test_data = (test_data.T - input_mean.T).T

    output, _ = inference(model, test_data)
    prediction = np.argmax(output, axis=0)
    accuracy = np.sum(prediction==test_label) / num_test
    print("num_test={:d}, testing_accuracy={:0.3f}".format(num_test, accuracy))


def make_val_data(input_mean_name):
    val_data = load_MNIST_images('../data/t10k-images.idx3-ubyte')
    val_label = load_MNIST_labels('../data/t10k-labels.idx1-ubyte')
    num_val = 500
    index = np.random.choice(val_data.shape[-1], num_val, replace=False)  
    val_data = val_data[..., index]
    val_label = val_label[..., index]
    input_mean = np.load(input_mean_name)
    val_data = (val_data.T - input_mean.T).T
    np.save('val_data', val_data)
    np.save('val_label', val_label)


def main():

    model_name = 'model.npz'
    input_mean_name = 'input_mean.npy'

    train = True
    use_trained = True

    if train: train_model(use_trained, model_name, input_mean_name)

    test_model(model_name, input_mean_name)


if __name__ == '__main__':
    main()
