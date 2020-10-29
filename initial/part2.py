import sys
sys.path += ['../data']
import numpy as np

from load_MNIST_images import load_MNIST_images
from load_MNIST_labels import load_MNIST_labels

from train import train
from init_layers import init_layers
from inference import inference

np.random.seed(0)


def train_model(model_name='model.npz'):
    # Load training data
    train_data = load_MNIST_images('../data/train-images.idx3-ubyte')
    train_label = load_MNIST_labels('../data/train-labels.idx1-ubyte')

    im_height, im_width, num_channels, num_train = train_data.shape
    output_size = int(np.max(train_label) - np.min(train_label) + 1)

    batch_size = 256
    learning_rate = 0.01
    weight_decay = 0.0005

    lr_decay = 0.95
    lr_decay_step = 10
    early_stop_ratio = 1e-5
    save_step = 1

    numIters = 300

    layers = [init_layers('conv', {'filter_size': 3, 
                                   'filter_depth': num_channels, 
                                   'num_filters': 4}),
              init_layers('relu', {}),

              init_layers('conv', {'filter_size': 3, 
                                   'filter_depth': 4, 
                                   'num_filters': 8}),
              init_layers('relu', {}),

              init_layers('pool', {'filter_size': 2,
                                   'stride': 2}),

              init_layers('conv', {'filter_size': 3, 
                                   'filter_depth': 8, 
                                   'num_filters': 4}),
              init_layers('relu', {}),

              init_layers('pool', {'filter_size': 2,
                                   'stride': 2}),

              init_layers('flatten', {}),

              init_layers('linear', {'num_in': 5*5*4,
                                     'num_out': output_size}),

              init_layers('softmax', {})]

    # layers = [init_layers('conv', {'filter_size': 3, 
    #                                'filter_depth': num_channels, 
    #                                'num_filters': 32}),
    #           init_layers('relu', {}),

    #           init_layers('conv', {'filter_size': 3, 
    #                                'filter_depth': 32, 
    #                                'num_filters': 16}),
    #           init_layers('relu', {}),

    #           init_layers('pool', {'filter_size': 2,
    #                                'stride': 2}),

    #           init_layers('conv', {'filter_size': 3, 
    #                                'filter_depth': 16, 
    #                                'num_filters': 4}),
    #           init_layers('relu', {}),

    #           init_layers('pool', {'filter_size': 2,
    #                                'stride': 2}),

    #           init_layers('flatten', {}),

    #           # init_layers('linear', {'num_in': 5*5*8,
    #           #                        'num_out': 32}),
    #           # init_layers('relu', {}),

    #           init_layers('linear', {'num_in': 5*5*4,
    #                                  'num_out': output_size}),

    #           init_layers('softmax', {})]

    model = {'layers': layers,
             'input_size': [im_height, im_width, num_channels],
             'output_size': output_size}

    params = {'learning_rate': learning_rate, 
              'weight_decay': weight_decay, 
              'batch_size': batch_size, 
              'save_file': model_name, 

              'lr_decay': lr_decay,
              'lr_decay_step': lr_decay_step,
              'early_stop_ratio': early_stop_ratio,
              'save_step': save_step}

    model, loss = train(model, train_data, train_label, params, numIters)


def test_model(model_name='model.npz'):
    # Load testing data
    test_data = load_MNIST_images('../data/t10k-images.idx3-ubyte')
    test_label = load_MNIST_labels('../data/t10k-labels.idx1-ubyte')
    num_test = test_data.shape[-1]

    model = np.load(model_name, allow_pickle=True)
    model = dict(model)

    output, _ = inference(model, test_data)
    prediction = np.argmax(output, axis=0)
    accuracy = np.sum(prediction==test_label) / num_test
    print("num_test={:d}, testing_accuracy={:0.3f}".format(num_test, accuracy))


def main():
    train = True

    model_name = 'model.npz'
    if train: train_model(model_name)
    test_model(model_name)


if __name__ == '__main__':
    main()
