import sys
sys.path += ['../data']
import numpy as np
import time
import matplotlib.pyplot as plt

from load_MNIST_images import load_MNIST_images
from load_MNIST_labels import load_MNIST_labels

from train import train
from init_layers import init_layers
from inference import inference

np.random.seed(0)


def train_model(use_trained, output_folder, model_name='model.npz', input_mean_name='input_mean.npy', plots_suffix='_'):
    # Load training data
    train_data = load_MNIST_images('../data/train-images.idx3-ubyte')
    train_label = load_MNIST_labels('../data/train-labels.idx1-ubyte')

    im_height, im_width, num_channels, num_train = train_data.shape
    input_mean = np.mean(train_data, axis=-1)
    train_data = (train_data.T - input_mean.T).T
    np.save(output_folder+input_mean_name, input_mean)
    make_val_data(input_mean_name, output_folder)
    output_size = int(np.max(train_label) - np.min(train_label) + 1)

    batch_size = 512
    learning_rate = 0.2
    weight_decay = 0.0001

    lr_decay = 0.98
    lr_decay_step = 2
    momentum_rho = 0.8

    early_stop_ratio = 1e-5
    save_step = 1

    numIters = 300

    # temp = 1
    # index = np.random.choice(num_train, batch_size*temp, replace=False)  
    # train_data = train_data[..., index]; train_label = train_label[..., index]

    if use_trained: 
        model = np.load(output_folder+model_name, allow_pickle=True)
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
              'save_step': save_step,
              'output_folder': output_folder
              }

    start = time.time()
    model, train_loss, train_accuracy, val_loss, val_accuracy = train(model, train_data, train_label, params, numIters)
    stop = time.time()
    print("Done training, time used: {:0.1f} min".format((stop-start)/60))

    np.save(output_folder+'training_loss_file', train_loss)
    np.save(output_folder+'training_accuracy_file', train_accuracy)
    np.save(output_folder+'val_loss_file', val_loss)
    np.save(output_folder+'val_accuracy_file', val_accuracy)

    # plot learning and test(val) loss curve
    train_iter_range = np.array(range(1, numIters+1))
    val_iter_range = np.array(range(save_step, numIters+1, save_step))

    fig, ax = plt.subplots()
    ax.plot(train_iter_range, train_loss, 'b', label='training')
    ax.plot(val_iter_range, val_loss, 'r', label='testing')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    legend = ax.legend(loc='upper right', shadow=True)
    plt.title('training and testing losses VS number of iterations')
    # plt.show()
    plt.savefig(output_folder+'plot_losses' + plots_suffix + '.png')

    fig, ax = plt.subplots()
    ax.plot(train_iter_range, train_accuracy, 'b', label='training')
    ax.plot(val_iter_range, val_accuracy, 'r', label='testing')
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    legend = ax.legend(loc='upper left', shadow=True)
    plt.title('training and testing accuracies VS number of iterations')
    # plt.show()
    plt.savefig(output_folder+'plot_accuracies' + plots_suffix + '.png')


def test_model(model_name, input_mean_name, output_folder):
    print('Testing...')
    # Load testing data
    test_data = load_MNIST_images('../data/t10k-images.idx3-ubyte')
    test_label = load_MNIST_labels('../data/t10k-labels.idx1-ubyte')
    num_test = test_data.shape[-1]

    model = np.load(output_folder+model_name, allow_pickle=True)
    model = dict(model)

    input_mean = np.load(output_folder+input_mean_name)
    test_data = (test_data.T - input_mean.T).T

    output, _ = inference(model, test_data)
    prediction = np.argmax(output, axis=0)
    accuracy = np.sum(prediction==test_label) / num_test
    print("num_test={:d}, testing_accuracy={:0.3f}".format(num_test, accuracy))


def make_val_data(input_mean_name, output_folder):
    val_data = load_MNIST_images('../data/t10k-images.idx3-ubyte')
    val_label = load_MNIST_labels('../data/t10k-labels.idx1-ubyte')
    num_val = 1000
    index = np.random.choice(val_data.shape[-1], num_val, replace=False)  
    val_data = val_data[..., index]
    val_label = val_label[..., index]
    input_mean = np.load(output_folder+input_mean_name)
    val_data = (val_data.T - input_mean.T).T
    np.save(output_folder+'val_data', val_data)
    np.save(output_folder+'val_label', val_label)


def main():

    model_name = 'model.npz'
    input_mean_name = 'input_mean.npy'

    output_folder = './out_lenet5/'

    train = True
    use_trained = False

    if train:
      plots_suffix = '_'
      train_model(use_trained, output_folder, model_name, input_mean_name, plots_suffix)

    test_model(model_name, input_mean_name, output_folder)


if __name__ == '__main__':
    main()
