from .utils import *
import numpy as np
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms as transforms
from .hw4_utils.hw4_models import GoogLeNet
from PIL import Image as PILImage
import scipy.ndimage
import cv2
import deepul.pytorch_util as ptu

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import numpy as np
import math
import sys

softmax = None
model = None
device = torch.device("cuda:0")

def plot_vae_training_plot(train_losses, test_losses, title, fname):
    elbo_train, recon_train, kl_train = train_losses[:, 0], train_losses[:, 1], train_losses[:, 2]
    elbo_test, recon_test, kl_test = test_losses[:, 0], test_losses[:, 1], test_losses[:, 2]
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, elbo_train, label='-elbo_train')
    plt.plot(x_train, recon_train, label='recon_loss_train')
    plt.plot(x_train, kl_train, label='kl_loss_train')
    plt.plot(x_test, elbo_test, label='-elbo_test')
    plt.plot(x_test, recon_test, label='recon_loss_test')
    plt.plot(x_test, kl_test, label='kl_loss_test')

    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    savefig(fname)


def sample_data_1_a(count):
    rand = np.random.RandomState(0)
    return [[1.0, 2.0]] + (rand.randn(count, 2) * [[5.0, 1.0]]).dot(
        [[np.sqrt(2) / 2, np.sqrt(2) / 2], [-np.sqrt(2) / 2, np.sqrt(2) / 2]])


def sample_data_2_a(count):
    rand = np.random.RandomState(0)
    return [[-1.0, 2.0]] + (rand.randn(count, 2) * [[1.0, 5.0]]).dot(
        [[np.sqrt(2) / 2, np.sqrt(2) / 2], [-np.sqrt(2) / 2, np.sqrt(2) / 2]])


def sample_data_1_b(count):
    rand = np.random.RandomState(0)
    return [[1.0, 2.0]] + rand.randn(count, 2) * [[5.0, 1.0]]


def sample_data_2_b(count):
    rand = np.random.RandomState(0)
    return [[-1.0, 2.0]] + rand.randn(count, 2) * [[1.0, 5.0]]


def q1_sample_data(part, dset_id):
    assert dset_id in [1, 2]
    assert part in ['a', 'b']
    if part == 'a':
        if dset_id == 1:
            dset_fn = sample_data_1_a
        else:
            dset_fn = sample_data_2_a
    else:
        if dset_id == 1:
            dset_fn = sample_data_1_b
        else:
            dset_fn = sample_data_2_b

    train_data, test_data = dset_fn(10000), dset_fn(2500)
    return train_data.astype('float32'), test_data.astype('float32')


def visualize_q1_data(part, dset_id):
    train_data, test_data = q1_sample_data(part, dset_id)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Train Data')
    ax1.scatter(train_data[:, 0], train_data[:, 1])
    ax2.set_title('Test Data')
    ax2.scatter(test_data[:, 0], test_data[:, 1])
    print(f'Dataset {dset_id}')
    plt.show()


def q1_a_save_results(part, dset_id, fn):
    train_data, test_data = q1_sample_data(part, dset_id)
    train_losses, test_losses, samples_noise, samples_nonoise = fn(train_data, test_data, part, dset_id)
    print(f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, '
          f'KL Loss: {test_losses[-1, 2]:.4f}')

    plot_vae_training_plot(train_losses, test_losses, f'Q1({part}) Dataset {dset_id} Train Plot',
                           f'results/q1_{part}_dset{dset_id}_train_plot.png')
    save_scatter_2d(samples_noise, title='Samples with Decoder Noise',
                    fname=f'results/q1_{part}_dset{dset_id}_sample_with_noise.png')
    save_scatter_2d(samples_nonoise, title='Samples without Decoder Noise',
                    fname=f'results/q1_{part}_dset{dset_id}_sample_without_noise.png')


def visualize_colored_shapes():
    data_dir = get_data_dir(3)
    train_data, test_data = load_pickled_data(join(data_dir, 'shapes_colored.pkl'))
    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs]
    show_samples(images, title='Colored Shapes Samples')


def visualize_svhn():
    data_dir = get_data_dir(3)
    train_data, test_data = load_pickled_data(join(data_dir, 'svhn.pkl'))
    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs]
    show_samples(images, title='SVHN Samples')


def visualize_cifar10():
    data_dir = get_data_dir(3)
    train_data, test_data = load_pickled_data(join(data_dir, 'cifar10.pkl'))
    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs]
    show_samples(images, title='CIFAR10 Samples')


def q2_save_results(part, dset_id, fn):
    assert part in ['a', 'b'] and dset_id in [1, 2]
    data_dir = get_data_dir(3)
    if dset_id == 1:
        train_data, test_data = load_pickled_data(join(data_dir, 'svhn.pkl'))
    else:
        train_data, test_data = load_pickled_data(join(data_dir, 'cifar10.pkl'))

    train_losses, test_losses, samples, reconstructions, interpolations = fn(train_data, test_data, dset_id)
    samples, reconstructions, interpolations = samples.astype('float32'), reconstructions.astype('float32'), interpolations.astype('float32')
    print(f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, '
          f'KL Loss: {test_losses[-1, 2]:.4f}')
    plot_vae_training_plot(train_losses, test_losses, f'Q2({part}) Dataset {dset_id} Train Plot',
                           f'results/q2_{part}_dset{dset_id}_train_plot.png')
    show_samples(samples, title=f'Q2({part}) Dataset {dset_id} Samples',
                 fname=f'results/q2_{part}_dset{dset_id}_samples.png')
    show_samples(reconstructions, title=f'Q2({part}) Dataset {dset_id} Reconstructions',
                 fname=f'results/q2_{part}_dset{dset_id}_reconstructions.png')
    show_samples(interpolations, title=f'Q2({part}) Dataset {dset_id} Interpolations',
                 fname=f'results/q2_{part}_dset{dset_id}_interpolations.png')


# def q3_save_results(dset_id, fn):
#     assert dset_id in [1, 2]
#     data_dir = get_data_dir(3)
#     if dset_id == 1:
#         train_data, test_data = load_pickled_data(join(data_dir, 'svhn.pkl'))
#     else:
#         train_data, test_data = load_pickled_data(join(data_dir, 'cifar10.pkl'))

#     vqvae_train_losses, vqvae_test_losses, pixelcnn_train_losses, pixelcnn_test_losses, samples, reconstructions = fn(train_data, test_data, dset_id)
#     samples, reconstructions = samples.astype('float32'), reconstructions.astype('float32')
#     print(f'VQ-VAE Final Test Loss: {vqvae_test_losses[-1]:.4f}')
#     print(f'PixelCNN Prior Final Test Loss: {pixelcnn_test_losses[-1]:.4f}')
#     save_training_plot(vqvae_train_losses, vqvae_test_losses,f'Q3 Dataset {dset_id} VQ-VAE Train Plot',
#                        f'results/q3_dset{dset_id}_vqvae_train_plot.png')
#     save_training_plot(pixelcnn_train_losses, pixelcnn_test_losses,f'Q3 Dataset {dset_id} PixelCNN Prior Train Plot',
#                        f'results/q3_dset{dset_id}_pixelcnn_train_plot.png')
#     show_samples(samples, title=f'Q3 Dataset {dset_id} Samples',
#                  fname=f'results/q3_dset{dset_id}_samples.png')
#     show_samples(reconstructions, title=f'Q3 Dataset {dset_id} Reconstructions',
#                  fname=f'results/q3_dset{dset_id}_reconstructions.png')


def q4_a_save_results(dset_id, fn):
    assert dset_id in [1, 2]
    data_dir = get_data_dir(3)
    if dset_id == 1:
        train_data, test_data = load_pickled_data(join(data_dir, 'svhn.pkl'))
    else:
        train_data, test_data = load_pickled_data(join(data_dir, 'cifar10.pkl'))

    vqvae_train_losses, vqvae_test_losses, pixelcnn_train_losses, pixelcnn_test_losses, samples, reconstructions = fn(train_data, test_data, dset_id)
    samples, reconstructions = samples.astype('float32'), reconstructions.astype('float32')
    print(f'VQ-VAE Final Test Loss: {vqvae_test_losses[-1]:.4f}')
    print(f'PixelCNN Prior Final Test Loss: {pixelcnn_test_losses[-1]:.4f}')
    save_training_plot(vqvae_train_losses, vqvae_test_losses,f'Q4(a) Dataset {dset_id} VQ-VAE Train Plot',
                       f'results/q4_a_dset{dset_id}_vqvae_train_plot.png')
    save_training_plot(pixelcnn_train_losses, pixelcnn_test_losses,f'Q4(a) Dataset {dset_id} PixelCNN Prior Train Plot',
                       f'results/q4_a_dset{dset_id}_pixelcnn_train_plot.png')
    show_samples(samples, title=f'Q4(a) Dataset {dset_id} Samples',
                 fname=f'results/q4_a_dset{dset_id}_samples.png')
    show_samples(reconstructions, title=f'Q4(a) Dataset {dset_id} Reconstructions',
                 fname=f'results/q4_a_dset{dset_id}_reconstructions.png')


def q4_b_save_results(fn):
    part = 'b'
    data_dir = get_data_dir(3)
    train_data, test_data = load_pickled_data(join(data_dir, 'mnist.pkl'))

    train_losses, test_losses, samples, reconstructions = fn(train_data, test_data)
    samples, reconstructions = samples.astype('float32') * 255, reconstructions.astype('float32') * 255
    print(f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, '
          f'KL Loss: {test_losses[-1, 2]:.4f}')
    plot_vae_training_plot(train_losses, test_losses, f'Q4({part}) Train Plot',
                           f'results/q4_{part}_train_plot.png')
    show_samples(samples, title=f'Q4({part}) Samples',
                 fname=f'results/q4_{part}_samples.png')
    show_samples(reconstructions, title=f'Q4({part}) Reconstructions',
                 fname=f'results/q4_{part}_reconstructions.png')

def af_vae_save_results(part, dset_id, fn):
    assert part in ['a', 'b'] and dset_id in [1, 2]
    data_dir = get_data_dir(3)
    if dset_id == 1:
        train_data, test_data = load_pickled_data(join(data_dir, 'svhn.pkl'))
    else:
        train_data, test_data = load_pickled_data(join(data_dir, 'cifar10.pkl'))

    train_losses, test_losses, samples, reconstructions, interpolations = fn(train_data, test_data, dset_id)
    samples, reconstructions, interpolations = samples.astype('float32'), reconstructions.astype('float32'), interpolations.astype('float32')
    print(f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, '
          f'KL Loss: {test_losses[-1, 2]:.4f}')
    plot_vae_training_plot(train_losses, test_losses, f'Q4({part}) Dataset {dset_id} Train Plot',
                           f'results/q4_{part}_dset{dset_id}_train_plot.png')
    show_samples(samples, title=f'Q4({part}) Dataset {dset_id} Samples',
                 fname=f'results/q4_{part}_dset{dset_id}_samples.png')
    show_samples(reconstructions, title=f'Q4({part}) Dataset {dset_id} Reconstructions',
                 fname=f'results/q4_{part}_dset{dset_id}_reconstructions.png')
    show_samples(interpolations, title=f'Q4({part}) Dataset {dset_id} Interpolations',
                 fname=f'results/q4_{part}_dset{dset_id}_interpolations.png')
###########
###########
## GANS
    
def plot_gan_training(losses, title, fname):
    plt.figure()
    n_itr = len(losses)
    xs = np.arange(n_itr)

    plt.plot(xs, losses, label='loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Training Iteration')
    plt.ylabel('Loss')
    savefig(fname)

def q1_gan_plot(data, samples, xs, ys, title, fname):
    plt.figure()
    plt.hist(samples, bins=50, density=True, alpha=0.7, label='fake')
    plt.hist(data, bins=50, density=True, alpha=0.7, label='real')

    plt.plot(xs, ys, label='discrim')
    plt.legend()
    plt.title(title)
    savefig(fname)
    
def q1_data(n=20000):
    assert n % 2 == 0
    gaussian1 = np.random.normal(loc=-1, scale=0.25, size=(n//2,))
    gaussian2 = np.random.normal(loc=0.5, scale=0.5, size=(n//2,))
    data = (np.concatenate([gaussian1, gaussian2]) + 1).reshape([-1, 1])
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    return 2 * scaled_data -1

def visualize_q1_dataset():
    data = q1_data()
    plt.hist(data, bins=50, alpha=0.7, label='train data')
    plt.legend()
    plt.show()


def q1_b_save_results(part, fn):
    data = q1_data()
    losses, samples1, xs1, ys1, samples_end, xs_end, ys_end = fn(data)
    part = 'b'
    # loss plot
    plot_gan_training(losses, 'Q1{} Losses'.format(part), 'results/q1{}_losses.png'.format(part))

    # samples
    q1_gan_plot(data, samples1, xs1, ys1, 'Q1{} Epoch 1'.format(part), 'results/q1{}_epoch1.png'.format(part))
    q1_gan_plot(data, samples_end, xs_end, ys_end, 'Q1{} Final'.format(part), 'results/q1{}_final.png'.format(part))
    
    
def load_q3_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return train_data, test_data

def visualize_q3_data():
    train_data, _ = load_q3_data()
    imgs = train_data.data[:100]
    show_samples(imgs.reshape([100, 28, 28, 1]) * 255.0, title=f'MNIST samples')

def plot_q3_supervised(pretrained_losses, random_losses, title, fname):
    plt.figure()
    xs = np.arange(len(pretrained_losses))
    plt.plot(xs, pretrained_losses, label='bigan')
    xs = np.arange(len(random_losses))
    plt.plot(xs, random_losses, label='random init')
    plt.legend()
    plt.title(title)
    savefig(fname)

def q3_save_results(fn):
    train_data, test_data = load_q3_data()
    gan_losses, samples, reconstructions, pretrained_losses, random_losses = fn(train_data, test_data)

    plot_gan_training(gan_losses, 'Q3 Losses', 'results/q3_gan_losses.png')
    plot_q3_supervised(pretrained_losses, random_losses, 'Linear classification losses', 'results/q3_supervised_losses.png')
    show_samples(samples * 255.0, fname='results/q3_samples.png', title='BiGAN generated samples')
    show_samples(reconstructions * 255.0, nrow=20, fname='results/q3_reconstructions.png', title=f'BiGAN reconstructions')
    print('BiGAN final linear classification loss:', pretrained_losses[-1])
    print('Random encoder linear classification loss:', random_losses[-1])
    
def get_colored_mnist(data):
    # from https://www.wouterbulten.nl/blog/tech/getting-started-with-gans-2-colorful-mnist/
    # Read Lena image
    lena = PILImage.open('deepul/deepul/hw4_utils/lena.jpg')

    # Resize
    batch_resized = np.asarray([scipy.ndimage.zoom(image, (2.3, 2.3, 1), order=1) for image in data])

    # Extend to RGB
    batch_rgb = np.concatenate([batch_resized, batch_resized, batch_resized], axis=3)

    # Make binary
    batch_binary = (batch_rgb > 0.5)

    batch = np.zeros((data.shape[0], 28, 28, 3))

    for i in range(data.shape[0]):
        # Take a random crop of the Lena image (background)
        x_c = np.random.randint(0, lena.size[0] - 64)
        y_c = np.random.randint(0, lena.size[1] - 64)
        image = lena.crop((x_c, y_c, x_c + 64, y_c + 64))
        image = np.asarray(image) / 255.0

        # Invert the colors at the location of the number
        image[batch_binary[i]] = 1 - image[batch_binary[i]]

        batch[i] = cv2.resize(image, (0, 0), fx=28 / 64, fy=28 / 64, interpolation=cv2.INTER_AREA)
    return batch.transpose(0, 3, 1, 2)

def load_q4_data():
    train, _ = load_q3_data()
    mnist = np.array(train.data.reshape(-1, 28, 28, 1) / 255.0)
    colored_mnist = get_colored_mnist(mnist)
    return mnist.transpose(0, 3, 1, 2), colored_mnist

def visualize_cyclegan_datasets():
    mnist, colored_mnist = load_q4_data()
    mnist, colored_mnist = mnist[:100], colored_mnist[:100]
    show_samples(mnist.reshape([100, 28, 28, 1]) * 255.0, title=f'MNIST samples')
    show_samples(colored_mnist.transpose([0, 2, 3, 1]) * 255.0, title=f'Colored MNIST samples')

def q4_save_results(fn):
    mnist, cmnist = load_q4_data()

    m1, c1, m2, c2, m3, c3 = fn(mnist, cmnist)
    m1, m2, m3 = m1.repeat(3, axis=3), m2.repeat(3, axis=3), m3.repeat(3, axis=3)
    mnist_reconstructions = np.concatenate([m1, c1, m2], axis=0)
    colored_mnist_reconstructions = np.concatenate([c2, m3, c3], axis=0)

    show_samples(mnist_reconstructions * 255.0, nrow=20,
                 fname='figures/q4_mnist.png',
                 title=f'Source domain: MNIST')
    show_samples(colored_mnist_reconstructions * 255.0, nrow=20,
                 fname='figures/q4_colored_mnist.png',
                 title=f'Source domain: Colored MNIST')
    pass

def calculate_is(samples):
    assert (type(samples[0]) == np.ndarray)
    assert (len(samples[0].shape) == 3)

    model = GoogLeNet().to(ptu.device)
    model.load_state_dict(torch.load("deepul/deepul/hw4_utils/classifier.pt"))
    softmax = nn.Sequential(model, nn.Softmax(dim=1))

    bs = 100
    softmax.eval()
    with torch.no_grad():
        preds = []
        n_batches = int(math.ceil(float(len(samples)) / float(bs)))
        for i in range(n_batches):
            sys.stdout.write(".")
            sys.stdout.flush()
            inp = ptu.FloatTensor(samples[(i * bs):min((i + 1) * bs, len(samples))])
            pred = ptu.get_numpy(softmax(inp))
            preds.append(pred)
    preds = np.concatenate(preds, 0)
    kl = preds * (np.log(preds) - np.log(np.expand_dims(np.mean(preds, 0), 0)))
    kl = np.mean(np.sum(kl, 1))
    return np.exp(kl)

def load_q2_data():
    train_data = torchvision.datasets.CIFAR10("./data", transform=torchvision.transforms.ToTensor(),
                                              download=True, train=True)
    return train_data

def visualize_q2_data():
    train_data = load_q2_data()
    imgs = train_data.data[:100]
    show_samples(imgs, title=f'CIFAR-10 Samples')

def q2_b_save_results(fn):
    train_data = load_q2_data()
    train_data = train_data.data.transpose((0, 3, 1, 2)) / 255.0
    train_losses, samples = fn(train_data)

    print("Inception score:", calculate_is(samples.transpose([0, 3, 1, 2])))
    plot_gan_training(train_losses, 'Q2b Losses', 'results/q2b_losses.png')
    show_samples(samples[:100] * 255.0, fname='results/q2b_samples.png', title=f'CIFAR-10 generated samples')