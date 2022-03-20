from .utils import *
from sklearn.datasets import make_moons
import numpy as np
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms as transforms
from .hw4_utils.hw4_models import GoogLeNet
import scipy.ndimage
import cv2
import deepul.pytorch_util as ptu
import math
import sys

def make_scatterplot(points, title=None, filename=None):
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], s=1)
    if title is not None:
        plt.title(title)
    # if filename is not None:
    #     plt.savefig("q1_{}.png".format(filename))

######################
##### Question 1 #####
######################

def load_smiley_face(n):
    count = n
    rand = np.random.RandomState(0)
    a = [[-1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    b = [[1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    c = np.c_[2 * np.cos(np.linspace(0, np.pi, count // 3)),
              -np.sin(np.linspace(0, np.pi, count // 3))]
    c += rand.randn(*c.shape) * 0.2
    data_x = np.concatenate([a, b, c], axis=0)
    data_y = np.array([0] * len(a) + [1] * len(b) + [2] * len(c))
    perm = rand.permutation(len(data_x))
    return data_x[perm], data_y[perm]

def load_half_moons(n):
    return make_moons(n_samples=n, noise=0.1)

def q1_sample_data_1():
    train_data, train_labels = load_smiley_face(2000)
    test_data, test_labels = load_smiley_face(1000)
    return train_data, train_labels, test_data, test_labels

def q1_sample_data_2():
    train_data, train_labels = load_half_moons(2000)
    test_data, test_labels = load_half_moons(1000)
    return train_data, train_labels, test_data, test_labels

def visualize_q1_data(dset_type):
    if dset_type == 1:
        train_data, train_labels, test_data, test_labels = q1_sample_data_1()
    elif dset_type == 2:
        train_data, train_labels, test_data, test_labels = q1_sample_data_2()
    else:
        raise Exception('Invalid dset_type:', dset_type)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.8))
    ax1.set_title('Train Data')
    ax1.scatter(train_data[:, 0], train_data[:, 1], s=1, c=train_labels)
    ax1.set_xlabel('x1')
    ax1.set_xlabel('x2')
    ax2.set_title('Test Data')
    ax2.scatter(test_data[:, 0], test_data[:, 1], s=1, c=test_labels)
    ax1.set_xlabel('x1')
    ax1.set_xlabel('x2')
    print(f'Dataset {dset_type}')
    plt.show()

def show_2d_samples(samples, fname=None, title='Samples'):
    plt.figure()
    plt.title(title)
    plt.scatter(samples[:, 0], samples[:, 1], s=1)
    plt.xlabel('x1')
    plt.ylabel('x2')

    if fname is not None:
        savefig(fname)
    else:
        plt.show()

def show_2d_latents(latents, labels, fname=None, title='Latent Space'):
    plt.figure()
    plt.title(title)
    plt.scatter(latents[:, 0], latents[:, 1], s=1, c=labels)
    plt.xlabel('z1')
    plt.ylabel('z2')

    if fname is not None:
        savefig(fname)
    else:
        plt.show()

def show_2d_densities(densities, dset_type, fname=None, title='Densities'):
    plt.figure()
    plt.title(title)
    dx, dy = 0.025, 0.025
    if dset_type == 1: # face
        x_lim = (-4, 4)
        y_lim = (-4, 4)
    elif dset_type == 2: # moons
        x_lim = (-1.5, 2.5)
        y_lim = (-1, 1.5)
    else:
        raise Exception('Invalid dset_type:', dset_type)
    y, x = np.mgrid[slice(y_lim[0], y_lim[1] + dy, dy),
                    slice(x_lim[0], x_lim[1] + dx, dx)]
    # mesh_xs = ptu.FloatTensor(np.stack([x, y], axis=2).reshape(-1, 2))
    # densities = np.exp(ptu.get_numpy(self.log_prob(mesh_xs)))
    plt.pcolor(x, y, densities.reshape([y.shape[0], y.shape[1]]))
    plt.pcolor(x, y, densities.reshape([y.shape[0], y.shape[1]]))
    plt.xlabel('z1')
    plt.ylabel('z2')
    if fname is not None:
        savefig(fname)
    else:
        plt.show()

def q1_save_results(dset_type, fn):
    if dset_type == 1:
        train_data, train_labels, test_data, test_labels = q1_sample_data_1()
    elif dset_type == 2:
        train_data, train_labels, test_data, test_labels = q1_sample_data_2()
    else:
        raise Exception('Invalid dset_type:', dset_type)

    train_losses, test_losses, densities, latents = fn(train_data, test_data, dset_type)

    print(f'Final Test Loss: {test_losses[-1]:.4f}')

    save_training_plot(train_losses, test_losses, f'Q1 Dataset {dset_type} Train Plot',
                       f'results/q1_dset{dset_type}_train_plot.png')
    show_2d_densities(densities, dset_type, fname=f'results/q1_dset{dset_type}_densities.png')
    show_2d_latents(latents, train_labels, f'results/q1_dset{dset_type}_latents.png')


######################
##### Question 2 #####
######################

def visualize_q2_q3_data():
    data_dir = get_data_dir(2)
    train_data, test_data = load_pickled_data(join(data_dir, 'celeb.pkl'))
    name = 'CelebA'

    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs].astype(np.float32) / 3.0 * 255.0
    show_samples(images, title=f'{name} Samples')

def get_q2_q3_data():
    data_dir = get_data_dir(2)
    train_data, test_data = load_pickled_data(join(data_dir, 'celeb.pkl'))
    train_data = np.transpose(train_data, axes=[0, 3, 1, 2])  # NCHW 20000 x 3 x 32 x 32
    test_data = np.transpose(test_data, axes=[0, 3, 1, 2])  # NCHW 6838 x 3 x 32 x 32
    return train_data, test_data

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

def q2_save_results(part, fn):
    assert part in ['a', 'b']
    train_data, test_data = get_q2_q3_data()

    train_losses, test_losses, samples, reconstructions, interpolations = fn(train_data, test_data)
    samples, reconstructions, interpolations = samples.astype('float32'), reconstructions.astype('float32'), interpolations.astype('float32')
    print(f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, '
          f'KL Loss: {test_losses[-1, 2]:.4f}')
    plot_vae_training_plot(train_losses, test_losses, f'Q2({part}) Train Plot',
                           f'results/q2_{part}_train_plot.png')
    show_samples(samples, title=f'Q2({part}) Samples',
                 fname=f'results/q2_{part}_samples.png')
    show_samples(reconstructions, title=f'Q2({part}) Reconstructions',
                 fname=f'results/q2_{part}_reconstructions.png')
    show_samples(interpolations, title=f'Q2({part}) Interpolations',
                 fname=f'results/q2_{part}_interpolations.png')

def q3_save_results(fn):
    data_dir = get_data_dir(3)
    train_data, test_data = get_q2_q3_data()

    vqvae_train_losses, vqvae_test_losses, pixelcnn_train_losses, pixelcnn_test_losses, samples, reconstructions = fn(train_data, test_data)
    samples, reconstructions = samples.astype('float32'), reconstructions.astype('float32')
    print(f'VQ-VAE Final Test Loss: {vqvae_test_losses[-1]:.4f}')
    print(f'PixelCNN Prior Final Test Loss: {pixelcnn_test_losses[-1]:.4f}')
    save_training_plot(vqvae_train_losses, vqvae_test_losses,f'Q3 VQ-VAE Train Plot',
                       f'results/q3_vqvae_train_plot.png')
    save_training_plot(pixelcnn_train_losses, pixelcnn_test_losses,f'Q3 PixelCNN Prior Train Plot',
                       f'results/q3_pixelcnn_train_plot.png')
    show_samples(samples, title=f'Q3 Samples',
                 fname=f'results/q3_samples.png')
    show_samples(reconstructions, title=f'Q3 Reconstructions',
                 fname=f'results/q3_reconstructions.png')
