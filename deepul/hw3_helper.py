
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from IPython.display import Image as GIF
from .utils import *

from six.moves.urllib.request import urlretrieve
import tarfile

import imageio
from PIL import Image
from urllib.error import URLError
from urllib.error import HTTPError
from sklearn.datasets import load_digits
import numpy as np
from torchvision.datasets import FashionMNIST
from torchvision import transforms

def get_file(fname,
             origin,
             untar=False,
             extract=False,
             archive_format='auto',
             cache_dir='data'):
    datadir = os.path.join(cache_dir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    print(fpath)
    if not os.path.exists(fpath):
        print('Downloading data from', origin)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath)
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise

    if untar:
        if not os.path.exists(untar_fpath):
            print('Extracting file.')
            with tarfile.open(fpath) as archive:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(archive, datadir)
        return untar_fpath

    return fpath

def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def generate_gif(directory_path, keyword=None):
    images = []
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(".png") and (keyword is None or keyword in filename):
            img_path = os.path.join(directory_path, filename)
            # print("adding image {}".format(img_path))
            images.append(imageio.imread(img_path))

    if keyword:
        imageio.mimsave(
            os.path.join(directory_path, 'anim_{}.gif'.format(keyword)), images)
    else:
        imageio.mimsave(os.path.join(directory_path, 'anim.gif'), images)


def create_image_grid(array, ncols=None):
    """
    """
    num_images, channels, cell_h, cell_w = array.shape
    if not ncols:
        ncols = int(np.sqrt(num_images))
    nrows = int(np.math.floor(num_images / float(ncols)))
    result = np.zeros((cell_h * nrows, cell_w * ncols, channels), dtype=array.dtype)
    for i in range(0, nrows):
        for j in range(0, ncols):
            result[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w, :] = array[i * ncols + j].transpose(1, 2,
                                                                                                                 0)

    if channels == 1:
        result = result.squeeze()
    return result

def get_emoji_dataloader(emoji_type):
    """Creates training and test data loaders.
    """
    transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    train_path = os.path.join('data/emojis', emoji_type)
    test_path = os.path.join('data/emojis', 'Test_{}'.format(emoji_type))

    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    train_dloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_dloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=0)

    return train_dloader, test_dloader

def merge_images(sources, targets):
    """Creates a grid consisting of pairs of columns, where the first column in
    each pair contains images source images and the second column in each pair
    contains images generated by the CycleGAN from the corresponding images in
    the first column.
    """
    _, _, h, w = sources.shape
    row = int(np.sqrt(32))
    merged = np.zeros([3, row * h, row * w * 2])
    for (idx, s, t) in (zip(range(row ** 2), sources, targets, )):
        i = idx // row
        j = idx % row
        merged[:, i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h] = s
        merged[:, i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h] = t
    return merged.transpose(1, 2, 0)

def create_image_grid(array, ncols=None):
    """
    """
    num_images, channels, cell_h, cell_w = array.shape
    if not ncols:
        ncols = int(np.sqrt(num_images))
    nrows = int(np.math.floor(num_images / float(ncols)))
    result = np.zeros((cell_h * nrows, cell_w * ncols, channels), dtype=array.dtype)
    for i in range(0, nrows):
        for j in range(0, ncols):
            result[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w, :] = array[i * ncols + j].transpose(1, 2,
                                                                                                                 0)

    if channels == 1:
        result = result.squeeze()
    return result

def gan_save_samples(G, fixed_noise, iteration):
    generated_images = G(fixed_noise)
    generated_images = to_data(generated_images)

    grid = create_image_grid(generated_images)

    # merged = merge_images(X, fake_Y, opts)
    temp_path = 'temp/q1/samples/'
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    path = os.path.join(temp_path, 'sample-{:06d}.png'.format(iteration))
    grid = (grid + 1)/2 * 255.
    grid = np.uint8(grid)
    imageio.imwrite(path, grid)
    print('Saved {}'.format(path))

def cyclegan_save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY):
    """Saves samples from both generators X->Y and Y->X.
    """
    fake_X = G_YtoX(fixed_Y)
    fake_Y = G_XtoY(fixed_X)

    X, fake_X = to_data(fixed_X), to_data(fake_X)
    Y, fake_Y = to_data(fixed_Y), to_data(fake_Y)
    
    temp_path = 'temp/q2/samples/'
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    merged = merge_images(X, fake_Y)
    path = os.path.join(temp_path, 'sample-{:06d}-X-Y.png'.format(iteration))
    merged = (merged + 1)/2 * 255.
    merged = np.uint8(merged)
    imageio.imwrite(path, merged)
    print('Saved {}'.format(path))

    merged = merge_images(Y, fake_X)
    path = os.path.join(temp_path, 'sample-{:06d}-Y-X.png'.format(iteration))
    merged = (merged + 1)/2 * 255.
    merged = np.uint8(merged)
    imageio.imwrite(path, merged)
    print('Saved {}'.format(path))

def show_gan_samples(samples, fname=None, title='GAN Samples'):
    plt.figure()
    plt.imshow(samples)
    plt.axis('off')
    plt.title(title)
    if fname is not None:
        savefig(fname)
    else:
        plt.show()

def plot_dcgan_losses(d_real_loss, d_fake_loss, g_loss, fname=None, title='GAN Training Plot'):
    plt.figure()
    plt.plot(d_real_loss)
    plt.plot(d_fake_loss)
    plt.plot(g_loss)
    plt.legend(["d_real", "d_fake", "generator"])
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    savefig(fname)

def q1_save_results(fn):
    d_real_loss, d_fake_loss, g_loss = fn()
    initial_samples = Image.open("temp/q1/samples/sample-000000.png")
    intermediate_samples = Image.open("temp/q1/samples/sample-000080.png")
    final_samples = Image.open("temp/q1/samples/sample-000149.png")
    
    print("Final discriminator real loss: ", d_real_loss[-1])
    print("Final discriminator fake loss: ", d_fake_loss[-1])
    print("Final generator loss: ", g_loss[-1])
    show_gan_samples(initial_samples, fname='results/q1_initial_samples.png', title='Initial Samples')
    show_gan_samples(intermediate_samples, fname='results/q1_intermediate_samples.png', title='Intermediate Samples')
    show_gan_samples(final_samples, fname='results/q1_final_samples.png', title='Final Samples')
    plot_dcgan_losses(d_real_loss, d_fake_loss, g_loss, fname='results/q1_training_plot.png')

    generate_gif("temp/q1/samples/")
    # GIF(open('temp/q1/samples/anim.gif','rb').read())


def q2_save_results(fn):
    d_real_loss, d_fake_loss, g_loss = fn()
    initial_samples_XY = Image.open("temp/q2/samples/sample-000000-X-Y.png")
    intermediate_samples_XY = Image.open("temp/q2/samples/sample-000020-X-Y.png")
    final_samples_XY = Image.open("temp/q2/samples/sample-000049-X-Y.png")
    
    initial_samples_YX = Image.open("temp/q2/samples/sample-000000-Y-X.png")
    intermediate_samples_YX = Image.open("temp/q2/samples/sample-000020-Y-X.png")
    final_samples_YX = Image.open("temp/q2/samples/sample-000049-Y-X.png")
    
    print("Final discriminator real loss: ", d_real_loss[-1])
    print("Final discriminator fake loss: ", d_fake_loss[-1])
    print("Final generator loss: ", g_loss[-1])
    show_gan_samples(initial_samples_XY, fname='results/q2_initial_samples_xy.png', title='Initial Samples XtoY')
    show_gan_samples(intermediate_samples_XY, fname='results/q2_intermediate_samples_xy.png', title='Intermediate Samples XtoY')
    show_gan_samples(final_samples_XY, fname='results/q2_final_samples_xy.png', title='Final Samples XtoY')

    show_gan_samples(initial_samples_YX, fname='results/q2_initial_samples_yx.png', title='Initial Samples YtoX')
    show_gan_samples(intermediate_samples_YX, fname='results/q2_intermediate_samples_yx.png', title='Intermediate Samples YtoX')
    show_gan_samples(final_samples_YX, fname='results/q2_final_samples_yx.png', title='Final Samples YtoX')

    plot_dcgan_losses(d_real_loss, d_fake_loss, g_loss, fname='results/q2_training_plot.png')
    generate_gif("temp/q2/samples/", keyword='X-Y')
    generate_gif("temp/q2/samples/", keyword='Y-X')
    # GIF(open('temp/q2/samples/anim_X-Y.gif','rb').read())
    # GIF(open('temp/q2/samples/anim_Y-X.gif','rb').read())


# HERE
def load_low_res_mnist():
    mnist = load_digits().data
    len_t = int(.8 * len(mnist))
    train_data = mnist[:len_t].astype(np.float32)
    test_data = mnist[len_t:].astype(np.float32)
    return train_data / 16 * 255 , test_data / 16 * 255

# HERE
def load_fashion_mnist():
    data_dir = get_data_dir(3)
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = FashionMNIST(data_dir, download=True, train=True, transform=transform).data.unsqueeze(3)
    test_data = FashionMNIST(data_dir, download=True, train=False, transform=transform).data.unsqueeze(3)
    pad = transforms.Pad(2)
    train_data = pad(train_data.moveaxis(3, 1)).moveaxis(1, 3)
    test_data = pad(test_data.moveaxis(3, 1)).moveaxis(1, 3)
    return train_data.numpy(), test_data.numpy()

# HERE
def visualize_low_res_mnist():
    train_data, test_data = load_low_res_mnist()    
    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs]
    show_samples(images.reshape(len(images), 8, 8, 1), title='MNIST Samples')

# HERE
def visualize_fashion_mnist():
    train_data, test_data = load_fashion_mnist()    
    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs]
    show_samples(images.reshape(len(images), 32, 32, 1), title='Fahion MNIST Samples')

# HERE
def plot_ddgm_training_plot(train_losses, test_losses, title, fname):
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_losses, label='train')
    plt.plot(x_test, test_losses, label='test')

    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    savefig(fname)


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

def q3_save_results(part, fn):
    if part == 'a':
        train_data, test_data = load_low_res_mnist()
    else:
        train_data, test_data = load_fashion_mnist()

    
    train_data = train_data / 255.
    test_data = test_data / 255.
    train_losses, test_losses, samples = fn(train_data, test_data, part)

    if part == 'a':
        print(f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, '
          f'KL Loss: {test_losses[-1, 2]:.4f}')

        plot_vae_training_plot(train_losses, test_losses, f'Q3({part}) Train Plot',
                           f'results/q3_{part}_train_plot.png')
        
    else:
        print(f'Final loss: {test_losses[-1]:.4f}')

        plot_ddgm_training_plot(train_losses, test_losses, f'Q3({part}) Train Plot',
                           f'results/q3_{part}_train_plot.png')

        samples = samples.reshape(samples.shape[0] * samples.shape[1], \
                    samples.shape[2], samples.shape[3], samples.shape[4]) 

                              
    show_samples(samples, title='Q3 MNIST Samples' if part == 'a' else 'Q3 Fashion Samples',
                    fname=f'results/q3_{part}_samples.png')

# PREVIOUS YEAR

# from .utils import *
# import numpy as np
# import torch.nn as nn
# import torch.utils.data
# import torchvision
# from torchvision import transforms as transforms
# from .hw4_utils.hw4_models import GoogLeNet
# from PIL import Image as PILImage
# import scipy.ndimage
# import cv2
# import deepul.pytorch_util as ptu

# CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# import numpy as np
# import math
# import sys

# softmax = None
# model = None
# device = torch.device("cuda:0")

# def plot_vae_training_plot(train_losses, test_losses, title, fname):
#     elbo_train, recon_train, kl_train = train_losses[:, 0], train_losses[:, 1], train_losses[:, 2]
#     elbo_test, recon_test, kl_test = test_losses[:, 0], test_losses[:, 1], test_losses[:, 2]
#     plt.figure()
#     n_epochs = len(test_losses) - 1
#     x_train = np.linspace(0, n_epochs, len(train_losses))
#     x_test = np.arange(n_epochs + 1)

#     plt.plot(x_train, elbo_train, label='-elbo_train')
#     plt.plot(x_train, recon_train, label='recon_loss_train')
#     plt.plot(x_train, kl_train, label='kl_loss_train')
#     plt.plot(x_test, elbo_test, label='-elbo_test')
#     plt.plot(x_test, recon_test, label='recon_loss_test')
#     plt.plot(x_test, kl_test, label='kl_loss_test')

#     plt.legend()
#     plt.title(title)
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     savefig(fname)


# def sample_data_1_a(count):
#     rand = np.random.RandomState(0)
#     return [[1.0, 2.0]] + (rand.randn(count, 2) * [[5.0, 1.0]]).dot(
#         [[np.sqrt(2) / 2, np.sqrt(2) / 2], [-np.sqrt(2) / 2, np.sqrt(2) / 2]])


# def sample_data_2_a(count):
#     rand = np.random.RandomState(0)
#     return [[-1.0, 2.0]] + (rand.randn(count, 2) * [[1.0, 5.0]]).dot(
#         [[np.sqrt(2) / 2, np.sqrt(2) / 2], [-np.sqrt(2) / 2, np.sqrt(2) / 2]])


# def sample_data_1_b(count):
#     rand = np.random.RandomState(0)
#     return [[1.0, 2.0]] + rand.randn(count, 2) * [[5.0, 1.0]]


# def sample_data_2_b(count):
#     rand = np.random.RandomState(0)
#     return [[-1.0, 2.0]] + rand.randn(count, 2) * [[1.0, 5.0]]


# def q1_sample_data(part, dset_id):
#     assert dset_id in [1, 2]
#     assert part in ['a', 'b']
#     if part == 'a':
#         if dset_id == 1:
#             dset_fn = sample_data_1_a
#         else:
#             dset_fn = sample_data_2_a
#     else:
#         if dset_id == 1:
#             dset_fn = sample_data_1_b
#         else:
#             dset_fn = sample_data_2_b

#     train_data, test_data = dset_fn(10000), dset_fn(2500)
#     return train_data.astype('float32'), test_data.astype('float32')


# def visualize_q1_data(part, dset_id):
#     train_data, test_data = q1_sample_data(part, dset_id)
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     ax1.set_title('Train Data')
#     ax1.scatter(train_data[:, 0], train_data[:, 1])
#     ax2.set_title('Test Data')
#     ax2.scatter(test_data[:, 0], test_data[:, 1])
#     print(f'Dataset {dset_id}')
#     plt.show()


# def q1_a_save_results(part, dset_id, fn):
#     train_data, test_data = q1_sample_data(part, dset_id)
#     train_losses, test_losses, samples_noise, samples_nonoise = fn(train_data, test_data, part, dset_id)
#     print(f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, '
#           f'KL Loss: {test_losses[-1, 2]:.4f}')

#     plot_vae_training_plot(train_losses, test_losses, f'Q1({part}) Dataset {dset_id} Train Plot',
#                            f'results/q1_{part}_dset{dset_id}_train_plot.png')
#     save_scatter_2d(samples_noise, title='Samples with Decoder Noise',
#                     fname=f'results/q1_{part}_dset{dset_id}_sample_with_noise.png')
#     save_scatter_2d(samples_nonoise, title='Samples without Decoder Noise',
#                     fname=f'results/q1_{part}_dset{dset_id}_sample_without_noise.png')


# def visualize_colored_shapes():
#     data_dir = get_data_dir(3)
#     train_data, test_data = load_pickled_data(join(data_dir, 'shapes_colored.pkl'))
#     idxs = np.random.choice(len(train_data), replace=False, size=(100,))
#     images = train_data[idxs]
#     show_samples(images, title='Colored Shapes Samples')


# def visualize_svhn():
#     data_dir = get_data_dir(3)
#     train_data, test_data = load_pickled_data(join(data_dir, 'svhn.pkl'))
#     idxs = np.random.choice(len(train_data), replace=False, size=(100,))
#     images = train_data[idxs]
#     show_samples(images, title='SVHN Samples')


# def visualize_cifar10():
#     data_dir = get_data_dir(3)
#     train_data, test_data = load_pickled_data(join(data_dir, 'cifar10.pkl'))
#     idxs = np.random.choice(len(train_data), replace=False, size=(100,))
#     images = train_data[idxs]
#     show_samples(images, title='CIFAR10 Samples')


# def q2_save_results(part, dset_id, fn):
#     assert part in ['a', 'b'] and dset_id in [1, 2]
#     data_dir = get_data_dir(3)
#     if dset_id == 1:
#         train_data, test_data = load_pickled_data(join(data_dir, 'svhn.pkl'))
#     else:
#         train_data, test_data = load_pickled_data(join(data_dir, 'cifar10.pkl'))

#     train_losses, test_losses, samples, reconstructions, interpolations = fn(train_data, test_data, dset_id)
#     samples, reconstructions, interpolations = samples.astype('float32'), reconstructions.astype('float32'), interpolations.astype('float32')
#     print(f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, '
#           f'KL Loss: {test_losses[-1, 2]:.4f}')
#     plot_vae_training_plot(train_losses, test_losses, f'Q2({part}) Dataset {dset_id} Train Plot',
#                            f'results/q2_{part}_dset{dset_id}_train_plot.png')
#     show_samples(samples, title=f'Q2({part}) Dataset {dset_id} Samples',
#                  fname=f'results/q2_{part}_dset{dset_id}_samples.png')
#     show_samples(reconstructions, title=f'Q2({part}) Dataset {dset_id} Reconstructions',
#                  fname=f'results/q2_{part}_dset{dset_id}_reconstructions.png')
#     show_samples(interpolations, title=f'Q2({part}) Dataset {dset_id} Interpolations',
#                  fname=f'results/q2_{part}_dset{dset_id}_interpolations.png')


# # def q3_save_results(dset_id, fn):
# #     assert dset_id in [1, 2]
# #     data_dir = get_data_dir(3)
# #     if dset_id == 1:
# #         train_data, test_data = load_pickled_data(join(data_dir, 'svhn.pkl'))
# #     else:
# #         train_data, test_data = load_pickled_data(join(data_dir, 'cifar10.pkl'))

# #     vqvae_train_losses, vqvae_test_losses, pixelcnn_train_losses, pixelcnn_test_losses, samples, reconstructions = fn(train_data, test_data, dset_id)
# #     samples, reconstructions = samples.astype('float32'), reconstructions.astype('float32')
# #     print(f'VQ-VAE Final Test Loss: {vqvae_test_losses[-1]:.4f}')
# #     print(f'PixelCNN Prior Final Test Loss: {pixelcnn_test_losses[-1]:.4f}')
# #     save_training_plot(vqvae_train_losses, vqvae_test_losses,f'Q3 Dataset {dset_id} VQ-VAE Train Plot',
# #                        f'results/q3_dset{dset_id}_vqvae_train_plot.png')
# #     save_training_plot(pixelcnn_train_losses, pixelcnn_test_losses,f'Q3 Dataset {dset_id} PixelCNN Prior Train Plot',
# #                        f'results/q3_dset{dset_id}_pixelcnn_train_plot.png')
# #     show_samples(samples, title=f'Q3 Dataset {dset_id} Samples',
# #                  fname=f'results/q3_dset{dset_id}_samples.png')
# #     show_samples(reconstructions, title=f'Q3 Dataset {dset_id} Reconstructions',
# #                  fname=f'results/q3_dset{dset_id}_reconstructions.png')


# def q4_a_save_results(dset_id, fn):
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
#     save_training_plot(vqvae_train_losses, vqvae_test_losses,f'Q4(a) Dataset {dset_id} VQ-VAE Train Plot',
#                        f'results/q4_a_dset{dset_id}_vqvae_train_plot.png')
#     save_training_plot(pixelcnn_train_losses, pixelcnn_test_losses,f'Q4(a) Dataset {dset_id} PixelCNN Prior Train Plot',
#                        f'results/q4_a_dset{dset_id}_pixelcnn_train_plot.png')
#     show_samples(samples, title=f'Q4(a) Dataset {dset_id} Samples',
#                  fname=f'results/q4_a_dset{dset_id}_samples.png')
#     show_samples(reconstructions, title=f'Q4(a) Dataset {dset_id} Reconstructions',
#                  fname=f'results/q4_a_dset{dset_id}_reconstructions.png')


# def q4_b_save_results(fn):
#     part = 'b'
#     data_dir = get_data_dir(3)
#     train_data, test_data = load_pickled_data(join(data_dir, 'mnist.pkl'))

#     train_losses, test_losses, samples, reconstructions = fn(train_data, test_data)
#     samples, reconstructions = samples.astype('float32') * 255, reconstructions.astype('float32') * 255
#     print(f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, '
#           f'KL Loss: {test_losses[-1, 2]:.4f}')
#     plot_vae_training_plot(train_losses, test_losses, f'Q4({part}) Train Plot',
#                            f'results/q4_{part}_train_plot.png')
#     show_samples(samples, title=f'Q4({part}) Samples',
#                  fname=f'results/q4_{part}_samples.png')
#     show_samples(reconstructions, title=f'Q4({part}) Reconstructions',
#                  fname=f'results/q4_{part}_reconstructions.png')

# ###########
# ###########
# ## GANS
    
# def plot_gan_training(losses, title, fname):
#     plt.figure()
#     n_itr = len(losses)
#     xs = np.arange(n_itr)

#     plt.plot(xs, losses, label='loss')
#     plt.legend()
#     plt.title(title)
#     plt.xlabel('Training Iteration')
#     plt.ylabel('Loss')
#     savefig(fname)

# def q1_gan_plot(data, samples, xs, ys, title, fname):
#     plt.figure()
#     plt.hist(samples, bins=50, density=True, alpha=0.7, label='fake')
#     plt.hist(data, bins=50, density=True, alpha=0.7, label='real')

#     plt.plot(xs, ys, label='discrim')
#     plt.legend()
#     plt.title(title)
#     savefig(fname)
    
# def q1_data(n=20000):
#     assert n % 2 == 0
#     gaussian1 = np.random.normal(loc=-1, scale=0.25, size=(n//2,))
#     gaussian2 = np.random.normal(loc=0.5, scale=0.5, size=(n//2,))
#     data = (np.concatenate([gaussian1, gaussian2]) + 1).reshape([-1, 1])
#     scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
#     return 2 * scaled_data -1

# def visualize_q1_dataset():
#     data = q1_data()
#     plt.hist(data, bins=50, alpha=0.7, label='train data')
#     plt.legend()
#     plt.show()


# def q1_b_save_results(part, fn):
#     data = q1_data()
#     losses, samples1, xs1, ys1, samples_end, xs_end, ys_end = fn(data)
#     part = 'b'
#     # loss plot
#     plot_gan_training(losses, 'Q1{} Losses'.format(part), 'results/q1{}_losses.png'.format(part))

#     # samples
#     q1_gan_plot(data, samples1, xs1, ys1, 'Q1{} Epoch 1'.format(part), 'results/q1{}_epoch1.png'.format(part))
#     q1_gan_plot(data, samples_end, xs_end, ys_end, 'Q1{} Final'.format(part), 'results/q1{}_final.png'.format(part))
    
    
# def load_q3_data():
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#     train_data = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
#     test_data = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
#     return train_data, test_data

# def visualize_q3_data():
#     train_data, _ = load_q3_data()
#     imgs = train_data.data[:100]
#     show_samples(imgs.reshape([100, 28, 28, 1]) * 255.0, title=f'MNIST samples')

# def plot_q3_supervised(pretrained_losses, random_losses, title, fname):
#     plt.figure()
#     xs = np.arange(len(pretrained_losses))
#     plt.plot(xs, pretrained_losses, label='bigan')
#     xs = np.arange(len(random_losses))
#     plt.plot(xs, random_losses, label='random init')
#     plt.legend()
#     plt.title(title)
#     savefig(fname)

# def q3_save_results(fn):
#     train_data, test_data = load_q3_data()
#     gan_losses, samples, reconstructions, pretrained_losses, random_losses = fn(train_data, test_data)

#     plot_gan_training(gan_losses, 'Q3 Losses', 'results/q3_gan_losses.png')
#     plot_q3_supervised(pretrained_losses, random_losses, 'Linear classification losses', 'results/q3_supervised_losses.png')
#     show_samples(samples * 255.0, fname='results/q3_samples.png', title='BiGAN generated samples')
#     show_samples(reconstructions * 255.0, nrow=20, fname='results/q3_reconstructions.png', title=f'BiGAN reconstructions')
#     print('BiGAN final linear classification loss:', pretrained_losses[-1])
#     print('Random encoder linear classification loss:', random_losses[-1])
    
# def get_colored_mnist(data):
#     # from https://www.wouterbulten.nl/blog/tech/getting-started-with-gans-2-colorful-mnist/
#     # Read Lena image
#     lena = PILImage.open('deepul/deepul/hw4_utils/lena.jpg')

#     # Resize
#     batch_resized = np.asarray([scipy.ndimage.zoom(image, (2.3, 2.3, 1), order=1) for image in data])

#     # Extend to RGB
#     batch_rgb = np.concatenate([batch_resized, batch_resized, batch_resized], axis=3)

#     # Make binary
#     batch_binary = (batch_rgb > 0.5)

#     batch = np.zeros((data.shape[0], 28, 28, 3))

#     for i in range(data.shape[0]):
#         # Take a random crop of the Lena image (background)
#         x_c = np.random.randint(0, lena.size[0] - 64)
#         y_c = np.random.randint(0, lena.size[1] - 64)
#         image = lena.crop((x_c, y_c, x_c + 64, y_c + 64))
#         image = np.asarray(image) / 255.0

#         # Invert the colors at the location of the number
#         image[batch_binary[i]] = 1 - image[batch_binary[i]]

#         batch[i] = cv2.resize(image, (0, 0), fx=28 / 64, fy=28 / 64, interpolation=cv2.INTER_AREA)
#     return batch.transpose(0, 3, 1, 2)

# def load_q4_data():
#     train, _ = load_q3_data()
#     mnist = np.array(train.data.reshape(-1, 28, 28, 1) / 255.0)
#     colored_mnist = get_colored_mnist(mnist)
#     return mnist.transpose(0, 3, 1, 2), colored_mnist

# def visualize_cyclegan_datasets():
#     mnist, colored_mnist = load_q4_data()
#     mnist, colored_mnist = mnist[:100], colored_mnist[:100]
#     show_samples(mnist.reshape([100, 28, 28, 1]) * 255.0, title=f'MNIST samples')
#     show_samples(colored_mnist.transpose([0, 2, 3, 1]) * 255.0, title=f'Colored MNIST samples')

# def q4_save_results(fn):
#     mnist, cmnist = load_q4_data()

#     m1, c1, m2, c2, m3, c3 = fn(mnist, cmnist)
#     m1, m2, m3 = m1.repeat(3, axis=3), m2.repeat(3, axis=3), m3.repeat(3, axis=3)
#     mnist_reconstructions = np.concatenate([m1, c1, m2], axis=0)
#     colored_mnist_reconstructions = np.concatenate([c2, m3, c3], axis=0)

#     show_samples(mnist_reconstructions * 255.0, nrow=20,
#                  fname='figures/q4_mnist.png',
#                  title=f'Source domain: MNIST')
#     show_samples(colored_mnist_reconstructions * 255.0, nrow=20,
#                  fname='figures/q4_colored_mnist.png',
#                  title=f'Source domain: Colored MNIST')
#     pass
