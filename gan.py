from __future__ import print_function

import torch
from torch.nn.functional import conv2d
import torch.utils.data
from torch import max_pool2d, nn, optim
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

NOISE_DIM = 96


def hello_gan():
    print("Hello from gan.py!")


def sample_noise(batch_size, noise_dim, dtype=torch.float, device="cpu"):
    """
    Generate a PyTorch Tensor of random noise from Gaussian distribution.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - noise_dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, noise_dim) containing
      noise from a Gaussian distribution.
    """
    noise = None
    ##############################################################################
    # TODO: Implement sample_noise.                                              #
    ##############################################################################
    
    noise = torch.randn(size=[batch_size, noise_dim], device=device, dtype=dtype)
    
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################

    return noise


def discriminator():
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement discriminator.                                           #
    ############################################################################
    
    model = nn.Sequential(
        nn.Linear(784, 400, bias=True),
        nn.LeakyReLU(0.05),
        nn.Linear(400, 200, bias=True),
        nn.LeakyReLU(0.05),
        nn.Linear(200, 100, bias=True),
        nn.LeakyReLU(0.05),
        nn.Linear(100, 1, bias=True))

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement generator.                                               #
    ############################################################################
    
    model = nn.Sequential(
        nn.Linear(noise_dim, 128, bias=True),
        nn.ReLU(),
        nn.Linear(128, 256, bias=True),
        nn.ReLU(),
        nn.Linear(256, 512, bias=True),
        nn.ReLU(),
        nn.Linear(512, 784, bias=True),
        nn.Tanh())

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement discriminator_loss.                                        #
    ##############################################################################
    
    input_size = logits_real.shape
    real_labels = torch.ones(input_size, device=logits_real.device, dtype=logits_real.dtype)
    fake_labels = torch.zeros(input_size, device=logits_fake.device, dtype=logits_fake.dtype)

    real_loss = nn.functional.binary_cross_entropy_with_logits(logits_real, real_labels)
    fake_loss = nn.functional.binary_cross_entropy_with_logits(logits_fake, fake_labels)  
    loss = real_loss + fake_loss

    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement generator_loss.                                            #
    ##############################################################################
    
    size = logits_fake.shape
    fake_labels = torch.ones(size, device=logits_fake.device, dtype=logits_fake.dtype)

    loss = nn.functional.binary_cross_entropy_with_logits(logits_fake, fake_labels)

    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = None
    ##############################################################################
    # TODO: Implement optimizer.                                                 #
    ##############################################################################

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))

    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return optimizer


def run_a_gan(D, G, D_solver, G_solver, loader_train, discriminator_loss, generator_loss, device, show_images, plt, show_every=250, 
              batch_size=128, noise_size=96, num_epochs=10):
  """
  Train a GAN!
  
  Inputs:
  - D, G: PyTorch models for the discriminator and generator
  - D_solver, G_solver: torch.optim Optimizers to use for training the
    discriminator and generator.
  - loader_train: the dataset used to train GAN
  - discriminator_loss, generator_loss: Functions to use for computing the generator and
    discriminator loss, respectively.
  - show_every: Show samples after every show_every iterations.
  - batch_size: Batch size to use for training.
  - noise_size: Dimension of the noise to use as input to the generator.
  - num_epochs: Number of epochs over the training dataset to use for training.
  """
  iter_count = 0
  for epoch in range(num_epochs):
    for x, _ in loader_train:
      if len(x) != batch_size:
        continue
      ##############################################################################
      # TODO: Implement an iteration of training the discriminator.                #
      # Replace 'pass' with your code.                                             #
      # Save the overall discriminator loss in the variable 'd_total_error',       #
      # which will be printed after every 'show_every' iterations.                 #
      #                                                                            #    
      # IMPORTANT: make sure to pre-process your real data (real images),          #
      # so as to make it in the range [-1,1].                                      #
      ##############################################################################

      D_solver.zero_grad()
      
      # Real data
      x = torch.tanh(x) # preprocess real data(x) to range [-1, 1]
      n, c, h, w = x.shape
      x = x.reshape((n * c, h * w)) # reshaping x to feed through D model
      # TODO: Wrong reshape? n, c*h*w?
      logits_real = D(x.to(device))

      # Fake data
      # Generating noise and creating fake images with G, feeding to D
      noise = sample_noise(batch_size, noise_size)
      fake_images = G(noise.to(device)).detach() # Detach works here
      logits_fake = D(fake_images.to(device)) 

      d_total_error = discriminator_loss(logits_real, logits_fake)

      d_total_error.backward()
      D_solver.step()

      ##############################################################################
      #                              END OF YOUR CODE                              #
      ##############################################################################        


      ##############################################################################
      # TODO: In the same iteration, implement training of the generator now   .   #
      # Replace 'pass' with your code.                                             #
      # Save the generator loss in the variable 'g_error', which will be printed.  #
      # after every 'show_every' iterations, and save the fake images generated    #
      # by G in the variable 'fake_images', which will be used to visualize the    #
      # generated images.
      ##############################################################################
      
      G_solver.zero_grad()

      # Generating new noise and creating fake images with G, feeding to D
      noise = sample_noise(batch_size, noise_size)
      fake_images = G(noise.to(device))
      logits_fake = D(fake_images.to(device))

      g_error = generator_loss(logits_fake)

      g_error.backward()
      G_solver.step()

      ##############################################################################
      #                              END OF YOUR CODE                              #
      ############################################################################## 
  
      if (iter_count % show_every == 0):
        print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_total_error.item(),g_error.item()))
        imgs_numpy = fake_images.data.cpu()#.numpy()
        show_images(imgs_numpy[0:16])
        plt.show()
        print()
      iter_count += 1
    if epoch == num_epochs - 1:
      return imgs_numpy    




def build_dc_classifier():
    """
    Build and return a PyTorch nn.Sequential model for the DCGAN discriminator
    implementing the architecture in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_classifier.                                     #
    ############################################################################
    
    model = nn.Sequential(
      nn.Unflatten(1, (1, 28, 28)), # Unflatten [n, 784] to [n, 1, 28, 28]
      nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1),
      nn.LeakyReLU(negative_slope=0.01),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
      nn.LeakyReLU(negative_slope=0.01),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Flatten(),
      nn.Linear(in_features=4*4*64, out_features=4*4*64),
      nn.LeakyReLU(negative_slope=0.01),
      nn.Linear(in_features=4*4*64, out_features=1)
    )

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def build_dc_generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the DCGAN
    generator using the architecture described in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_generator.                                      #
    ############################################################################

    model = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7*7*128),
            nn.ReLU(),
            nn.BatchNorm1d(7*7*128),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.Flatten(),
        )

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model
