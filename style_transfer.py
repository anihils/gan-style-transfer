"""
Implements a style transfer in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

import torch
import torch.nn as nn
from ps4_helper import *

def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from style_transfer.py!')

# 5 points
def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.
    
    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_original: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

    Returns:
    - scalar content loss
    """
    ############################################################################
    # TODO: Compute the content loss for style transfer.                       #
    ############################################################################

    # Orginal implementation: 0 rel error but high absolute error
    content_loss = content_weight * \
      torch.sum(torch.square(content_current - content_original))

    return content_loss

    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################

# 9 points
def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: PyTorch Tensor of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Tensor of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    gram = None
    ############################################################################
    # TODO: Compute the Gram matrix from features.                             #
    # Don't forget to implement for both normalized and non-normalized version #
    ############################################################################

    n, c, h, w = features.shape
    
    gram = torch.zeros((n, c, c), device = features.device)

    # Working implementation: one loop
    # for i, feat in enumerate(features):
    #   feature = torch.reshape(feat, (c, h*w))
    #   gram[i] = torch.mm(feature, torch.t(feature))
    
    features = torch.reshape(features, (n, c, h*w))
    features_t = features.permute(0, 2, 1)
    gram = torch.matmul(features, features.permute(0, 2, 1))

    if normalize:
      gram /= (c * h * w)
   
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return gram

# 9 points
def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    ############################################################################
    # TODO: Computes the style loss at a set of layers.                        #
    # Hint: you can do this with one for loop over the style layers, and       #
    # should not be very much code (~5 lines).                                 #
    # You will need to use your gram_matrix function.                          #
    ############################################################################
    
    style_loss = 0
    for i, layer in enumerate(style_layers):
      feat_target = gram_matrix(feats[layer])
      difference = torch.square(feat_target - style_targets[i])
      style_loss += (torch.sum(difference) * style_weights[i])

    return style_loss
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################

# 8 points
def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    ############################################################################
    # TODO: Compute total variation loss.                                      #
    # Your implementation should be vectorized and not require any loops!      #
    ############################################################################
    
    loss = 0
    n, c, h, w = img.shape

    horizontal_diff = torch.sum(torch.square(img[0,:,1:,:] - img[0,:,:h-1,:]))
    vertical_diff = torch.sum(torch.square(img[0,:,:,1:] - img[0,:,:,:w-1]))

    loss = tv_weight * (horizontal_diff + vertical_diff)

    return loss
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################

# 10 points
def guided_gram_matrix(features, masks, normalize=True):
  """
  Inputs:
    - features: PyTorch Tensor of shape (N, R, C, H, W) giving features for
      a batch of N images.
    - masks: PyTorch Tensor of shape (N, R, H, W)
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Tensor of shape (N, R, C, C) giving the
      (optionally normalized) guided Gram matrices for the N input images.
  """
  guided_gram = None
  ##############################################################################
  # TODO: Compute the guided Gram matrix from features.                        #
  # Apply the regional guidance mask to its corresponding feature and          #
  # calculate the Gram Matrix. You are allowed to use one for-loop in          #
  # this problem.                                                              #
  ##############################################################################

  N, R, C, H, W = features.shape
  guided_gram = torch.zeros((N, R, C, C), device = features.device)

  # first implementation: error - 0.38
  # features = torch.reshape(features, (N, R, C, H*W)) # 1, 2, 3, 9
  # masks = masks.repeat(1, 1, 1, C) # Repeating mask for 3 channels - 1, 2, 3, 9

  # second implementation: error - 0.27
  # features = torch.reshape(features, (N, R, H*W, C)) # 1, 2, 3, 9
  # masks = masks.repeat(1, 1, C, 1) # Repeating mask for 3 channels - 1, 2, 3, 9
    
  # for r in range(R):
  #   guided_features = masks[:, r] * features[:, r] # 1, 3, 9
  #   guided_features_t = guided_features.permute(0, 2, 1)
  #   output = torch.matmul(guided_features_t, guided_features)
  #   guided_gram[:, r] = output

  guided_features = torch.unsqueeze(masks, 2) * features
  guided_features = torch.reshape(guided_features, (N, R, C, H*W))
  guided_features_t = guided_features.permute(0, 1, 3, 2)
  guided_gram = torch.matmul(guided_features, guided_features_t)

  if normalize: # normalise gram
    guided_gram /= (C * H * W)

  return guided_gram

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

# 9 points
def guided_style_loss(feats, style_layers, style_targets, style_weights, content_masks):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the guided Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
    - content_masks: List of the same length as feats, giving a binary mask to the
      features of each layer.
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    ############################################################################
    # TODO: Computes the guided style loss at a set of layers.                 #
    ############################################################################
    
    style_loss = 0
    for i, layer in enumerate(style_layers):
      feat_target = guided_gram_matrix(feats[layer], content_masks[layer])
      difference = torch.square(feat_target - style_targets[i])
      style_loss += (torch.sum(difference) * style_weights[i])

    return style_loss

    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
