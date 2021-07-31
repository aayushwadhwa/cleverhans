import copy
import numpy as np
import tensorflow as tf
from cleverhans.tf2.utils import compute_gradient

def deep_fool_attack(model, input_img_batch, num_classes=10, overshoot=0.02, max_itr=50):
  """
  Deepfool attack (https://arxiv.org/pdf/1511.04599.pdf), proposed by 
  
  :param model: any NN that produces result before apply softmax
  :param input_img_batch: batch of all the input images
  :param num_classes: number of classes to check while producing attack image
  :param max_iter: maximum number of iterations for deepfool (default = 50)
  :return: perturbed image and minimal perturbation that fools the classifier

  """

  f_batch = model(input_img_batch).numpy()
  f_batch_labels = np.argsort(-f_batch)[:,:num_classes]
  labels = f_batch_labels[:, :1].flatten()

  input_shape = input_img_batch.shape
  pert_image_batch = copy.deepcopy(input_img_batch)
  backup = copy.deepcopy(input_img_batch)
  w = np.zeros(input_shape)
  r_tot = np.zeros(input_shape)

  noise = np.zeros(input_shape)

  i, itr = 0, 0
  k = labels.copy()
  x = tf.Variable(backup[i])
  x = tf.expand_dims(x, axis=0)
  fs = model(x)

  def loss_func(labels, logits):
      return logits[0, labels]
    
    
  while i < input_shape[0]:
    x = tf.Variable(backup[i])
    while k[i] == labels[i] and itr < max_itr:
      x = tf.expand_dims(x, axis=0)

      pert = np.inf
      grad_orig = compute_gradient(model, loss_func, x, f_batch_labels[i][0], False)

      for j in range(1, num_classes):
        curr_grad = compute_gradient(model, loss_func, x, f_batch_labels[i][j], False)
        
        w_k = curr_grad - grad_orig
        f_k = (fs[0, f_batch_labels[i][j]] - fs[0, f_batch_labels[i][0]]).numpy()
        pert_k = abs(f_k) / np.linalg.norm(tf.reshape(w_k, [-1]))

        if pert_k < pert:
          pert = pert_k
          w[i] = w_k

      r_i = (pert + 1e-4) * w[i] / np.linalg.norm(w[i])

      r_tot[i] = np.float32(r_tot[i] + r_i)
      pert_image_batch[i] = input_img_batch[i] + (1 + overshoot) * r_tot[i]

      x = tf.Variable(pert_image_batch[i])
      noise[i] = (1 + overshoot) * r_tot[i]
      fs = model(tf.expand_dims(x, axis=0))
      k[i] = np.argmax(np.array(fs).flatten())
      itr += 1
    r_tot[i] = (1 + overshoot) * r_tot[i]
    i += 1

  return pert_image_batch, noise