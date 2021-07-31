import copy
import numpy as np
import tensorflow as tf
from cleverhans.tf2.utils import compute_gradient

# My Implementation
def deep_fool_attack(model, input_img_batch, num_classes=10, overshoot=0.02, max_itr=50):

  # getting top 10 labels of each image in the batch and storing it as numpy array
  f_batch = model(input_img_batch).numpy()
  f_batch_labels = np.argsort(-f_batch)[:,:num_classes]
  labels = f_batch_labels[:, :1].flatten()

  # Copy input image and initialize w and r
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

#   def loss_func(logits, I, k):
#     return logits[0, I[k]]

  def loss_func(labels, logits):
      return logits[0, labels]
    
    
  # Start loop for each image and change its label
  while i < input_shape[0]:
    x = tf.Variable(backup[i])
    while k[i] == labels[i] and itr < max_itr:
      x = tf.expand_dims(x, axis=0)

      pert = np.inf
    #   with tf.GradientTape() as tape:
    #     tape.watch(x)
    #     fs = model(x)
    #     loss_value = loss_func(fs, f_batch_labels[i], 0)
      
    #   grad_orig = tape.gradient(loss_value, x)

      grad_orig = compute_gradient(model, loss_func, x, f_batch_labels[i][0], False)

      for j in range(1, num_classes):
        # with tf.GradientTape() as tape:
        #   tape.watch(x)
        #   fs = model(x)
        #   loss_value = loss_func(fs, f_batch_labels[i], j)
        # curr_grad = tape.gradient(loss_value, x)
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