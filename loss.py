import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.backend as b


def BN_loss(inputs, t_layer,rescale):

    sum=0
    for i in range(len(t_layer)):

        #get running mean and variance from batch normalization layer
        running_mean=t_layer[i].moving_mean
        running_var=t_layer[i].moving_variance

        in_mean=b.mean(inputs[i], [0,1,2])
        in_var=b.var(inputs[i], [0,1,2])

        sum += rescale[i] *(tf.norm(running_var - in_var ) + tf.norm(running_mean - in_mean) )
    return sum 

def ADI_loss(student,outputs_t, inputs_jit):

    kl_loss = k.losses.KLDivergence(reduction=k.losses.Reduction.SUM_OVER_BATCH_SIZE)
    #loss_verifier_cig = torch.zeros(1)
       
    outputs_s = student(inputs_jit)

    T = 3.0
    if 1:
        T = 3.0
        # Jensen Shanon divergence:
        # another way to force KL between negative probabilities
        P = k.activations.softmax(outputs_s / T, dim=1)
        Q = k.activations.softmax(outputs_t / T, dim=1)
        M = 0.5 * (P + Q)

        P = tf.clip_by_value(P, clip_value_min=0.01, clip_value_max=0.99)
        Q = tf.clip_by_value(Q, clip_value_min=0.01, clip_value_max=0.99)
        M = tf.clip_by_value(M, clip_value_min=0.01, clip_value_max=0.99)
        eps = 0.0
        loss_verifier_cig = 0.5 * kl_loss(b.log(P + eps), M) + 0.5 * kl_loss(b.log(Q + eps), M)
            # JS criteria - 0 means full correlation, 1 - means completely different
        loss_verifier_cig = 1.0 - tf.clip_by_value(loss_verifier_cig,clip_value_min= 0.0, clip_value_max=1.0)

    return loss_verifier_cig




def TV_loss(inputs_jit):
    diff1 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff2 = inputs_jit[:, :-1, :, :] - inputs_jit[:, 1:, :, :]
    diff3 = inputs_jit[:, 1:, :-1, :] - inputs_jit[:, :-1, 1:, :]
    diff4 = inputs_jit[:, :-1, :-1, :] - inputs_jit[:, 1:, 1:, :]

    #print(tf.norm(diff1,ord='euclidean'))
    loss_var_l2 = tf.norm(diff1,ord='euclidean') + tf.norm(diff2,ord='euclidean') + tf.norm(diff3,ord='euclidean') + tf.norm(diff4,ord='euclidean')
    loss_var_l1 = tf.keras.backend.mean(tf.math.abs(diff1) / 255.0) \
                + tf.keras.backend.mean(tf.math.abs(diff2) / 255.0) \
                + tf.keras.backend.mean(tf.math.abs(diff3) / 255.0) \
                + tf.keras.backend.mean(tf.math.abs(diff4) / 255.0)
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2
