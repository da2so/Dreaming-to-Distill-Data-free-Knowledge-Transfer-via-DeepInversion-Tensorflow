import time
import random
import numpy as np
import sys
import os

import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.backend as b

from utils import load_model, data_info
from loss import TV_loss, BN_loss, ADI_loss

class Deep_Inversion():
    def __init__(self,dataset,t_model_path,adi_coeff,s_model_path, r_feature, tv_l1, \
        tv_l2, l2 ,lr , n_iters, first_bn_mul, main_mul ,bs, jitter, random_label, save_path):
        
        #load dataset information
        self.dataset=dataset
        self.num_classes,self.target_label,self.img_resol, self.lower_res =data_info(self.dataset)
        self.jitter=jitter
        self.r_feature=r_feature
        self.tv_l1=tv_l1
        self.tv_l2=tv_l2
        self.l2=l2
        self.lr=lr
        self.n_iters=n_iters
        self.first_bn_mul=first_bn_mul
        self.main_mul=main_mul
        self.bs=bs
        self.adi_coeff=adi_coeff
        self.random_label=random_label
        self.save_path=save_path
        
        #for one-hot loss
        self.criterion=k.losses.CategoricalCrossentropy()

        #load teacher model
        self.teacher=load_model(t_model_path)

        #find batch normalization layers
        self.t_layer_input=[]
        self.t_layer=[]
        for layer in self.teacher.layers:
            if 'BatchNormalization' in str(layer):
                self.t_layer.append(layer)
                self.t_layer_input.append(layer.input)

        #find a output layer
        self.t_layer_input.append(layer.output)
        self.teacher_rec = k.models.Model(inputs=self.teacher.input, outputs=self.t_layer_input)

        #load student model
        if self.adi_coeff != 0.0:
            self.student=load_model(s_model_path)


    def build(self):
        tf.random.set_seed(int(time.time()))

        #get target labels
        if self.random_label  == True:
            t_label=tf.random.uniform(shape=[self.bs], maxval=self.num_classes, dtype=tf.int64)
        else:
            t_label=self.target_label
            t_label=tf.constant(t_label, dtype=tf.int64)

        #one hot encoding for target labels
        t_label= tf.one_hot(t_label, self.num_classes)
        
        
        inputs = tf.random.normal((self.bs, self.img_resol, self.img_resol, 3),0, 1)
        inputs = tf.clip_by_value(inputs, -1, 1)

        lim_0, lim_1 = self.jitter // (self.lower_res), self.jitter // (self.lower_res)

        for i_iter in range(self.n_iters+1):

            # apply random jitter offsets
            off1 = random.randint(-lim_0, lim_0)
            off2 = random.randint(-lim_1, lim_1)

            with tf.GradientTape() as tape:
                tape.watch(inputs)
                inputs_jit = tf.roll(inputs, shift=(off1,off2), axis=(1,2))

                outputs_comb = self.teacher_rec(inputs_jit)
                outputs_in_layer = outputs_comb[:-1]
                outputs_softmax = outputs_comb[-1]

                pred_label=tf.argmax(outputs_softmax[0,:])

                #one-hot loss
                l_oh=self.criterion(outputs_softmax, t_label)
                
                #tv loss
                l_var_l1, l_var_l2 =TV_loss(inputs_jit)

                #bn loss
                rescale = [self.first_bn_mul] + [1. for _ in range(len(self.t_layer)-1)]
                l_bn=BN_loss(outputs_in_layer, self.t_layer,rescale)

                #l2 loss
                l_l2=b.mean( tf.norm(b.reshape(inputs_jit, (self.bs, -1)), axis=1))

                loss_aux= self.tv_l2 * l_var_l2 + \
                        self.tv_l1 * l_var_l1 + \
                        self.r_feature * l_bn + \
                        self.l2 * l_l2

                l_sum = self.main_mul * l_oh + loss_aux
                if self.adi_coeff != 0.0:
                    l_adi=ADI_loss(self.student, outputs, inputs_jit)
                    l_sum= self.adi_coeff*l_adi +self.main_mul * l_oh + loss_aux

            gradients = tape.gradient(l_sum, inputs)
            gradients /= tf.math.reduce_std(gradients) + 1e-8 
        
            inputs = inputs - gradients*self.lr
            inputs = tf.clip_by_value(inputs, -1, 1)



            if i_iter %  100== 0:
                if self.adi_coeff != 0.0:
                    print(f'[{i_iter}/{self.n_iters}] Loss: One-hot: {l_oh}\t TV_L1: {l_var_l1}\t TV_L2: {l_var_l2}\tBN: {l_bn} \t L2: {l_l2} ADI: {l_adi}')
                else:
                    print(f'[{i_iter}/{self.n_iters}] Loss: One-hot: {l_oh}\t TV_L1: {l_var_l1}\t TV_L2: {l_var_l2}\t BN: {l_bn}\t L2: {l_l2}')
                
                save_path=self.save_path+'/'+self.dataset+'/'+str(i_iter // 100)+'/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                for bs in range(self.bs):
                    k.preprocessing.image.save_img(f'{save_path}{str(bs)}.png', inputs[bs,:,:,:])



