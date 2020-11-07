import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import argparse

import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.backend as b
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from network.utils import load_dataset, load_model_arch, load_teacher,lr_scheduler
from utils import load_DI

class TrainStudent(object):
    def __init__(self,dataset_name, teacher_path, student_name, batch_size,epochs, data_dir,  save_dir , metrics='accuracy'):
        self.dataset_name= dataset_name
        self.batch_size=batch_size
        self.epochs= epochs
        self.student_name=student_name
        self.save_dir= save_dir
        
        #load train dataset from DeepInversion
        self.datagen= load_DI(data_dir+dataset_name+'/',self.batch_size)

        #load validation dataset
        self.x_train,self.y_train,self.x_test, self.y_test,self.num_classes , self.img_shape = load_dataset(self.dataset_name, self.batch_size)

        self.mean_image = np.mean(self.x_train, axis=0)


        #load teacher model
        self.teacher=load_teacher(teacher_path)

        #load student model architecture
        self.student=load_model_arch(self.student_name,self.img_shape,self.num_classes)
        
        self.train_loss= k.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)
        self.test_loss = k.losses.CategoricalCrossentropy()
        self.test_acc= k.metrics.BinaryAccuracy()
        self.optimizer = Adam(learning_rate=lr_scheduler(0))
        

    def train(self):
        
        #reconstruct output of teacher and student models for Knowledge Distillation (KD)
        self.teacher_rec = k.models.Model(inputs=self.teacher.input, outputs=[self.teacher.layers[-2].output,self.teacher.layers[-1].output])
        self.student_rec = k.models.Model(inputs=self.student.input, outputs=[self.student.layers[-2].output,self.student.layers[-1].output])
        for epoch in range(self.epochs):
            #train student via KD

            batches=0
            train_loss_mean = 0.0
            for x_batch, y_batch in self.datagen:

                x_batch =(x_batch / 255.)- self.mean_image

                with tf.GradientTape() as tape:

                    #extract logit scores for teacher and student models
                    output_t=self.teacher_rec(x_batch)[0]
                    output_s=self.student_rec(x_batch,training=True)[0]

                    train_loss=self.train_loss(output_t,output_s)
                gradients= tape.gradient(train_loss, self.student_rec.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.student_rec.trainable_variables))
                train_loss_mean+= train_loss
                batches+=1
                if batches >= self.datagen.__len__():
                    # we need to break the loop by hand because
                    # the generator loops indefinitely
                    break
            self.optimizer=Adam(learning_rate=lr_scheduler(epoch))
            train_loss_mean = train_loss_mean/ float(batches)


            #test student performance
            test_loss_mean = 0.0
            test_acc_mean=0.0

            batches=0
            for x_batch, y_batch in tf.data.Dataset.from_tensor_slices((self.x_test,self.y_test)).batch(self.batch_size):
                output_s =self.student_rec(x_batch)[1]

                test_loss=self.test_loss(y_batch,output_s)
                
                test_loss_mean+= test_loss
                test_acc=self.test_acc.update_state(y_batch, output_s)
                
                batches+=1
                if batches >= len(self.x_test):
                    break
            test_acc_mean=self.test_acc.result().numpy()
            test_loss_mean= test_loss_mean / float(batches)

            print(f'[{epoch}/{self.epochs}] Train loss: {train_loss}\t ')
            print(f'[{epoch}/{self.epochs}] Test loss: {test_loss_mean} Test acc: {test_acc_mean}')            
    
        #reconstruct output of student model
        self.student=k.models.Model(inputs=self.student_rec.input, outputs=self.student_rec.layers[-1].output)

        #save student model
        tf.keras.models.save_model(self.student, self.save_dir+'student_'+self.dataset_name+'_'+self.student_name+'.h5', save_format='h5')

    def test(self):
        self.student.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        scores =self.student.evaluate(self.val_dataset,batch_size=self.batch_size)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

    def build(self):

        self.train()
        self.test()



def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='Dataset [ "cifar10", "cifar100" ] ')
    parser.add_argument('--teacher_path', type=str, default='./saved_models/cifar10_resnet34.h5', help='teacher model path')

    parser.add_argument('--student_name', type=str, default='resnet18', help='student network name')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Epoch')
    parser.add_argument('--data_dir', type=str, default='./results/', help='Saved model path')
    parser.add_argument('--save_dir', type=str, default='./saved_models/', help='Saved model path')


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['TF2_BEHAVIOR'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES']= '0'

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    
    args = parser.parse_args()

    trainer=TrainStudent(dataset_name=args.dataset_name,
                        teacher_path=args.teacher_path, 
                        student_name=args.student_name, 
                        batch_size= args.batch_size,
                        epochs=args.epochs, 
                        data_dir=args.data_dir,
                        save_dir=args.save_dir,
                        metrics='accuracy'
                        )

    trainer.train()



if __name__ == '__main__':

    main()