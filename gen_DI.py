
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from DeepInversion import Deep_Inversion

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imagenet', help='dataset ["imagenet", "cifar10", "cifar100"] ')
    parser.add_argument('--t_model_path', type=str, default='resnet50v2', help='Teacher model path')
    parser.add_argument('--adi_coeff', type=float, default=0.0, help='Coefficient for Adaptive Deep Inversion')
    parser.add_argument('--s_model_path', type=str, default='resnet34', help='Student model path')
    parser.add_argument('--n_iters', default=3000, type=int ,help='iterations')
    parser.add_argument('--bs', default=32, type=int, help='batch size')
    parser.add_argument('--jitter', default=30, type=int, help='jittering factor')
    parser.add_argument('--r_feature', type=float, default=0.01, help='coefficient for feature distribution regularization')
    parser.add_argument('--first_bn_mul', type=float, default=10., help='additional multiplier on first bn layer of R_feature')
    parser.add_argument('--tv_l1', type=float, default=0.0, help='coefficient for total variation L1 loss')
    parser.add_argument('--tv_l2', type=float, default=0.004, help='coefficient for total variation L2 loss')
    parser.add_argument('--lr', type=float, default=0.02, help='learning rate for optimization')
    parser.add_argument('--l2', type=float, default=0.0002, help='l2 loss on the image')
    parser.add_argument('--main_mul', type=float, default=1.0, help='coefficient for the main loss in optimization')
    parser.add_argument('--random_label', type=str2bool, default='True', help='generate random label??')
    parser.add_argument('--save_path', type=str, default='./results', help='saved directory path')
    parser.add_argument('--epochs', type=int, default=1, help='epoch')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['TF2_BEHAVIOR'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES']= '0'

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    
    args = parser.parse_args()

    DI_obj=Deep_Inversion(dataset=args.dataset,
                        t_model_path=args.t_model_path,
                        adi_coeff=args.adi_coeff,
                        s_model_path=args.s_model_path,
                        r_feature=args.r_feature,
                        tv_l1=args.tv_l1,
                        tv_l2=args.tv_l2, 
                        l2=args.l2, 
                        lr=args.lr,  
                        n_iters=args.n_iters,
                        first_bn_mul=args.first_bn_mul,
                        main_mul=args.main_mul,
                        bs=args.bs,
                        jitter=args.jitter,
                        save_path=args.save_path,
                        random_label=args.random_label,
                        epochs= args.epochs
                        )
    DI_obj.build()

if __name__ == '__main__':

    main()