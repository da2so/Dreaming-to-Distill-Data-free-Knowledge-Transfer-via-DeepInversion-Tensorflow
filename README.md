# Dreaming to Distill: Data free Knowledge Transfer via DeepInversion Tensorflow Keras

<p align="center">
	<img src="./assets/fig1.gif" alt="drawing2" width="500"/>
</p>

## Requirements

- Tensorflow 2.3.0 
- Python 3.6


## Running the code

### ImageNet

The following code generates DeepInversion images by using pretrained resnet50v2 model. 

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --dataset=imagenet --t_model_path=resnet50v2 --adi_coeff=0.0 --bs=32 --n_iters=3000 --lr=0.02 --jitter=30
```

Arguments:

- `dataset` - Select a dataset ['imagenet']
	- I will update some codes for cifar10 and cifar100 datasets.
- `t_model_path` - teacher model path
	- Available list of teacher models: ['resnet50v2', 'mobilenet', 'mobilenetv2', 'vgg19']
- `adi_coeff` - Coefficient for Adaptive Deep Inversion
- `s_model_path` - Student model path
- `n_iters` - iterations
- `bs` - Batch size
- `jitter` - jittering factor
- `r_feature` - Coefficient for feature distribution regularization
- `first_bn_mul` - Additional multiplier on first bn layer of R_feature
- `tv_l1` - Coefficient for total variation L1 loss
- `tv_l2` - Coefficient for total variation L2 loss
- `lr` - Learning rate
- `l2` - L2 loss on the image
- `main_mul` - Coefficient for the main loss in optimization
- `random_label` - Generate random label??
- `save_path` - Saved directory path



![2](./assets/fig2.png)


## Understanding this paper

Check my blog!!
[Here](https://da2so.github.io/2020-08-18-Dreaming_to_Distill_Data-free_Knowledge_Transfer_via_DeepInversion/)