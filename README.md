# Virtual Pooling Caffe

Implementation of virtual pooling technique to speedup convolutional neural networks in Caffe.

We provide scripts to automatically generate train_val and solver prototxt for sensitivity analysis and grouped fine-tuning 
(need to create tmp_solver and tmp_train_val folders under caffe/models/{net} first)
For example, run python ./models/resnet/modify_net.py

Script for parsing accuracy and timing output results is provided as well (need to create {net}_results folder under caffe/ first)
For example, run python ./models/resnet/parse_results.py

To do sensitivity analysis, run
./build/tools/caffe test --model=models/vgg16/train_val_lininterp_conv11_afterAct.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 0 -iterations 1000 
./build/tools/caffe test --model=models/vgg16/train_val_lininterp_conv12_afterAct.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 0 -iterations 1000 
./build/tools/caffe test --model=models/vgg16/train_val_lininterp_conv21_afterAct.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 0 -iterations 1000 
.
.
.

To do grouped fine-tuning, run
./build/tools/caffe train --solver=models/vgg16/solver_lininterp_finetune_round1.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 0 

To get timing info, run
./build/tools/caffe time --model=models/vgg16/train_val_lininterp_round1_afterAct.prototxt  -gpu 0

