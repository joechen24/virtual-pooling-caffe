#!/bin/bash

##### Thinet Sensitivity analysis
./build/tools/caffe test --model=models/vgg16/train_val_thinet.prototxt --weights=models/vgg16/ThiNet-GAP.caffemodel -gpu 2 -iterations 1000 2>&1 | tee  thinet_results/original_thinet.out

./build/tools/caffe test --model=models/vgg16/train_val_thinet_lininterp_conv11_afterAct.prototxt --weights=models/vgg16/ThiNet-GAP.caffemodel -gpu 2 -iterations 1000 2>&1 | tee  thinet_results/vgg16_lininterp_conv11_afterAct.out
./build/tools/caffe test --model=models/vgg16/train_val_thinet_lininterp_conv12_afterAct.prototxt --weights=models/vgg16/ThiNet-GAP.caffemodel -gpu 2 -iterations 1000 2>&1 | tee  thinet_results/vgg16_lininterp_conv12_afterAct.out
./build/tools/caffe test --model=models/vgg16/train_val_thinet_lininterp_conv21_afterAct.prototxt --weights=models/vgg16/ThiNet-GAP.caffemodel -gpu 2 -iterations 1000 2>&1 | tee  thinet_results/vgg16_lininterp_conv21_afterAct.out
./build/tools/caffe test --model=models/vgg16/train_val_thinet_lininterp_conv22_afterAct.prototxt --weights=models/vgg16/ThiNet-GAP.caffemodel -gpu 2 -iterations 1000 2>&1 | tee  thinet_results/vgg16_lininterp_conv22_afterAct.out
./build/tools/caffe test --model=models/vgg16/train_val_thinet_lininterp_conv31_afterAct.prototxt --weights=models/vgg16/ThiNet-GAP.caffemodel -gpu 2 -iterations 1000 2>&1 | tee  thinet_results/vgg16_lininterp_conv31_afterAct.out

./build/tools/caffe test --model=models/vgg16/train_val_thinet_lininterp_conv32_afterAct.prototxt --weights=models/vgg16/ThiNet-GAP.caffemodel -gpu 2 -iterations 1000 2>&1 | tee  thinet_results/vgg16_lininterp_conv32_afterAct.out
./build/tools/caffe test --model=models/vgg16/train_val_thinet_lininterp_conv33_afterAct.prototxt --weights=models/vgg16/ThiNet-GAP.caffemodel -gpu 2 -iterations 1000 2>&1 | tee  thinet_results/vgg16_lininterp_conv33_afterAct.out
./build/tools/caffe test --model=models/vgg16/train_val_thinet_lininterp_conv41_afterAct.prototxt --weights=models/vgg16/ThiNet-GAP.caffemodel -gpu 2 -iterations 1000 2>&1 | tee  thinet_results/vgg16_lininterp_conv41_afterAct.out
./build/tools/caffe test --model=models/vgg16/train_val_thinet_lininterp_conv42_afterAct.prototxt --weights=models/vgg16/ThiNet-GAP.caffemodel -gpu 2 -iterations 1000 2>&1 | tee  thinet_results/vgg16_lininterp_conv42_afterAct.out
./build/tools/caffe test --model=models/vgg16/train_val_thinet_lininterp_conv43_afterAct.prototxt --weights=models/vgg16/ThiNet-GAP.caffemodel -gpu 2 -iterations 1000 2>&1 | tee  thinet_results/vgg16_lininterp_conv43_afterAct.out
./build/tools/caffe test --model=models/vgg16/train_val_thinet_lininterp_conv51_afterAct.prototxt --weights=models/vgg16/ThiNet-GAP.caffemodel -gpu 2 -iterations 1000 2>&1 | tee  thinet_results/vgg16_lininterp_conv51_afterAct.out
./build/tools/caffe test --model=models/vgg16/train_val_thinet_lininterp_conv52_afterAct.prototxt --weights=models/vgg16/ThiNet-GAP.caffemodel -gpu 2 -iterations 1000 2>&1 | tee  thinet_results/vgg16_lininterp_conv52_afterAct.out
./build/tools/caffe test --model=models/vgg16/train_val_thinet_lininterp_conv53_afterAct.prototxt --weights=models/vgg16/ThiNet-GAP.caffemodel -gpu 2 -iterations 1000 2>&1 | tee  thinet_results/vgg16_lininterp_conv53_afterAct.out

################ ViP vs. double-stride+pooling removal
#./build/tools/caffe test --model=models/vgg16/train_val_removePool1.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 1 -iterations 1000 2>&1 | tee ./vgg_results/train_val_removePool1.out
#./build/tools/caffe test --model=models/vgg16/train_val_removePool2.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 1 -iterations 1000 2>&1 | tee ./vgg_results/train_val_removePool2.out
#./build/tools/caffe test --model=models/vgg16/train_val_removePool3.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 1 -iterations 1000 2>&1 | tee ./vgg_results/train_val_removePool3.out
#./build/tools/caffe test --model=models/vgg16/train_val_removePool4.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 1 -iterations 1000 2>&1 | tee ./vgg_results/train_val_removePool4.out
#./build/tools/caffe test --model=models/vgg16/train_val_removePool5.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 1 -iterations 1000 2>&1 | tee ./vgg_results/train_val_removePool5.out
#./build/tools/caffe test --model=models/vgg16/train_val_lininterp_conv12_afterAct.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 1 -iterations 1000 2>&1 | tee ./vgg_results/train_val_conv12.out
#./build/tools/caffe test --model=models/vgg16/train_val_lininterp_conv22_afterAct.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 1 -iterations 1000 2>&1 | tee ./vgg_results/train_val_conv22.out
#./build/tools/caffe test --model=models/vgg16/train_val_lininterp_conv33_afterAct.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 1 -iterations 1000 2>&1 | tee ./vgg_results/train_val_conv33.out
#./build/tools/caffe test --model=models/vgg16/train_val_lininterp_conv43_afterAct.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 1 -iterations 1000 2>&1 | tee ./vgg_results/train_val_conv43.out
#./build/tools/caffe test --model=models/vgg16/train_val_lininterp_conv53_afterAct.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 1 -iterations 1000 2>&1 | tee ./vgg_results/train_val_conv53.out




#./build/tools/caffe test --model=models/vgg16/train_val_lininterp_conv11_afterAct.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 0 -iterations 1000
#./build/tools/caffe time --model=models/vgg16/train_val.prototxt -iterations 5   2>&1 | tee  ./vgg16_results/time_vgg16_original_cpu.out
#./build/tools/caffe time --model=models/vgg16/train_val_lininterp_round1_afterAct.prototxt -iterations 5   2>&1 | tee  ./vgg16_results/time_vgg16_lininterp_finetune1_afterAct_cpu.out
#./build/tools/caffe time --model=models/vgg16/train_val_lininterp_round2_afterAct.prototxt -iterations 5  2>&1 | tee   ./vgg16_results/time_vgg16_lininterp_finetune2_afterAct_cpu.out
#./build/tools/caffe time --model=models/vgg16/train_val_lininterp_round3_afterAct.prototxt -iterations 5  2>&1 | tee   ./vgg16_results/time_vgg16_lininterp_finetune3_afterAct_cpu.out
#./build/tools/caffe time --model=models/vgg16/train_val_lininterp_round4_afterAct.prototxt -iterations 5 2>&1 | tee    ./vgg16_results/time_vgg16_lininterp_finetune4_afterAct_cpu.out

#./build/tools/caffe test --model=models/vgg16/train_val.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 0 -iterations 1000 2>&1 | tee  original_vgg16.out
#
#./build/tools/caffe test --model=models/vgg16/train_val_lininterp_conv11_afterAct.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 0 -iterations 1000 2>&1 | tee  vgg16_lininterp_conv11_afterAct.out
#./build/tools/caffe test --model=models/vgg16/train_val_lininterp_conv12_afterAct.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 0 -iterations 1000 2>&1 | tee  vgg16_lininterp_conv12_afterAct.out
#./build/tools/caffe test --model=models/vgg16/train_val_lininterp_conv21_afterAct.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 0 -iterations 1000 2>&1 | tee  vgg16_lininterp_conv21_afterAct.out
#./build/tools/caffe test --model=models/vgg16/train_val_lininterp_conv22_afterAct.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 0 -iterations 1000 2>&1 | tee  vgg16_lininterp_conv22_afterAct.out
#./build/tools/caffe test --model=models/vgg16/train_val_lininterp_conv31_afterAct.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 0 -iterations 1000 2>&1 | tee  vgg16_lininterp_conv31_afterAct.out

#./build/tools/caffe test --model=models/vgg16/train_val_lininterp_conv32_afterAct.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 1 -iterations 1000 2>&1 | tee  vgg16_lininterp_conv32_afterAct.out
#./build/tools/caffe test --model=models/vgg16/train_val_lininterp_conv33_afterAct.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 1 -iterations 1000 2>&1 | tee  vgg16_lininterp_conv33_afterAct.out
#./build/tools/caffe test --model=models/vgg16/train_val_lininterp_conv41_afterAct.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 1 -iterations 400 2>&1 | tee  vgg16_lininterp_conv41_afterAct.out
#./build/tools/caffe test --model=models/vgg16/train_val_lininterp_conv42_afterAct.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 1 -iterations 400 2>&1 | tee  vgg16_lininterp_conv42_afterAct.out
#./build/tools/caffe test --model=models/vgg16/train_val_lininterp_conv43_afterAct.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 1 -iterations 400 2>&1 | tee  vgg16_lininterp_conv43_afterAct.out
#./build/tools/caffe test --model=models/vgg16/train_val_lininterp_conv51_afterAct.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 1 -iterations 400 2>&1 | tee  vgg16_lininterp_conv51_afterAct.out
#./build/tools/caffe test --model=models/vgg16/train_val_lininterp_conv52_afterAct.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 1 -iterations 400 2>&1 | tee  vgg16_lininterp_conv52_afterAct.out
#./build/tools/caffe test --model=models/vgg16/train_val_lininterp_conv53_afterAct.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 1 -iterations 400 2>&1 | tee  vgg16_lininterp_conv53_afterAct.out

#./build/tools/caffe train --solver=models/vgg16/solver_lininterp_finetune.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 0  2>&1 | tee  vgg16_lininterp_finetune1_afterAct.out
#./build/tools/caffe test --model=models/vgg16/train_val_lininterp_round1_afterAct.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 0 -iterations 500 2>&1 | tee  vgg16_lininterp_round1_afterAct.out

#./build/tools/caffe train --solver=models/vgg16/solver_lininterp_finetune.prototxt --weights=models/vgg16/vgg16_lininterp_finetune1_afterAct_iter_10000withLR0.001.caffemodel -gpu 0  2>&1 | tee  vgg16_lininterp_finetune1_afterAct_withLR0.0001.out
#./build/tools/caffe test --model=models/vgg16/train_val_lininterp_round1_afterAct.prototxt --weights=models/vgg16/vgg16_lininterp_finetune1_afterAct_withLR0.0001_iter_9098.caffemodel -gpu 1 -iterations 1000 2>&1 | tee  vgg16_lininterp_finetune1_afterAct_withLR0.0001_iter_9098_CUDA.out

#./build/tools/caffe train --solver=models/vgg16/solver_lininterp_finetune_round2.prototxt --weights=models/vgg16/vgg16_lininterp_finetune1_afterAct_withLR0.0001_iter_9098.caffemodel -gpu 0  2>&1 | tee  vgg16_lininterp_finetune2_afterAct_CUDA.out
#./build/tools/caffe train --solver=models/vgg16/solver_lininterp_finetune_round2.prototxt --weights=models/vgg16/vgg16_lininterp_finetune2_afterAct_withLR0.001_iter_8000.caffemodel -gpu 0  2>&1 | tee  vgg16_lininterp_finetune2_afterAct_withLR0.0001.out
#./build/tools/caffe time --model=models/vgg16/train_val_lininterp_round1_afterAct.prototxt  -gpu 1  2>&1 | tee  time_vgg16_lininterp_finetune1_afterAct_interpCUDA.out
#./build/tools/caffe time --model=models/vgg16/train_val.prototxt  -gpu 1  2>&1 | tee  time_vgg16_original.out
#
#./build/tools/caffe test --model=models/vgg16/train_val_lininterp_round2_afterAct.prototxt --weights=models/vgg16/vgg16_lininterp_finetune2_afterAct_withLR0.0001_iter_8000.caffemodel -gpu 0 -iterations 400 2>&1 | tee  vgg16_lininterp_finetune2_afterAct_withLR0.0001_iter_8000.out

#./build/tools/caffe train --solver=models/vgg16/solver_lininterp_finetune.prototxt --weights=models/vgg16/VGG_ILSVRC_16_layers.caffemodel -gpu 1  2>&1 | tee  vgg16_lininterp_finetune1_afterAct_CUDA_per15k.out
#./build/tools/caffe train --solver=models/vgg16/solver_lininterp_finetune_round2.prototxt --weights=models/vgg16/vgg16_lininterp_finetune2_afterAct_CUDA_iter_30000.caffemodel -gpu 0  2>&1 | tee  vgg16_lininterp_finetune2_afterAct_withLR000001_CUDA.out
#./build/tools/caffe test --model=models/vgg16/train_val_lininterp_round1_afterAct.prototxt --weights=models/vgg16/vgg16_lininterp_finetune1_afterAct_CUDA_iter_45000.caffemodel -gpu 1 -iterations 2000 2>&1 | tee  vgg16_lininterp_finetune1_afterAct_CUDA_iter_45000.out
#./build/tools/caffe train --solver=models/vgg16/solver_lininterp_finetune_round2.prototxt --weights=models/vgg16/vgg16_lininterp_finetune1_afterAct_CUDA_iter_45000.caffemodel -gpu 1  2>&1 | tee  vgg16_lininterp_finetune2_afterAct_withCUDA45k.out
#./build/tools/caffe train --solver=models/vgg16/solver_lininterp_finetune_round3_2.prototxt --weights=models/vgg16/vgg16_lininterp_finetune2_afterAct_withLR000001_CUDA_iter_30000.caffemodel -gpu 0  2>&1 | tee  vgg16_lininterp_finetune3_2_afterAct_fromPythonResult.out
#./build/tools/caffe train --solver=models/vgg16/solver_lininterp_finetune_round3.prototxt --weights=models/vgg16/vgg16_lininterp_finetune2_afterAct_withCUDA45kRound1_iter_50000.caffemodel -gpu 0  2>&1 | tee  vgg16_lininterp_finetune3_afterAct_LR0.0001_fromCUDAResult.out
#./build/tools/caffe train --solver=models/vgg16/solver_lininterp_finetune_round3.prototxt --weights=models/vgg16/vgg16_lininterp_finetune2_afterAct_withCUDA45kRound1_iter_50000.caffemodel -gpu 0  2>&1 | tee  vgg16_lininterp_finetune3_afterAct_LR0.0001_gamma05_fromCUDAResult.out
#./build/tools/caffe time --model=models/vgg16/train_val_lininterp_round2_afterAct.prototxt  -gpu 1  2>&1 | tee  time_vgg16_lininterp_finetune2_afterAct_interpCUDA.out
#./build/tools/caffe time --model=models/vgg16/train_val_lininterp_round3_afterAct.prototxt  -gpu 1  2>&1 | tee  time_vgg16_lininterp_finetune3_afterAct_interpCUDA.out
#./build/tools/caffe time --model=models/vgg16/train_val_lininterp_round3_2_afterAct.prototxt  -gpu 1  2>&1 | tee  time_vgg16_lininterp_finetune3_2_afterAct_interpCUDA.out

#./build/tools/caffe test --model=models/vgg16/train_val_lininterp_round3_afterAct.prototxt --weights=models/vgg16/vgg16_lininterp_finetune3_afterAct_LR0.0001_gamma05_fromCUDAResult_iter_50000.caffemodel -gpu 0 -iterations 2000
#./build/tools/caffe train --solver=models/vgg16/solver_lininterp_finetune_round3.prototxt --snapshot=models/vgg16/vgg16_lininterp_finetune3_afterAct_LR0.0001_gamma05_fromCUDAResult_iter_50000.solverstate -gpu 0  2>&1 | tee  vgg16_lininterp_finetune3_afterAct_LR0.0001_gamma05_fromCUDAResult_resumeAfter50k.out
#./build/tools/caffe train --solver=models/vgg16/solver_lininterp_finetune_round4.prototxt --weights=models/vgg16/vgg16_lininterp_finetune3_afterAct_LR0.0001_gamma05_fromCUDAResult_iter_70000.caffemodel -gpu 0  2>&1 | tee  vgg16_lininterp_finetune4_afterAct_fromCUDAResult.out

#./build/tools/caffe time --model=models/vgg16/train_val_lininterp_round1_afterAct.prototxt  -gpu 1  2>&1 | tee  time_vgg16_lininterp_finetune1_afterAct_interpCUDAv2.out
#./build/tools/caffe time --model=models/vgg16/train_val_lininterp_round2_afterAct.prototxt  -gpu 1  2>&1 | tee  time_vgg16_lininterp_finetune2_afterAct_interpCUDAv2.out
#./build/tools/caffe time --model=models/vgg16/train_val_lininterp_round3_afterAct.prototxt  -gpu 1  2>&1 | tee  time_vgg16_lininterp_finetune3_afterAct_interpCUDAv2.out
#./build/tools/caffe time --model=models/vgg16/train_val_lininterp_round3_2_afterAct.prototxt  -gpu 1  2>&1 | tee  time_vgg16_lininterp_finetune3_2_afterAct_interpCUDAv2.out

#/usr/local/cuda/bin/nvprof --metrics dram_read_transactions,dram_write_transactions ./build/tools/caffe time --model=models/vgg16/train_val_lininterp_round4_afterAct.prototxt -iterations 1 -gpu 1  2>&1 | tee  time_vgg16_lininterp_finetune4_afterAct_interpCUDAv3.out
#./build/tools/caffe train --solver=models/vgg16/solver_lininterp_finetune_round3.prototxt --snapshot=models/vgg16/vgg16_lininterp_finetune3_afterAct_LR0.0001_gamma05_fromCUDAResult_iter_50000.solverstate -gpu 0  2>&1 | tee  vgg16_lininterp_finetune3_afterAct_LR0.0001_gamma05_fromCUDAResult_resumeAfter50k_v3.out

#./build/tools/caffe train --solver=models/vgg16/solver_lininterp_finetune_round4.prototxt --weights=models/vgg16/vgg16_lininterp_finetune3_afterAct_LR0.0001_gamma05_fromCUDAResult_v3_iter_100000.caffemodel -gpu 0  2>&1 | tee  vgg16_lininterp_finetune4_afterAct_fromCUDAResult_v3.out

#for exp_id in {1..1}
#do
#	./${folder}/train_script.sh refnet_hier_nofc 0 refnet_template_solver.prototxt ${exp_id} gamma1
#done
#for exp_id in {1..1}
#do
#	./${folder}/train_script.sh refnet_hier_smaller 0 refnet_template_solver.prototxt ${exp_id} gamma1
#done
#for exp_id in {1..1}
#do
#	./${folder}/train_script.sh refnet_hier_smallBranch 0 refnet_template_solver.prototxt ${exp_id} gamma1
#done
#for exp_id in {1..1}
#do
#	./${folder}/train_script.sh refnet_hier 0 refnet_template_solver.prototxt ${exp_id} gamma1
#done


#########################################################
###### old examples
#########################################################
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_hier 0 lenet_template_solver.prototxt ${exp_id} gamma001
#done
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_hier_small 0 lenet_template_solver.prototxt ${exp_id} gamma001
#done

#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_hier 0 lenet_template_solver.prototxt ${exp_id} gamma01
#done

#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_hier 0 lenet_template_solver.prototxt ${exp_id} gamma1
#done

#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_smallBranch 0 lenet_template_solver.prototxt ${exp_id} origin
#done

#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_small 0 lenet_template_solver.prototxt ${exp_id} origin
#done
####################################################
##### CIFAR 10 experiments
####################################################

#for exp_id in {1..20}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_small 0 lenet_cifar10_template_solver.prototxt ${exp_id}
#done
#
#for exp_id in {1..20}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_smallBclass 0 lenet_cifar10_template_solver.prototxt ${exp_id}
#done
#
#for exp_id in {1..20}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_smallBranch 0 lenet_cifar10_template_solver.prototxt ${exp_id}
#done


## experiment on longer epoch
#for exp_id in {1..20}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_small 0 lenet_cifar10_template_solver_450k.prototxt ${exp_id} 450k
#done

## experiment with gamma=0.001

#for exp_id in {1..12}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_small 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma0001
#done
#
#for exp_id in {1..12}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_smallBclass 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma0001
#done
#
#for exp_id in {1..12}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_smallBranch 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma0001
#done

#for exp_id in {1..12}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier3_smallBranch 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma0001
#done

#for exp_id in {1..12}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier3_smallBranch 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma001
#done

#for exp_id in {1..12}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier3_smallBranch 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma01
#done

#####################################################
### see how base lr affects accuracy
#####################################################

#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_small 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma001_lr01
#done
#
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma001_lr01
#done
#
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_smallBranch 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma001_lr01
#done
#
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_cifar10_hierMoE_small 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma001_lr01
#done
#
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_small 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma001_lr0001
#done
#
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma001_lr0001
#done
#
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_smallBranch 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma001_lr0001
#done
#
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_cifar10_hierMoE_small 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma001_lr0001
#done



