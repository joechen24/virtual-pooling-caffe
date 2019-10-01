#!/bin/bash

#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round100_train33test1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round100_interpCUDAv5_train33test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round200_train33test1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round200_interpCUDAv5_train33test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round300_train33test1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round300_interpCUDAv5_train33test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round400_train33test1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round400_interpCUDAv5_train33test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round500_train33test1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round500_interpCUDAv5_train33test1.out

./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round600_train33test1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round600_interpCUDAv5_train33test1.out
#./build/tools/caffe train --solver=models/resnet/tmp_solver/solver_resnet50_round600.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 2>&1 | tee ./resnet_results/resnet50_lininterp_round600.out


#./build/tools/caffe train --solver=models/resnet/tmp_solver/solver_resnet50_round100.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 1 2>&1 | tee ./resnet_results/resnet50_lininterp_round100.out
#./build/tools/caffe train --solver=models/resnet/tmp_solver/solver_resnet50_round200.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 1 2>&1 | tee ./resnet_results/resnet50_lininterp_round200.out
#./build/tools/caffe train --solver=models/resnet/tmp_solver/solver_resnet50_round300.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 1 2>&1 | tee ./resnet_results/resnet50_lininterp_round300.out
#./build/tools/caffe train --solver=models/resnet/tmp_solver/solver_resnet50_round400.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 1 2>&1 | tee ./resnet_results/resnet50_lininterp_round400.out
#./build/tools/caffe train --solver=models/resnet/tmp_solver/solver_resnet50_round500.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 1 2>&1 | tee ./resnet_results/resnet50_lininterp_round500.out

#./build/tools/caffe time --model=models/resnet/train_val_resnet50.prototxt -iterations 5  2>&1 | tee  ./resnet_results/time_resnet50_original_cpu.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round1_train33test1.prototxt -iterations 5  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round1_interpCUDAv5_train33test1_cpu.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round2_train33test1.prototxt -iterations 5  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round2_interpCUDAv5_train33test1_cpu.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round31_train33test1.prototxt -iterations 5  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round31_interpCUDAv5_train33test1_cpu.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round32_train33test1.prototxt -iterations 5  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round32_interpCUDAv5_train33test1_cpu.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round33_train33test1.prototxt -iterations 5  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round33_interpCUDAv5_train33test1_cpu.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round34_train33test1.prototxt -iterations 5  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round34_interpCUDAv5_train33test1_cpu.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round35_train33test1.prototxt -iterations 5  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round35_interpCUDAv5_train33test1_cpu.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round36_train33test1.prototxt -iterations 5  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round36_interpCUDAv5_train33test1_cpu.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round37_train33test1.prototxt -iterations 5  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round37_interpCUDAv5_train33test1_cpu.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round38_train33test1.prototxt -iterations 5  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round38_interpCUDAv5_train33test1_cpu.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round39_train33test1.prototxt -iterations 5  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round39_interpCUDAv5_train33test1_cpu.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round40_train33test1.prototxt -iterations 5  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round40_interpCUDAv5_train33test1_cpu.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round41_train33test1.prototxt -iterations 5  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round41_interpCUDAv5_train33test1_cpu.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round42_train33test1.prototxt -iterations 5  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round42_interpCUDAv5_train33test1_cpu.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round43_train33test1.prototxt -iterations 5  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round43_interpCUDAv5_train33test1_cpu.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round44_train33test1.prototxt -iterations 5  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round44_interpCUDAv5_train33test1_cpu.out

#./build/tools/caffe time --model=models/resnet/resnet-50-cp.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50prune_original.out
#./build/tools/caffe test --model=models/resnet/resnet-50-cp.prototxt --weights=models/resnet/resnet-50-cp.caffemodel -gpu 1 -iterations 10000 2>&1 | tee  ./resnet_results/test_resnet50prune_original.out


#========================================= resnet 50 ========================================
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round2.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round2_interpCUDAv5.out
#./build/tools/caffe train --solver=models/resnet/solver_resnet50_round2.prototxt --weights=models/resnet/resnet50_lininterp_finetune1_CUDAv3_per15k_35batchsize_iter_60000.caffemodel -gpu 1 2>&1 | tee ./resnet_results/resnet50_lininterp_round2_CUDAv5_per30k_35batchsize.out
#./build/tools/caffe train --solver=models/resnet/solver_resnet50_round3.prototxt --weights=models/resnet/resnet50_lininterp_finetune2_CUDAv5_per30k_35batchsize_iter_105000.caffemodel -gpu 1 2>&1 | tee ./resnet_results/resnet50_lininterp_round3_CUDAv5_per30k_35batchsize.out
#./build/tools/caffe train --solver=models/resnet/solver_resnet50_round2.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 1 2>&1 | tee ./resnet_results/resnet50_lininterp_round2_per30k_35batchsize.out
#./build/tools/caffe train --solver=models/resnet/tmp_solver/solver_resnet50_round31.prototxt --weights=models/resnet/snapshots/resnet50_lininterp_finetune2_CUDAv5_per30k_35batchsize_iter_105000.caffemodel -gpu 1 2>&1 | tee ./resnet_results/resnet50_lininterp_round31_CUDAv5_newLRpolicyFrom1e-4_35batchsize.out
#./build/tools/caffe train --solver=models/resnet/tmp_solver/solver_resnet50_round32.prototxt --weights=models/resnet/snapshots/resnet50_lininterp_finetune31_CUDAv5_newLRpolicyFrom1e-4_35batchsize_iter_60000.caffemodel -gpu 1 2>&1 | tee ./resnet_results/resnet50_lininterp_round32_CUDAv5_newLRpolicyFrom1e-4_35batchsize.out
#./build/tools/caffe train --solver=models/resnet/tmp_solver/solver_resnet50_round33.prototxt --weights=models/resnet/snapshots/resnet50_lininterp_finetune32_CUDAv5_newLRpolicyFrom1e-4_35batchsize_iter_60000.caffemodel -gpu 1 2>&1 | tee ./resnet_results/resnet50_lininterp_round33_CUDAv5_newLRpolicyFrom1e-4_35batchsize.out
#./build/tools/caffe train --solver=models/resnet/tmp_solver/solver_resnet50_round34.prototxt --weights=models/resnet/snapshots/resnet50_lininterp_finetune33_CUDAv5_newLRpolicyFrom1e-4_35batchsize_iter_60000.caffemodel -gpu 1 2>&1 | tee ./resnet_results/resnet50_lininterp_round34_CUDAv5_newLRpolicyFrom1e-4_35batchsize.out
#./build/tools/caffe train --solver=models/resnet/tmp_solver/solver_resnet50_round35.prototxt --weights=models/resnet/snapshots/resnet50_lininterp_finetune34_CUDAv5_newLRpolicyFrom1e-4_35batchsize_iter_60000.caffemodel -gpu 1 2>&1 | tee ./resnet_results/resnet50_lininterp_round35_CUDAv5_newLRpolicyFrom1e-4_35batchsize.out
#./build/tools/caffe train --solver=models/resnet/tmp_solver/solver_resnet50_round36.prototxt --weights=models/resnet/snapshots/resnet50_lininterp_finetune35_CUDAv5_newLRpolicyFrom1e-4_35batchsize_iter_60000.caffemodel -gpu 1 2>&1 | tee ./resnet_results/resnet50_lininterp_round36_CUDAv5_newLRpolicyFrom1e-4_35batchsize.out
#./build/tools/caffe train --solver=models/resnet/tmp_solver/solver_resnet50_round37.prototxt --weights=models/resnet/snapshots/resnet50_lininterp_finetune36_CUDAv5_newLRpolicyFrom1e-4_35batchsize_iter_60000.caffemodel -gpu 1 2>&1 | tee ./resnet_results/resnet50_lininterp_round37_CUDAv5_newLRpolicyFrom1e-4_35batchsize.out
#./build/tools/caffe train --solver=models/resnet/tmp_solver/solver_resnet50_round38.prototxt --weights=models/resnet/snapshots/resnet50_lininterp_finetune37_CUDAv5_newLRpolicyFrom1e-4_35batchsize_iter_60000.caffemodel -gpu 1 2>&1 | tee ./resnet_results/resnet50_lininterp_round38_CUDAv5_newLRpolicyFrom1e-4_35batchsize.out
#./build/tools/caffe train --solver=models/resnet/tmp_solver/solver_resnet50_round39.prototxt --weights=models/resnet/snapshots/resnet50_lininterp_finetune38_CUDAv5_newLRpolicyFrom1e-4_35batchsize_iter_60000.caffemodel -gpu 1 2>&1 | tee ./resnet_results/resnet50_lininterp_round39_CUDAv5_newLRpolicyFrom1e-4_35batchsize.out
#./build/tools/caffe train --solver=models/resnet/tmp_solver/solver_resnet50_round40.prototxt --weights=models/resnet/snapshots/resnet50_lininterp_finetune39_CUDAv5_newLRpolicyFrom1e-4_35batchsize_iter_60000.caffemodel -gpu 1 2>&1 | tee ./resnet_results/resnet50_lininterp_round40_CUDAv5_newLRpolicyFrom1e-4_35batchsize.out
#./build/tools/caffe train --solver=models/resnet/tmp_solver/solver_resnet50_round41.prototxt --weights=models/resnet/snapshots/resnet50_lininterp_finetune40_CUDAv5_newLRpolicyFrom1e-4_35batchsize_iter_60000.caffemodel -gpu 1 2>&1 | tee ./resnet_results/resnet50_lininterp_round41_CUDAv5_newLRpolicyFrom1e-4_35batchsize.out
#./build/tools/caffe train --solver=models/resnet/tmp_solver/solver_resnet50_round42.prototxt --weights=models/resnet/snapshots/resnet50_lininterp_finetune41_CUDAv5_newLRpolicyFrom1e-4_35batchsize_iter_60000.caffemodel -gpu 1 2>&1 | tee ./resnet_results/resnet50_lininterp_round42_CUDAv5_newLRpolicyFrom1e-4_35batchsize.out
#./build/tools/caffe train --solver=models/resnet/tmp_solver/solver_resnet50_round43.prototxt --weights=models/resnet/snapshots/resnet50_lininterp_finetune42_CUDAv5_newLRpolicyFrom1e-4_35batchsize_iter_60000.caffemodel -gpu 1 2>&1 | tee ./resnet_results/resnet50_lininterp_round43_CUDAv5_newLRpolicyFrom1e-4_35batchsize.out
#./build/tools/caffe train --solver=models/resnet/tmp_solver/solver_resnet50_round44.prototxt --weights=models/resnet/snapshots/resnet50_lininterp_finetune43_CUDAv5_newLRpolicyFrom1e-4_35batchsize_iter_60000.caffemodel -gpu 1 2>&1 | tee ./resnet_results/resnet50_lininterp_round44_CUDAv5_newLRpolicyFrom1e-4_35batchsize.out

#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round31_train1test1.prototxt  -gpu 0  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round31_interpCUDAv5_train1test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round32_train1test1.prototxt  -gpu 0  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round32_interpCUDAv5_train1test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round33_train1test1.prototxt  -gpu 0  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round33_interpCUDAv5_train1test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round34_train1test1.prototxt  -gpu 0  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round34_interpCUDAv5_train1test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round35_train1test1.prototxt  -gpu 0  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round35_interpCUDAv5_train1test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round36_train1test1.prototxt  -gpu 0  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round36_interpCUDAv5_train1test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round37_train1test1.prototxt  -gpu 0  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round37_interpCUDAv5_train1test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round38_train1test1.prototxt  -gpu 0  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round38_interpCUDAv5_train1test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round39_train1test1.prototxt  -gpu 0  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round39_interpCUDAv5_train1test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round40_train1test1.prototxt  -gpu 0  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round40_interpCUDAv5_train1test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round41_train1test1.prototxt  -gpu 0  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round41_interpCUDAv5_train1test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round42_train1test1.prototxt  -gpu 0  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round42_interpCUDAv5_train1test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round43_train1test1.prototxt  -gpu 0  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round43_interpCUDAv5_train1test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round44_train1test1.prototxt  -gpu 0  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round44_interpCUDAv5_train1test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round1_train1test1.prototxt  -gpu 0  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round1_interpCUDAv5_train1test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round2_train1test1.prototxt  -gpu 0  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round2_interpCUDAv5_train1test1.out

#./build/tools/caffe time --model=models/resnet/train_val_resnet50.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_original.out
#./build/tools/caffe time --model=models/resnet/train_val_resnet50_train33test1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_original_train33test1.out
#./build/tools/caffe time --model=models/resnet/train_val_resnet50_train30test5.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_original_train30test5.out
#./build/tools/caffe time --model=models/resnet/train_val_resnet50_train30test10.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_original_train30test10.out
#./build/tools/caffe time --model=models/resnet/train_val_resnet50_train30test15.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_original_train30test15.out
#./build/tools/caffe time --model=models/resnet/train_val_resnet50_train30test20.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_original_train30test20.out
#./build/tools/caffe time --model=models/resnet/train_val_resnet50_train30test25.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_original_train30test25.out
#./build/tools/caffe time --model=models/resnet/train_val_resnet50_train30test30.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_original_train30test30.out
#./build/tools/caffe time --model=models/resnet/train_val_resnet50_train30test35.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_original_train30test35.out
#./build/tools/caffe time --model=models/resnet/train_val_resnet50_train30test50.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_original_train30test50.out
#./build/tools/caffe time --model=models/resnet/train_val_resnet50_train1test1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_original_train1test1.out
#./build/tools/caffe time --model=models/resnet/train_val_resnet50_train1test30.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_original_train1test30.out
#./build/tools/caffe time --model=models/resnet/train_val_resnet50_train1test35.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_original_train1test35.out
#./build/tools/caffe time --model=models/resnet/train_val_resnet50_train1test40.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_original_train1test40.out

#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round1_train30test50.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round1_interpCUDAv5_train30test50.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round2_train30test50.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round2_interpCUDAv5_train30test50.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round31_train30test50.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round31_interpCUDAv5_train30test50.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round32_train30test50.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round32_interpCUDAv5_train30test50.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round33_train30test50.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round33_interpCUDAv5_train30test50.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round34_train30test50.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round34_interpCUDAv5_train30test50.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round35_train30test50.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round35_interpCUDAv5_train30test50.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round36_train30test50.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round36_interpCUDAv5_train30test50.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round37_train30test50.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round37_interpCUDAv5_train30test50.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round38_train30test50.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round38_interpCUDAv5_train30test50.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round39_train30test50.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round39_interpCUDAv5_train30test50.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round40_train30test50.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round40_interpCUDAv5_train30test50.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round41_train30test50.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round41_interpCUDAv5_train30test50.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round42_train30test50.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round42_interpCUDAv5_train30test50.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round43_train30test50.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round43_interpCUDAv5_train30test50.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round44_train30test50.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round44_interpCUDAv5_train30test50.out

#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round1_train33test1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round1_interpCUDAv5_train33test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round2_train33test1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round2_interpCUDAv5_train33test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round31_train33test1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round31_interpCUDAv5_train33test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round32_train33test1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round32_interpCUDAv5_train33test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round33_train33test1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round33_interpCUDAv5_train33test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round34_train33test1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round34_interpCUDAv5_train33test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round35_train33test1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round35_interpCUDAv5_train33test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round36_train33test1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round36_interpCUDAv5_train33test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round37_train33test1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round37_interpCUDAv5_train33test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round38_train33test1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round38_interpCUDAv5_train33test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round39_train33test1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round39_interpCUDAv5_train33test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round40_train33test1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round40_interpCUDAv5_train33test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round41_train33test1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round41_interpCUDAv5_train33test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round42_train33test1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round42_interpCUDAv5_train33test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round43_train33test1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round43_interpCUDAv5_train33test1.out
#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round44_train33test1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round44_interpCUDAv5_train33test1.out




#./build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round1_interpCUDAv3.out
#./build/tools/caffe train --solver=models/resnet/solver_resnet50_round1.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 2>&1 | tee ./resnet_results/resnet50_lininterp_round1.out
#./build/tools/caffe train --solver=models/resnet/solver_resnet50_round1.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 1 2>&1 | tee ./resnet_results/resnet50_lininterp_round1_per30k.out
#./build/tools/caffe train --solver=models/resnet/solver_resnet50_round1.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 1 2>&1 | tee ./resnet_results/resnet50_lininterp_round1_per15k_35batchsize.out

#./build/tools/caffe test --model=models/resnet/train_val_resnet50.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_origin.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_conv1_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_conv1_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_scale2a_branch1.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_scale2a_branch1.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res2a_branch2a_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res2a_branch2a_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res2a_branch2b_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res2a_branch2b_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_scale2a_branch2c.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_scale2a_branch2c.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res2b_branch2a_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res2b_branch2a_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res2b_branch2b_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res2b_branch2b_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_scale2b_branch2c.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_scale2b_branch2c.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res2c_branch2a_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res2c_branch2a_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res2c_branch2b_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res2c_branch2b_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_scale2c_branch2c.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_scale2c_branch2c.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_scale3a_branch1.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_scale3a_branch1.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res3a_branch2a_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res3a_branch2a_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res3a_branch2b_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res3a_branch2b_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_scale3a_branch2c.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_scale3a_branch2c.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res3b_branch2a_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res3b_branch2a_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res3b_branch2b_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res3b_branch2b_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_scale3b_branch2c.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_scale3b_branch2c.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res3c_branch2a_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res3c_branch2a_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res3c_branch2b_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res3c_branch2b_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_scale3c_branch2c.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_scale3c_branch2c.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res3d_branch2a_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res3d_branch2a_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res3d_branch2b_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res3d_branch2b_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_scale3d_branch2c.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_scale3d_branch2c.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_scale4a_branch1.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_scale4a_branch1.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res4a_branch2a_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res4a_branch2a_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res4a_branch2b_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res4a_branch2b_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_scale4a_branch2c.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_scale4a_branch2c.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res4b_branch2a_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res4b_branch2a_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res4b_branch2b_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res4b_branch2b_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_scale4b_branch2c.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_scale4b_branch2c.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res4c_branch2a_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res4c_branch2a_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res4c_branch2b_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res4c_branch2b_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_scale4c_branch2c.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_scale4c_branch2c.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res4d_branch2a_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res4d_branch2a_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res4d_branch2b_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res4d_branch2b_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_scale4d_branch2c.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_scale4d_branch2c.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res4e_branch2a_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res4e_branch2a_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res4e_branch2b_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res4e_branch2b_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_scale4e_branch2c.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_scale4e_branch2c.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res4f_branch2a_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res4f_branch2a_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res4f_branch2b_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res4f_branch2b_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_scale4f_branch2c.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_scale4f_branch2c.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_scale5a_branch1.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_scale5a_branch1.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res5a_branch2a_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res5a_branch2a_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res5a_branch2b_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res5a_branch2b_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_scale5a_branch2c.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_scale5a_branch2c.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res5b_branch2a_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res5b_branch2a_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res5b_branch2b_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res5b_branch2b_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_scale5b_branch2c.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_scale5b_branch2c.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res5c_branch2a_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res5c_branch2a_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_res5c_branch2b_relu.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_res5c_branch2b_relu.out
#./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_scale5c_branch2c.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_scale5c_branch2c.out







## =================================================== VGG16 ===============================$$
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



