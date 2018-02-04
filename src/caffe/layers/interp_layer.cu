#include <vector>

#include "caffe/layers/interp_layer.hpp"

namespace caffe {


template <typename Dtype>
__global__ void InterpForward(Dtype* const top, const Dtype* const bottom, int height, int width, int num_channel, int batch_size){
    int batch_id = blockIdx.x*blockDim.x+threadIdx.x;
    int channel_id = blockIdx.y*blockDim.y+threadIdx.y;
    if(batch_id < batch_size && channel_id < num_channel) {
	    //printf("%d/%d, %d/%d\n", batch_id, batch_size, channel_id, num_channel);
	    for(int i=0; i<height; i++)
	      for(int j=0; j<width; j++)
	    {
		 //direct copy
		 top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + i*2*width*2+j*2]=
		 bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j];
	 
		 //compute odd rows
		//if(j!=width-1)
		if(i!=height-1)
		 top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + (i*2+1)*width*2 + j*2]=
		 (bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j]+
		 bottom[batch_id*num_channel*height*width + channel_id*height*width + (i+1)*width+j])*0.5;
		 
	    }

	    //compute even columns besides the last one 
	    for(int i=0; i<height*2; i++)
	      for(int j=0; j<width-1; j++){
		 top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + i*width*2 + j*2+1]=
		 (top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + i*width*2+j*2]+
		 top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + i*width*2+j*2+2])*0.5;
	   }
	  
	   //last row
	   for(int i=0; i<width*2; i++) 
		 top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + (height*2-1)*width*2 + i]=
		 top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + (height*2-2)*width*2 + i];

	  // last column
	  for(int i=0; i<height*2; i++)  
		 top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + i*width*2 + width*2-1]=
		 top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + i*width*2 + width*2-2];
    }
}

template <typename Dtype>
void InterpLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  //SinForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
  //    count, bottom_data, top_data);
  //InterpForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(top_data, bottom_data, height_, width_, channels_, batch_);
  //printf("CAFFE BLOCKS: %d CAFFE NUM THREADS: %d \n", CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS);
    dim3 numb(batch_/32+1, channels_/32);
    dim3 block(32, 32);
  InterpForward<Dtype><<<numb, block>>>(top_data, bottom_data, height_, width_, channels_, batch_);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void InterpBackward(const Dtype* const top, Dtype* const bottom, int height, int width, int num_channel, int batch_size){
    int batch_id = blockIdx.x*blockDim.x + threadIdx.x;
    int channel_id = blockIdx.y*blockDim.y + threadIdx.y;
    if(batch_id < batch_size && channel_id < num_channel) {
	    //printf("%d %d\n", batch_id, channel_id);
	    for(int i=0; i<height; i++)
	      for(int j=0; j<width; j++)
	    {
		 //direct copy
		 bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j]=
		 top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + i*2*width*2+j*2];
	 
		 //compute odd rows
		//if(j!=width-1) {
		if(i!=height-1) {

		 bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j] += 0.5*
		 top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + (i*2+1)*width*2 + j*2];

		 
		 bottom[batch_id*num_channel*height*width + channel_id*height*width + (i+1)*width+j]+=0.5*
		 top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + (i*2+1)*width*2 + j*2];
		}
		 
	    }

	    //compute even columns odd rows 
	    for(int i=0; i<height*2; i+=2)
	      for(int j=1; j<width*2-1; j+=2){
		 bottom[batch_id*num_channel*height*width + channel_id*height*width + (i/2)*width+j/2]+=0.5*
		 top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + i*width*2 + j];

		 bottom[batch_id*num_channel*height*width + channel_id*height*width + (i/2)*width+j/2+1]+=0.5*
		 top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + i*width*2 + j];
	   }

	   //even columns even rows: aka corners
	    for(int i=1; i<height*2-1; i+=2)
	      for(int j=1; j<width*2-1; j+=2){
		 bottom[batch_id*num_channel*height*width + channel_id*height*width + (i/2)*width+j/2]+=0.25*
		 top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + i*width*2 + j];

		 bottom[batch_id*num_channel*height*width + channel_id*height*width + (i/2+1)*width+j/2]+=0.25*
		 top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + i*width*2 + j];

		 bottom[batch_id*num_channel*height*width + channel_id*height*width + (i/2)*width+j/2+1]+=0.25*
		 top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + i*width*2 + j];

		 bottom[batch_id*num_channel*height*width + channel_id*height*width + (i/2+1)*width+j/2+1]+=0.25*
		 top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + i*width*2 + j];

	   }
	}
}

template <typename Dtype>
void InterpLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = top[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    //SinBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    //    count, top_diff, bottom_data, bottom_diff);
    //CUDA_POST_KERNEL_CHECK;
    //InterpBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(top_diff, bottom_diff, height_, width_, channels_, batch_);
    dim3 numb(batch_/32+1, channels_/32);
    dim3 block(32, 32);
    InterpBackward<Dtype><<<numb,block>>>(top_diff, bottom_diff, height_, width_, channels_, batch_);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InterpLayer);


}  // namespace caffe
