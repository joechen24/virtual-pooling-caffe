#include <vector>

#include "caffe/layers/interp_layer.hpp"

namespace caffe {


template <typename Dtype>
void InterpLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	channels_=bottom[0]->channels();
	height_=bottom[0]->height();
	width_=bottom[0]->width();
	batch_=bottom[0]->num();

}


template <typename Dtype>
void InterpLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  //accommondate odd number of previous conv's input feature map
  if (height_ != 4)
	  top[0]->Reshape(batch_, channels_, height_*2,
	      width_*2);
  else
	  top[0]->Reshape(batch_, channels_, height_*2-1,
	      width_*2-1);

}

template <typename Dtype>
void InterpForward(const int nthreads, Dtype* const top, const Dtype* const bottom, int height, int width, int num_channel, int batch_size){
   //CUDA_KERNEL_LOOP(index, nthreads){
   for(int index=0; index<nthreads; index++){
      int batch_id=(index / width / height /num_channel);
      int channel_id=(index / width / height)%num_channel;
      int i=(index/width)%height;
      int j=index % width;
    

      if(height != 4){
	     //direct copy
	      top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + i*2*width*2+j*2]=
	      bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j];
	     //compute odd rows
	      if(i!=height-1)
		top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + (i*2+1)*width*2 + j*2]=
			 (bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j]+
			 bottom[batch_id*num_channel*height*width + channel_id*height*width + (i+1)*width+j])*0.5;
	     //compute even columns besides the last one
	      if(j!=width-1){
		top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + i*2*width*2 + j*2+1]=
			 (bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j]+
			 bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j+1])*0.5;
	      }
	   if( (i!=height-1) && (j!=width-1)) 
		top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + (i*2+1)*width*2 + j*2+1]=
			 (bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j]+
			 bottom[batch_id*num_channel*height*width + channel_id*height*width + (i+1)*width+j]+
			 bottom[batch_id*num_channel*height*width + channel_id*height*width + (i+1)*width+(j+1)]+
			 bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j+1])*0.25;
	 
	    //last row
	    if(i==height-1){
			 top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + (height*2-1)*width*2 + j*2]=
			 bottom[batch_id*num_channel*height*width + channel_id*height*width + (height-1)*width+j];

			if(j!=width-1)
			 top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + (height*2-1)*width*2 + j*2+1]=
			 (bottom[batch_id*num_channel*height*width + channel_id*height*width + (height-1)*width+j]+
			 bottom[batch_id*num_channel*height*width + channel_id*height*width + (height-1)*width+j+1])*0.5;

	    }

	   //last column
	   if(j==width-1){
			 top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + 2*i*width*2 + width*2-1]=
			 bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+width-1];

			if(i!=height-1){
			 top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + (2*i+1)*width*2 + width*2-1]=
			 (bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+width-1]+
			 bottom[batch_id*num_channel*height*width + channel_id*height*width + (i+1)*width+width-1])*0.5;
			}else{
			 top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + (2*i+1)*width*2 + width*2-1]=
			 bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+width-1];
			}
	   }
	}else{
	     //direct copy
	      int new_height =  2* height - 1;
	      int new_width =  2* width - 1;
	      int top_batch_channel_loc = batch_id*num_channel*new_height*new_width + channel_id*new_height*new_width;
	
	      top[top_batch_channel_loc + i*2*new_width+j*2]=
	      bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j];
	    
	      //compute odd rows
	      if(i!=height-1)
		top[top_batch_channel_loc + (i*2+1)*new_width + j*2]=
			 (bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j]+
			 bottom[batch_id*num_channel*height*width + channel_id*height*width + (i+1)*width+j])*0.5;

	      //compute even columns besides the last one
	      if(j!=width-1){
		top[top_batch_channel_loc + i*2*new_width + j*2+1]=
			 (bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j]+
			 bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j+1])*0.5;
	      }
	      if( (i!=height-1) && (j!=width-1)) 
		top[top_batch_channel_loc + (i*2+1)*new_width + j*2+1]=
			 (bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j]+
			 bottom[batch_id*num_channel*height*width + channel_id*height*width + (i+1)*width+j]+
			 bottom[batch_id*num_channel*height*width + channel_id*height*width + (i+1)*width+(j+1)]+
			 bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j+1])*0.25;
	}
	 
  }  

}

template <typename Dtype>
void InterpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  //SinForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
  //    count, bottom_data, top_data);
  InterpForward<Dtype>(count, top_data, bottom_data, height_, width_, channels_, batch_);
  //printf("CAFFE BLOCKS: %d CAFFE NUM THREADS: %d \n", CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS);

    //dim3 numb(batch_/32+1, channels_/32);
    //dim3 block(32, 32);
  //InterpForward<Dtype><<<numb, block>>>(top_data, bottom_data, height_, width_, channels_, batch_);
}


template <typename Dtype>
void InterpBackward(const int nthreads, const Dtype* const top, Dtype* const bottom, int height, int width, int num_channel, int batch_size){

//CUDA_KERNEL_LOOP(index, nthreads){
for(int index=0; index<nthreads; index++){
      int batch_id=(index / width / height /num_channel);
      int channel_id=(index / width / height)%num_channel;
      int i=(index/width)%height;
      int j=index % width;

      if (height != 4){
		 //direct copy
		 bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j]=
		 top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + i*2*width*2+j*2];

		 //compute odd rows
		 if( (i!=0) && (i!=height-1)) {
		 
		  bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j] += 0.5*
		  (top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + (i*2+1)*width*2 + j*2]+
		  top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + (i*2-1)*width*2 + j*2]);
		 }
		 if(i==0){
		  bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j] += 0.5*
		  top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + (i*2+1)*width*2 + j*2];
		 }
		 if(i==height-1){
		  bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j] += 0.5*
		  top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + (i*2-1)*width*2 + j*2];
		 }

		  //compute even columns odd rows 
		 if( (j!=0) && (j!=width-1))
		 {
		       bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j]+=0.5*
		       (top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + i*2*width*2 + j*2+1]+
		       top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + i*2*width*2 + j*2-1]);
		 }
		 if(j==0){
		       bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j]+=0.5*
		       top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + i*2*width*2 + j*2+1];
		 }
		 if(j==width-1){
		       bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j]+=0.5*
		       top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + i*2*width*2 + j*2-1];
		 }
	 
		if( (i*2+1<height*2-1) && (i*2+1 >=0) && (j*2+1 <width*2-1) && (j*2+1>=0))
			bottom[batch_id*num_channel*height*width + channel_id*height*width + (i)*width+j]+=0.25*top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + (i*2+1)*width*2 + j*2+1];
		if( (i*2+1<height*2-1) && (i*2+1 >=0) && (j*2-1 <width*2-1) && (j*2-1>=0))
			bottom[batch_id*num_channel*height*width + channel_id*height*width + (i)*width+j]+=0.25*top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + (i*2+1)*width*2 + j*2-1];
		if( (i*2-1<height*2-1) && (i*2-1 >=0) && (j*2+1 <width*2-1) && (j*2+1>=0))
			bottom[batch_id*num_channel*height*width + channel_id*height*width + (i)*width+j]+=0.25*top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + (i*2-1)*width*2 + j*2+1];
		if( (i*2-1<height*2-1) && (i*2-1 >=0) && (j*2-1 <width*2-1) && (j*2-1>=0))
			bottom[batch_id*num_channel*height*width + channel_id*height*width + (i)*width+j]+=0.25*top[batch_id*num_channel*height*width*4 + channel_id*height*width*4 + (i*2-1)*width*2 + j*2-1];
	}else{
	      int new_height =  2* height - 1;
	      int new_width =  2* width - 1;
	      int top_batch_channel_loc = batch_id*num_channel*new_height*new_width + channel_id*new_height*new_width;
		 //direct copy
		 bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j]=
		 top[top_batch_channel_loc + i*2*new_width+j*2];

		 //compute odd rows
		 if( (i!=0) && (i!=height-1)) {
		 
		  bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j] += 0.5*
		  (top[top_batch_channel_loc+ (i*2+1)*new_width + j*2]+
		  top[top_batch_channel_loc + (i*2-1)*new_width + j*2]);
		 }
		 if(i==0){
		  bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j] += 0.5*
		  top[top_batch_channel_loc + (i*2+1)*new_width + j*2];
		 }
		 if(i==height-1){
		  bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j] += 0.5*
		  top[top_batch_channel_loc + (i*2-1)*new_width + j*2];
		 }

		  //compute even columns odd rows 
		 if( (j!=0) && (j!=width-1))
		 {
		       bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j]+=0.5*
		       (top[top_batch_channel_loc + i*2*new_width + j*2+1]+
		       top[top_batch_channel_loc + i*2*new_width + j*2-1]);
		 }
		 if(j==0){
		       bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j]+=0.5*
		       top[top_batch_channel_loc + i*2*new_width + j*2+1];
		 }
		 if(j==width-1){
		       bottom[batch_id*num_channel*height*width + channel_id*height*width + i*width+j]+=0.5*
		       top[top_batch_channel_loc + i*2*new_width + j*2-1];
		 }
	 
		if( (i*2+1<height*2-2) && (i*2+1 >=0) && (j*2+1 <width*2-2) && (j*2+1>=0))
			bottom[batch_id*num_channel*height*width + channel_id*height*width + (i)*width+j]+=0.25*top[top_batch_channel_loc+ (i*2+1)*new_width + j*2+1];
		if( (i*2+1<height*2-2) && (i*2+1 >=0) && (j*2-1 <width*2-2) && (j*2-1>=0))
			bottom[batch_id*num_channel*height*width + channel_id*height*width + (i)*width+j]+=0.25*top[top_batch_channel_loc+ (i*2+1)*new_width + j*2-1];
		if( (i*2-1<height*2-2) && (i*2-1 >=0) && (j*2+1 <width*2-2) && (j*2+1>=0))
			bottom[batch_id*num_channel*height*width + channel_id*height*width + (i)*width+j]+=0.25*top[top_batch_channel_loc+ (i*2-1)*new_width + j*2+1];
		if( (i*2-1<height*2-2) && (i*2-1 >=0) && (j*2-1 <width*2-2) && (j*2-1>=0))
			bottom[batch_id*num_channel*height*width + channel_id*height*width + (i)*width+j]+=0.25*top[top_batch_channel_loc+ (i*2-1)*new_width + j*2-1];

	}

	}
}

template <typename Dtype>
void InterpLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    //SinBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    //    count, top_diff, bottom_data, bottom_diff);
    //CUDA_POST_KERNEL_CHECK;
    InterpBackward<Dtype>(count, top_diff, bottom_diff, height_, width_, channels_, batch_);

    //dim3 numb(batch_/32+1, channels_/32);
    //dim3 block(32, 32);
    //InterpBackward<Dtype><<<numb,block>>>(top_diff, bottom_diff, height_, width_, channels_, batch_);
  }
}

INSTANTIATE_CLASS(InterpLayer);
REGISTER_LAYER_CLASS(Interp);


}  // namespace caffe
