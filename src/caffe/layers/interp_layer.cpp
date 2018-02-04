#include <algorithm>
#include <vector>


#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer_factory.hpp"

#include "caffe/layers/interp_layer.hpp"
using namespace std;

namespace caffe{

template <typename Dtype>
void InterpLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top){

std::cout << "in interp layer!"<<std::endl;
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
  top[0]->Reshape(batch_, channels_, height_*2,
      width_*2);

}

template <typename Dtype>
void InterpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void InterpLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

INSTANTIATE_CLASS(InterpLayer);
REGISTER_LAYER_CLASS(Interp);
}
