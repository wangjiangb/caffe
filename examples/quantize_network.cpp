// Copyright 2013 Jiang Wang
// This program get the data from a trained network, and
// quantize it.


#include <cuda_runtime.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/solver.hpp"

using namespace caffe;


void quantizeVector(vector<float>& data, int quantize_size)
{
//std::cout << "Beginning Quantization" << std::endl;
  float data_sum = 0.0;
  for (int i = 0; i < data.size(); ++i)
    data_sum += data[i];
  float data_min = *std::min_element(data.begin(), data.end());
  float data_max = *std::max_element(data.begin(), data.end());
  float interval = (data_max - data_min) / quantize_size;
  for (int i = 0; i < data.size(); ++i) {
      // std::cout << "From: " << data[i] << " to " <<
      // floor((data[i] - data_min) / interval) * interval +
      // data_min << std::endl;
    data[i] = floor((data[i] - data_min) / interval) * interval +
      data_min;
  }
}

void quantizeBlob(BlobProto* blob, int quantize_size)
{
  CHECK(quantize_size > 1);
  CHECK_NOTNULL(blob);
  vector<float> data;
  if (blob->data_size() > 0) {
    data.reserve(blob->data_size());
    for (int i = 0; i < blob->data_size(); ++i) {
      data.push_back(blob->data(i));
    }
    quantizeVector(data, quantize_size);
    CHECK(data.size() == blob->data_size());
    for (int i = 0; i < blob->data_size(); ++i) {
      blob->set_data(i, data[i]);
    }
  }
  vector<float> diff;
  if (blob->diff_size() > 0) {
    diff.reserve(blob->diff_size());
    for (int i = 0; i < blob->diff_size(); ++i) {
      diff.push_back(blob->diff(i));
    }
    quantizeVector(diff, quantize_size);
    CHECK(diff.size() == blob->diff_size());
    for (int i = 0; i < blob->diff_size(); ++i) {
      blob->set_diff(i, diff[i]);
    }
  }
}

int main(int argc, char** argv) {
  cudaSetDevice(1);
  Caffe::set_mode(Caffe::GPU);
  Caffe::set_phase(Caffe::TEST);

  CHECK(argc == 4);

  LOG(INFO) << "Reading trained parameters";
  NetParameter trained_net_param;
  ReadProtoFromBinaryFile(argv[1], &trained_net_param);

  int quant_size = atoi(argv[3]);
  LOG(INFO) << "Quantizing trained parameters";
  NetParameter quantized_net_param = trained_net_param;
  for (int i = 0; i < quantized_net_param.layers_size(); ++i) {
    LayerParameter* layer = quantized_net_param.mutable_layers(i)->mutable_layer();
    for (int blob_id = 0; blob_id < layer->blobs_size(); ++blob_id) {
      quantizeBlob(layer->mutable_blobs(blob_id), quant_size);
    }
  }

  LOG(INFO) << "Outputting quantized parameters";
  WriteProtoToBinaryFile(quantized_net_param, argv[2]);
  LOG(INFO) << "Finish writing quantized parameters";

  return 0;
}
