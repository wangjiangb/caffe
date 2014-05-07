// Copyright 2013 Jiang Wang
// This program evaluate a network given the saved network parameters


#include <cuda_runtime.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>

#include <cstring>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/solver.hpp"

using namespace caffe;

template <typename Dtype>
void TestNet(const shared_ptr<Net<float> >& test_net) {
  vector<Dtype> test_score;
  vector<Blob<Dtype>*> bottom_vec;
  const vector<Blob<Dtype>*>& result =
    test_net->Forward(bottom_vec);
  for (int j = 0; j < result.size(); ++j) {
    const Dtype* result_vec = result[j]->cpu_data();
    for (int k = 0; k < result[j]->count(); ++k) {
      test_score.push_back(result_vec[k]);
    }
  }
  for (int i = 0; i < test_score.size(); ++i) {
    LOG(INFO) << "Test score #" << i << ": "
              << test_score[i];
  }
}

int main(int argc, char** argv) {
  cudaSetDevice(1);
  Caffe::set_mode(Caffe::GPU);
  Caffe::set_phase(Caffe::TEST);

  NetParameter trained_net_param;
  ReadProtoFromBinaryFile(argv[1], &trained_net_param);

  NetParameter test_net_param;
  shared_ptr<Net<float> > test_net;
  ReadProtoFromTextFile(argv[2], &test_net_param);
  test_net.reset(new Net<float>(test_net_param));
  test_net->CopyTrainedLayersFrom(trained_net_param);
}
