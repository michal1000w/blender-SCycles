/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/util/diffraction_tensor.h"
#include <fstream>
#include <iostream>
#include <vector>
using namespace ccl;
std::vector<float> read(const char *path) {
  std::ifstream f(path,std::ios::binary|std::ios::ate);
  if(!f || f.tellg()<=0 || f.tellg()%4) return {};
  std::vector<float> v(size_t(f.tellg())/4);f.seekg(0);
  f.read(reinterpret_cast<char *>(v.data()),v.size()*4);return f?v:std::vector<float>{};
}
int main(int argc,char **argv) {
  if(argc!=5)return 2;
  auto model=read(argv[1]),queries=read(argv[2]),expected=read(argv[3]);
  int rank=std::stoi(argv[4]);
  if(model.empty() || queries.empty() || queries.size()%3 || expected.size()!=queries.size()/3*400)return 2;
  std::vector<float2> packed((model.size()+1)/2,make_float2(0,0));
  for(size_t i=0;i<model.size();++i) {
    if(i&1) packed[i/2].y=model[i]; else packed[i/2].x=model[i];
  }
  const DiffractionTensorFloatView packed_view{packed.data(),0};
  float2 chart[400];double maximum=0;
  for(size_t q=0;q<queries.size()/3;++q) {
    if(!diffraction_tensor_chart<20>(model.data(),model.size(),rank,make_float3(queries[3*q],queries[3*q+1],queries[3*q+2]),chart))return 3;
    float2 packed_chart[400];
    if(!diffraction_tensor_chart<20>(packed_view,model.size(),rank,make_float3(queries[3*q],queries[3*q+1],queries[3*q+2]),packed_chart))return 3;
    for(int i=0;i<400;++i)
      if(chart[i].x!=packed_chart[i].x || chart[i].y!=packed_chart[i].y)return 8;
    int upper=0;
    for(int r=0;r<20;++r) {
      maximum=std::max(maximum,double(std::abs(chart[r*20+r].y-expected[q*400+r])));
      for(int c=r+1;c<20;++c,++upper) {
        maximum=std::max(maximum,double(std::abs(chart[r*20+c].y-expected[q*400+20+upper]*M_SQRT1_2F)));
        maximum=std::max(maximum,double(std::abs(chart[r*20+c].x+expected[q*400+210+upper]*M_SQRT1_2F)));
        if(chart[c*20+r].x!=-chart[r*20+c].x || chart[c*20+r].y!=chart[r*20+c].y)return 4;
      }
    }
  }
  if(diffraction_tensor_chart<20>(model.data(),13,rank,make_float3(.5f,.5f,.5f),chart) ||
     diffraction_tensor_chart<20>(model.data(),model.size(),-1,make_float3(.5f,.5f,.5f),chart) ||
     diffraction_tensor_chart<20>(model.data(),model.size(),rank,make_float3(-.1f,.5f,.5f),chart))return 5;
  std::cout<<"{\"queries\":"<<queries.size()/3<<",\"rank\":"<<rank<<",\"maximum_chart_component_error\":"<<maximum<<"}\n";
  return maximum<1e-5?0:1;
}
