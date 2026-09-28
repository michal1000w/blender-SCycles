/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/util/diffraction_tensor.h"
#define DiffractionTensorFloatView CandidateFloatView
#define diffraction_tensor_basis candidate_basis
#define diffraction_tensor_chart candidate_chart
#include "diffraction_tensor_nodal_candidate.h"
#undef DiffractionTensorFloatView
#undef diffraction_tensor_basis
#undef diffraction_tensor_chart
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
using namespace ccl;
int main(int argc,char **argv) {
  if(argc!=3)return 2;
  std::ifstream in(argv[1],std::ios::binary|std::ios::ate);
  if(!in||in.tellg()<=0||in.tellg()%4)return 2;
  std::vector<float> model(size_t(in.tellg())/4);in.seekg(0);
  in.read(reinterpret_cast<char *>(model.data()),model.size()*4);
  const int rank=std::stoi(argv[2]);
  std::vector<float3> queries;
  for(int x=0;x<7;x++)for(int y=0;y<7;y++)for(int z=0;z<7;z++)
    queries.push_back(make_float3(model[x],model[y],model[z]));
  std::mt19937 random(73812);std::uniform_real_distribution<float> unit(0,1);
  for(int i=0;i<2048;i++) {
    queries.push_back(make_float3(unit(random),unit(random),unit(random)));
    queries.push_back(make_float3(model[i%7],model[(i/7)%7],unit(random)));
  }
  double maximum=0;int differences=0;
  for(auto q:queries) {
    float2 a[400],b[400];
    if(!diffraction_tensor_chart<20>(model.data(),model.size(),rank,q,a)||
       !candidate_chart<20>(model.data(),model.size(),rank,q,b))return 3;
    for(int i=0;i<400;i++) {
      maximum=std::max(maximum,double(std::max(fabsf(a[i].x-b[i].x),fabsf(a[i].y-b[i].y))));
      differences+=a[i].x!=b[i].x||a[i].y!=b[i].y;
    }
  }
  std::cout<<"{\"queries\":"<<queries.size()<<",\"rank\":"<<rank<<",\"different_complex_components\":"<<differences<<",\"maximum_error\":"<<maximum<<"}\n";
  return differences?1:0;
}
