/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction.h"
#include "kernel/util/diffraction_scene_data.h"
#include <iostream>
using namespace ccl;
int main() {
  DiffractionGratingPackedCell cell;
  cell.bounds={{-.5,-1,380},{.5,1,780}};
  cell.ports={{0,false}};cell.active_ports={0};
  cell.operator_chart=true;cell.chart_degree=3;cell.tensor_rank=0;
  cell.matrices={make_float2(1,0),make_float2(0,0),make_float2(0,0),make_float2(1,0)};
  float model[18]={};
  const float weights[7]={-.05f,.3f,-.75f,1,-.75f,.3f,-.05f};
  for(int i=0;i<7;++i){model[i]=float(i)/6;model[7+i]=weights[i];}
  for(int i=0;i<18;i+=2)cell.matrices.push_back(make_float2(model[i],model[i+1]));
  DiffractionGratingCache cache;cache.bounds=cell.bounds;cache.nodes={make_int4(-1,0,0,0)};cache.cells={cell};
  DiffractionGratingDeviceBuffers buffers;std::string error;
  if(!diffraction_grating_device_buffers(cache,buffers,error)){std::cerr<<error;return 1;}
  int4 descriptors[2]={make_int4(0,1,0,1),make_int4(0,0,0,0)};
  float4 domains[2]={make_float4(-.5f,-1,380,0),make_float4(.5f,1,780,0)};
  DiffractionSceneData data{buffers.nodes.data(),buffers.layout.data(),buffers.bounds.data(),
      buffers.ports.data(),buffers.active.data(),buffers.matrices.data(),descriptors,domains,1};
  for(float wavelength:{380.f,580.f,780.f}) {
    DiffractionCacheCellView view;int incoming;float power[1];
    if(!diffraction_data_power_column<2>(&data,0,make_float3(0,0,-1),false,1,1,wavelength,740,&view,&incoming,power))return 2;
    if(view.degree!=3 || view.tensor_rank!=0 || view.tensor_floats!=18 || std::abs(power[0]-1)>1e-6)return 3;
    DiffractionSceneSample result;
    if(!diffraction_data_sample<2>(&data,0,make_float3(0,0,-1),false,1,1,wavelength,740,.25f,&result))return 4;
    if(result.transmission || result.relative_order || std::abs(result.probability-1)>1e-6 ||
       std::abs(result.throughput-1)>1e-6 || result.direction.z<.999f)return 5;
  }
  int rejected=0;
  auto reject=[&](DiffractionGratingPackedCell changed){
    cache.cells={changed};DiffractionGratingDeviceBuffers out;
    if(diffraction_grating_device_buffers(cache,out,error))return false;
    ++rejected;return out.nodes.empty()&&out.matrices.empty();
  };
  auto bad=cell;bad.matrices.pop_back();if(!reject(bad))return 6;
  bad=cell;bad.tensor_rank=-1;if(!reject(bad))return 6;
  bad=cell;bad.active_ports.clear();if(!reject(bad))return 6;
  bad=cell;bad.matrices[0].x=2;if(!reject(bad))return 6;
  bad=cell;bad.matrices[4].y=0;if(!reject(bad))return 6;
  bad=cell;bad.matrices[7].y=0;if(!reject(bad))return 6;
  std::cout<<"{\"scene_lookup_matching_sampling\":true,\"malformed_payloads_rejected\":"<<rejected<<"}\n";
}
