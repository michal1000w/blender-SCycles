/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction.h"
#include "kernel/tables.h"
#include "kernel/util/diffraction_scene_data.h"
#include <fstream>
#include <algorithm>
#include <array>
#include <iostream>
#include <iomanip>
using namespace ccl;
int main(int argc,char **argv)
{
  if(argc!=2)return 2;
  std::ifstream input(argv[1],std::ios::binary);
  DiffractionGratingDeviceBuffers buffers;
  auto read=[&](auto &v) {
    uint64_t count=0;input.read(reinterpret_cast<char *>(&count),sizeof(count));
    if(!input||count>10000000)return false;
    v.resize(count);input.read(reinterpret_cast<char *>(v.data()),count*sizeof(v[0]));
    return bool(input);
  };
  if(!read(buffers.nodes)||!read(buffers.layout)||!read(buffers.bounds)||!read(buffers.ports)||
     !read(buffers.active)||!read(buffers.matrices)||input.peek()!=EOF)return 2;
  int4 descriptors[2]={make_int4(0,buffers.nodes.size(),0,buffers.layout.size()/2),make_int4(0,0,0,1)};
  float4 domains[2]={make_float4(-.5,-1,380,0),make_float4(0,0,780,0)};
  DiffractionSceneData data{buffers.nodes.data(),buffers.layout.data(),buffers.bounds.data(),
    buffers.ports.data(),buffers.active.data(),buffers.matrices.data(),descriptors,domains,1};
  const DiffractionGratingProfile profile{740,150,double(float(.41)),1,1.5,1,1};
  const double conversion[3][3]={{3.2404542,-1.5371385,-.4985314},
    {-.9692660,1.8760108,.0415560},{.0556434,-.2040259,1.0572252}};
  std::vector<double> edges;
  for(int w=380;w<=780;w+=5)edges.push_back(w);
  edges.push_back(.9*740);std::sort(edges.begin(),edges.end());
  std::cout<<std::setprecision(14)<<"{\"kind\":\"cached scene power integration\",\"color_space\":\"linear BT.709\",\"runs\":[";
  bool first=true;
  for(int modes:{16})for(int subdivision:{1,2,4,8}) {
    double rgb[2][3][3]={},neutral[3]={};int evaluations=0;
    for(size_t interval=1;interval<edges.size();interval++)for(int sample=0;sample<subdivision;sample++) {
      const double width=(edges[interval]-edges[interval-1])/subdivision;
      const double wavelength=edges[interval-1]+width*(sample+.5);
      const double coordinate=(wavelength-380)/5;const int i=int(coordinate);const double f=coordinate-i;
      double xyz[3],weight[3]={};
      for(int c=0;c<3;c++)xyz[c]=((1-f)*cie_color_match[i][c]+f*cie_color_match[i+1][c])*
        ((1-f)*cie_d65_spd[i]+f*cie_d65_spd[i+1])*CIE_D65_NORMALIZATION*width/400;
      for(int c=0;c<3;c++)for(int j=0;j<3;j++)weight[c]+=conversion[c][j]*xyz[j];
      for(int c=0;c<3;c++)neutral[c]+=weight[c];
      DiffractionCacheCellView view;int incoming;float powers[10];
      if(!diffraction_data_power_column<20>(&data,0,make_float3(0,0,-1),false,1,1,
                                           float(wavelength),740,&view,&incoming,powers))return 3;
      evaluations++;
      for(int port=0;port<view.ports;port++) {
        const int2 order=buffers.ports[view.port_offset+port];
        const double x=std::abs(order.x*wavelength/740);
        const int region=x<.25?0:(x>.9?2:1);
        for(int c=0;c<3;c++)rgb[order.y?1:0][region][c]+=powers[port]*weight[c];
      }
    }
    if(!first)std::cout<<",";first=false;
    std::cout<<"{\"half_orders\":"<<modes<<",\"subdivision\":"<<subdivision<<",\"evaluations\":"<<evaluations<<",\"neutral_rgb\":[";
    for(int c=0;c<3;c++)std::cout<<(c?",":"")<<neutral[c];
    std::cout<<"],\"regions\":[";
    for(int side=0;side<2;side++)for(int region=0;region<3;region++) {
      if(side||region)std::cout<<",";
      std::cout<<"{\"side\":\""<<(side?"transmission":"reflection")<<"\",\"region\":\""
        <<(region==0?"central":region==1?"middle":"outer")<<"\",\"rgb\":[";
      for(int c=0;c<3;c++)std::cout<<(c?",":"")<<rgb[side][region][c];
      std::cout<<"]}";
    }
    std::cout<<"]}"<<std::flush;
  }
  std::cout<<"]}\n";
}
