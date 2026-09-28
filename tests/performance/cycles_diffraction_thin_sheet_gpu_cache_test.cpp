/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "device/device.h"
#include "scene/diffraction_albedo.h"
#include "util/path.h"
#include "util/profiling.h"
#include "util/stats.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
using namespace ccl;
int main(int argc,char **argv)
{
  if(argc!=3){std::cerr<<"Usage: TEST OUTPUT.json SOURCE_ROOT\n";return 2;}
  path_init(argv[2]);auto devices=Device::available_devices(DEVICE_MASK_METAL);
  if(devices.empty())return 2;
  Stats stats;Profiler profiler;auto device=Device::create(devices.front(),stats,profiler,true);
  if(!device || device->have_error())return 2;
  std::ofstream out(argv[1]);if(!out)return 2;out.precision(9);
  out<<"{\"scope\":\"actual CPU/Metal one-sided and ThinSheet cache parity\",\"cases\":[\n";
  bool all=true;
  for(int kind=0;kind<5;kind++){
    DiffractionAlbedoRequest r;r.alpha_x=.36f;r.alpha_y=.27f;
    r.pitch_nm=1150;r.depth_nm=320;r.duty=.42f;
    r.wavelength_count=4;r.mu_count=6;r.phi_count=8;r.facet_samples=256;
    if(kind==4){r.wavelength_count=16;r.mu_count=24;r.phi_count=16;r.facet_samples=512;}
    r.thin_sheet=kind!=0;
    if(kind==2)r.transmission_tint=.8f;
    if(kind==3){r.transmission_is_spectral=true;r.transmission_bt709=make_float3(.2f,.5f,.8f);r.film_ior=1.32f;r.film_thickness_nm=250;r.inv_abbe=.02f;}
    DiffractionAlbedoTable cpu,metal;std::string error;
    const auto cpu_start=std::chrono::steady_clock::now();
    const bool cpu_built=diffraction_albedo_build_cpu(r,cpu,error);
    const auto metal_start=std::chrono::steady_clock::now();
    const bool metal_built=cpu_built && device->build_diffraction_albedo(r,metal,error);
    const auto metal_end=std::chrono::steady_clock::now();
    const double cpu_seconds=std::chrono::duration<double>(metal_start-cpu_start).count();
    const double metal_seconds=std::chrono::duration<double>(metal_end-metal_start).count();
    bool built=cpu_built && metal_built;
    bool finite=true;
    double directional=0,average=0;
    if(built && cpu.values.size()==metal.values.size() && cpu.averages.size()==metal.averages.size()){
      for(size_t i=0;i<cpu.values.size();i++){
        finite &= std::isfinite(cpu.values[i]) && std::isfinite(metal.values[i]);
        if(std::isfinite(cpu.values[i]) && std::isfinite(metal.values[i]))directional=std::max(directional,std::abs(double(cpu.values[i])-metal.values[i]));
      }
      for(size_t i=0;i<cpu.averages.size();i++){
        finite &= std::isfinite(cpu.averages[i]) && std::isfinite(metal.averages[i]);
        if(std::isfinite(cpu.averages[i]) && std::isfinite(metal.averages[i]))average=std::max(average,std::abs(double(cpu.averages[i])-metal.averages[i]));
      }
    }else built=false;
    bool pass=built && finite && directional<=.002 && average<=.0005;all &=pass;
    out<<"{\"kind\":"<<kind<<",\"grid_scope\":\""<<(kind==4?"shipping white ThinSheet":"small-grid branch parity")<<"\",\"built\":"<<(built?"true":"false")<<",\"all_finite\":"<<(finite?"true":"false")<<",\"wavelength_count\":"<<r.wavelength_count<<",\"mu_count\":"<<r.mu_count<<",\"phi_count\":"<<r.phi_count<<",\"facet_samples\":"<<r.facet_samples<<",\"cpu_build_seconds\":"<<cpu_seconds<<",\"metal_build_seconds\":"<<metal_seconds<<",\"max_q_or_albedo_error\":"<<directional<<",\"max_projected_average_error\":"<<average<<",\"passed\":"<<(pass?"true":"false")<<"}"<<(kind==4?"\n":",\n");
    std::cout<<"case "<<kind<<" built="<<built<<" dir="<<directional<<" avg="<<average<<" "<<error<<"\n";
  }
  out<<"],\"passed\":"<<(all?"true":"false")<<"}\n";return all?0:1;
}
