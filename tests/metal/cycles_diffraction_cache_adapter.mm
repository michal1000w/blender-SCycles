/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "device/metal/diffraction/reference_backend.h"
#include "scene/diffraction.h"
#include "scene/diffraction_convergence.h"
#include <cstdio>
#include <limits>
#include <mutex>
#include <atomic>
#ifdef DIFFRACTION_PACKAGED_SOURCE
#include "diffraction_shader_source.h"
#endif
struct Pair {float x,y;};
int main(int argc,const char **argv) {
#ifdef DIFFRACTION_PACKAGED_SOURCE
  if(argc!=1)return 2;
  const bool probe=false,modal=false,metal=true;
#else
  if(argc!=3&&argc!=4)return 2;
  const bool probe=argc==4&&std::string(argv[3])=="metal-probe";
  const bool modal=argc==4&&std::string(argv[3])=="metal-modal";
  const bool metal=probe||modal||(argc==4&&std::string(argv[3])=="metal");
  if(argc==4&&!metal)return 2;
#endif
  @autoreleasepool {
    try {
      NSError *error=nil;
#ifdef DIFFRACTION_PACKAGED_SOURCE
      NSString *source=[NSString stringWithUTF8String:diffraction_metal_shader_source];
#else
      NSString *base=[NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
      NSString *extra=[NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:&error];
      if(!base||!extra)return 2;
      NSString *source=[base stringByAppendingString:extra];
#endif
      bool missing_device_rejected=false;
      try { DiffractionResidentMatrix invalid(source,nil); }
      catch(const std::runtime_error &) { missing_device_rejected=true; }
      if(!missing_device_rejected)throw std::runtime_error("Missing device was accepted");
      id<MTLDevice> selected_device=MTLCreateSystemDefaultDevice();
      auto owner=std::make_shared<DiffractionMetalReferenceBackend>(source,selected_device);
      auto &engine=owner->engine;
      if(engine.device!=selected_device)throw std::runtime_error("GPU selection changed");
      fprintf(stderr,"Selected GPU: %s; registry ID: %llu; missing-device rejection passed\n",
              engine.device.name.UTF8String,(unsigned long long)engine.device.registryID);
      std::atomic<size_t> calls{0};
      auto reference_solver=diffraction_metal_reference_solver(owner);
      ccl::DiffractionReferenceSolver backend=[&](const auto &profile,double wavelength,double kx,double ky,int n,int retained,auto &out,auto &message) {
        ++calls;
        if(probe)fprintf(stderr,"GPU reference N%d\n",n);
        return reference_solver(profile,wavelength,kx,ky,n,retained,out,message);
      };
      ccl::DiffractionGratingProfile profile{740,30,.41,1,1.5,1,1.5};
      profile.ridge_spectrum={{590,1.4},{600.5,1.51},{610,1.6}};
      profile.groove_spectrum={{590,1.0},{610,1.02}};
      if(metal) {
        profile.ridge_ior=profile.substrate_ior={.9,6};
        profile.ridge_spectrum={{590,{.8,5.8}},{600.5,{.91,6.01}},{610,{1,6.2}}};
        profile.absorbing_substrate_spectrum=profile.ridge_spectrum;
      }
      if(probe) {
        NSMutableArray *reports=[NSMutableArray array];
        ccl::DiffractionGratingBlock gpu_reference;
        for(bool gpu:{true,false}) {
          ccl::DiffractionModalOptions convergence;
          convergence.minimum_half_orders=4;convergence.maximum_half_orders=diffraction_resident_max_orders;
          convergence.power_tolerance=0.001;
          if(gpu)convergence.reference_solver=backend;
          ccl::DiffractionGratingBlock block;std::vector<ccl::DiffractionModalObservation> observations;std::string message;
          bool accepted=ccl::diffraction_grating_converged_reference(profile,600,-0.00079180743243243248,-0.0009765625,2,convergence,block,observations,message);
          if(!accepted&&!block.matrix.empty())throw std::runtime_error("Unconverged result was published");
          NSMutableArray *trace=[NSMutableArray array];
          for(const auto &o:observations)[trace addObject:@{@"half_orders":@(o.half_orders),@"adjacent_power":@(o.adjacent_power_difference),@"spanning_power":@(o.spanning_power_difference)}];
          NSMutableDictionary *report=[@{@"backend":gpu?@"Metal":@"CPU",@"accepted":@(accepted),@"error":@(message.c_str()),@"observations":trace} mutableCopy];
          if(gpu) {
            report[@"sampled_peak_device_bytes"]=@(engine.sampled_peak_device_bytes);
            if(accepted)gpu_reference=block;
          }
          if(!gpu&&accepted&&!gpu_reference.matrix.empty()) {
            if(block.matrix.size()!=gpu_reference.matrix.size())throw std::runtime_error("Accepted backend shapes differ");
            double maximum=0;
            for(size_t i=0;i<block.matrix.size();++i)maximum=std::max(maximum,std::abs(block.matrix[i]-gpu_reference.matrix[i]));
            ccl::DiffractionGratingBlock gp,cp;
            if(!ccl::diffraction_grating_match_reference(profile,600,-0.00079180743243243248,-0.0009765625,gpu_reference,gp,message)||
               !ccl::diffraction_grating_match_reference(profile,600,-0.00079180743243243248,-0.0009765625,block,cp,message))throw std::runtime_error(message);
            ccl::DiffractionGratingPowerBlock gpower,cpower;
            ccl::diffraction_grating_power_block(gp,gpower);ccl::diffraction_grating_power_block(cp,cpower);
            if(gpower.matrix.size()!=cpower.matrix.size())throw std::runtime_error("Accepted physical shapes differ");
            double power_max=0,column_max=0;size_t ports=gpower.ports.size();
            for(size_t col=0;col<ports;++col) {
              double column=0;
              for(size_t row=0;row<ports;++row) {
                double delta=std::abs(gpower.matrix[row*ports+col]-cpower.matrix[row*ports+col]);
                if(!std::isfinite(delta))throw std::runtime_error("Nonfinite accepted power difference");
                power_max=std::max(power_max,delta);column+=delta;
              }
              column_max=std::max(column_max,column);
            }
            report[@"gpu_cpu_reference_max_error"]=@(maximum);
            report[@"gpu_cpu_order_power_max_error"]=@(power_max);
            report[@"gpu_cpu_column_power_l1_max_error"]=@(column_max);
            report[@"reference_component_gate_pass"]=@(maximum<3e-4);
          }
          [reports addObject:report];
        }
        NSData *json=[NSJSONSerialization dataWithJSONObject:reports options:NSJSONWritingPrettyPrinted error:&error];
        if(!json)return 2;fwrite(json.bytes,1,json.length,stdout);puts("");return 0;
      }
      ccl::DiffractionGratingCacheOptions options;
      options.bounds={{-0.0009765625,-0.0009765625,600},{0.0009765625,0.0009765625,601}};
      options.half_orders=4;options.retained_half_orders=2;
      if(modal){options.modal_power_tolerance=0.001;options.maximum_half_orders=16;}
      options.maximum_nodes=31;options.maximum_depth=4;options.validation_workers=2;
      options.reference_solver=backend;options.reference_backend_key="test-metal-wide-v1";
      ccl::DiffractionGratingCache gpu,cpu; ccl::DiffractionGratingCacheStats gs,cs;std::string message;
      if(!ccl::diffraction_grating_build_cache(profile,options,gpu,gs,message))throw std::runtime_error(message);
      if(gpu.cells.empty()||!calls)throw std::runtime_error("No GPU cache constructed");
      options.reference_solver={};options.reference_backend_key.clear();
      if(!ccl::diffraction_grating_build_cache(profile,options,cpu,cs,message))throw std::runtime_error(message);
      ccl::DiffractionGratingDeviceBuffers gb,cb;
      if(!ccl::diffraction_grating_device_buffers(gpu,gb,message)||!ccl::diffraction_grating_device_buffers(cpu,cb,message))throw std::runtime_error(message);
      if(gb.matrices.size()!=cb.matrices.size()||gb.layout.size()!=cb.layout.size())return 3;
      double maximum=0;
      for(size_t i=0;i<gb.matrices.size();++i)maximum=std::max(maximum,std::hypot(double(gb.matrices[i].x)-cb.matrices[i].x,double(gb.matrices[i].y)-cb.matrices[i].y));
      if(maximum>3e-4)return 4;
      options.reference_solver=backend;options.reference_backend_key="test-metal-wide-v1";
      options.progress=[](const auto &){return false;};ccl::DiffractionGratingCache cancelled;
      if(ccl::diffraction_grating_build_cache(profile,options,cancelled,cs,message)||!cancelled.cells.empty())return 5;
      printf("GPU cache cells %zu, reference calls %zu, packed coefficient max difference %.9g; cancellation rejected without publication\n",gpu.cells.size(),calls.load(),maximum);
      options.progress={};
      options.modal_power_tolerance=0.001;options.maximum_half_orders=16;
      const size_t before=calls.load();
      options.cancelled=[&] { return calls.load()>before; };
      ccl::DiffractionGratingBlock interrupted;size_t solves=0;
      if(ccl::diffraction_grating_cache_reference(profile,options,600,.01,.02,
          interrupted,solves,message)||calls.load()!=before+1||solves!=1||
          !interrupted.matrix.empty()||message.find("cancelled")==std::string::npos)return 6;
      // The same GPU engine must remain usable after a cancelled query.
      options.cancelled={};options.modal_power_tolerance=0;
      if(!ccl::diffraction_grating_cache_reference(profile,options,600,.01,.02,
          interrupted,solves,message)||interrupted.matrix.empty()||solves!=1)return 7;
      const auto before_failure=interrupted.matrix;
      const auto recovery_matches=[&] {
        if(interrupted.matrix.size()!=before_failure.size())return false;
        double maximum=0;
        for(size_t i=0;i<before_failure.size();++i) {
          const double difference=std::abs(interrupted.matrix[i]-before_failure[i]);
          if(!std::isfinite(difference))return false;
          maximum=std::max(maximum,difference);
        }
        printf("Recovery maximum component difference %.9g (gate 3e-4)\n",maximum);
        return maximum<3e-4;
      };
      printf("Mid-refinement GPU cancellation: one completed solve, empty cancelled output; same-engine recovery passed\n");
      const NSUInteger downloads_before=engine.downloads;
      unsigned cancellation_checks=0;
      engine.cancelled=[&] { return ++cancellation_checks>=10; };
      if(ccl::diffraction_grating_cache_reference(profile,options,600,.01,.02,
          interrupted,solves,message)||!interrupted.matrix.empty()||
          message.find("cancelled")==std::string::npos||engine.downloads!=downloads_before||
          cancellation_checks!=10)return 8;
      engine.cancelled={};
      if(!ccl::diffraction_grating_cache_reference(profile,options,600,.01,.02,
          interrupted,solves,message)||!recovery_matches())return 9;
      printf("In-solve GPU cancellation: stopped at checkpoint 10 without payload download; same-engine recovery passed\n");
      unsigned failure_checks=0;
      engine.cancelled=[&] {
        if(++failure_checks==10)throw std::runtime_error("Injected query exception");
        return false;
      };
      if(ccl::diffraction_grating_cache_reference(profile,options,600,.01,.02,
          interrupted,solves,message)||!interrupted.matrix.empty()||
          message.find("Injected query exception")==std::string::npos)return 12;
      engine.cancelled={};
      if(!ccl::diffraction_grating_cache_reference(profile,options,600,.01,.02,
          interrupted,solves,message)||!recovery_matches())return 13;
      printf("Injected query exception discarded pending work; same-engine recovery passed\n");
      std::weak_ptr<DiffractionMetalReferenceBackend> lifetime=owner;
      owner.reset();
      if(lifetime.expired()||!reference_solver(profile,600,.01,.02,4,2,interrupted,message))return 10;
      reference_solver={};
      if(!lifetime.expired())return 11;
      printf("Reference callback owns backend until release; query after creator release passed\n");
      return 0;
    }catch(const std::exception &e){fprintf(stderr,"%s\n",e.what());return 2;}
  }
}
