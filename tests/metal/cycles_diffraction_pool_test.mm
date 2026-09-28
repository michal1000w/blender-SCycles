/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "device/metal/diffraction/reference_pool.h"
#include <atomic>
#include <cstdio>
#include <thread>

int main(int argc,const char **argv)
{
  if(argc!=3)return 2;
  @autoreleasepool {
    try {
      NSError *error=nil;
      NSString *base=[NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
      NSString *extra=[NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:&error];
      if(!base||!extra)throw std::runtime_error("Missing shader source");
      std::atomic<bool> cancel{false};
      id<MTLDevice> device=MTLCreateSystemDefaultDevice();
      DiffractionMetalReferencePool pool([base stringByAppendingString:extra],device,[&]{return cancel.load();});
      const ccl::DiffractionGratingProfile profile{740,30,double(float(.41)),1,{.9,6},1,{.9,6}};
      const std::array<int,8> orders{4,16,64,32,4,128,16,64};
      std::array<ccl::DiffractionGratingBlock,8> expected,actual;
      std::array<std::string,8> messages;
      for(size_t i=0;i<orders.size();++i)
        if(!ccl::diffraction_grating_solve_reference(profile,600+i,.01,.02,orders[i],2,expected[i],messages[i]))
          throw std::runtime_error(messages[i]);
      std::atomic<size_t> next{0};std::array<bool,8> success{};
      std::vector<std::thread> workers;
      for(int worker=0;worker<6;++worker)workers.emplace_back([&] {
        for(size_t i;(i=next.fetch_add(1))<orders.size();)
          success[i]=pool.solve(profile,600+i,.01,.02,orders[i],2,actual[i],messages[i]);
      });
      for(auto &worker:workers)worker.join();
      double maximum=0;
      for(size_t i=0;i<orders.size();++i) {
        if(!success[i])throw std::runtime_error(messages[i]);
        if(actual[i].matrix.size()!=expected[i].matrix.size())throw std::runtime_error("Mixed-size shape mismatch");
        for(size_t j=0;j<actual[i].matrix.size();++j) {
          double delta=std::abs(actual[i].matrix[j]-expected[i].matrix[j]);
          if(!std::isfinite(delta))throw std::runtime_error("Nonfinite mixed-size result");
          maximum=std::max(maximum,delta);
        }
      }
      if(maximum>=3e-4)throw std::runtime_error("Mixed-size component gate failed");
      ccl::DiffractionGratingBlock result=actual[0];std::string message;
      if(pool.solve(profile,600,.01,.02,257,2,result,message)||!result.matrix.empty())
        throw std::runtime_error("Invalid large query accepted");
      cancel=true;
      if(pool.solve(profile,600,.01,.02,4,2,result,message)||!result.matrix.empty()||
          message.find("cancelled")==std::string::npos)throw std::runtime_error("Pre-cancellation failed");
      cancel=false;
      if(!pool.solve(profile,600,.01,.02,4,2,result,message)||result.matrix.empty())
        throw std::runtime_error("Pool recovery failed");
      printf("Mixed N4/N16/N32/N64/N128 queries with six callers passed; max component error %.12g\n",maximum);
      puts("Invalid exclusive query, pre-cancellation and subsequent pool recovery passed");
      ccl::DiffractionGratingBlock large_result, waiting_result;
      std::string large_error, waiting_error;
      bool large_ok=false,waiting_ok=false;
      auto wait_for_state=[&](auto predicate) {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        while(!predicate(pool.scheduling_state())) {
          if(std::chrono::steady_clock::now()>deadline)return false;
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return true;
      };
      std::jthread large([&] {
        large_ok=pool.solve(profile,600,.01,.02,256,2,large_result,large_error);
      });
      if(!wait_for_state([](auto s){return s.active==1&&s.exclusive;})) {
        cancel=true;large.join();throw std::runtime_error("Exclusive query did not become active");
      }
      std::jthread waiting([&] {
        waiting_ok=pool.solve(profile,600,.01,.02,4,2,waiting_result,waiting_error);
      });
      if(!wait_for_state([](auto s){return s.active==1&&s.exclusive&&s.waiting==1;})) {
        cancel=true;waiting.join();large.join();throw std::runtime_error("Small query was not observed waiting");
      }
      cancel=true;
      waiting.join();large.join();
      if(waiting_ok||large_ok||!waiting_result.matrix.empty()||!large_result.matrix.empty()||
          waiting_error.find("pool wait cancelled")==std::string::npos||
          large_error.find("cancelled")==std::string::npos)
        throw std::runtime_error("In-flight/waiting cancellation failed");
      const auto idle=pool.scheduling_state();
      if(idle.active||idle.waiting||idle.exclusive)throw std::runtime_error("Pool scheduling state leaked");
      cancel=false;
      if(!pool.solve(profile,600,.01,.02,4,2,result,message))throw std::runtime_error(message);
      puts("Observed queued query cancellation, exclusive-solve cancellation, idle state and recovery passed");
      return 0;
    }catch(const std::exception &e){fprintf(stderr,"%s\n",e.what());return 2;}
  }
}
