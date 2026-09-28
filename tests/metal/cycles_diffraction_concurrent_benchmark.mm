/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "cycles_diffraction_resident_response.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>
#ifdef DIFFRACTION_TEST_POOL
#include "device/metal/diffraction/reference_pool.h"
#endif

struct Pair { float x,y; };
struct Fixture {
  DiffractionResidentProfile profile;
  std::vector<std::pair<double,double>> expected;
};

int main(int argc,const char **argv)
{
  if(argc!=4)return 2;
  @autoreleasepool {
    try {
      NSError *error=nil;
      NSString *base=[NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
      NSString *extra=[NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:&error];
      NSData *data=[NSData dataWithContentsOfFile:@(argv[3])];
      if(!base||!extra||!data)throw std::runtime_error("Missing benchmark input");
      NSArray *inputs=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
      if(![inputs isKindOfClass:[NSArray class]]||!inputs.count)throw std::runtime_error("Invalid fixtures");
      std::vector<Fixture> fixtures;
      for(NSDictionary *f in inputs) {
        NSArray *ridge=f[@"ridge"];
        Fixture item;
        item.profile={[f[@"half_orders"] unsignedIntValue],[f[@"retained_half_orders"] unsignedIntValue],
          [f[@"pitch_nm"] floatValue],[f[@"wavelength_nm"] floatValue],[f[@"depth_nm"] floatValue],
          [f[@"duty"] floatValue],[f[@"incident_ior"] floatValue],[ridge[0] floatValue],[ridge[1] floatValue],
          [f[@"incident_ior"] floatValue],0,[ridge[0] floatValue],[ridge[1] floatValue],
          [f[@"kx"] floatValue],[f[@"ky"] floatValue]};
        if(item.profile.half_orders>32)throw std::runtime_error("Small-solve benchmark only");
        for(NSArray *z in f[@"matrix"])item.expected.emplace_back([z[0] doubleValue],[z[1] doubleValue]);
        fixtures.push_back(std::move(item));
      }
      id<MTLDevice> device=MTLCreateSystemDefaultDevice();
#ifdef DIFFRACTION_TEST_POOL
      DiffractionMetalReferencePool pool([base stringByAppendingString:extra],device);
#else
      std::vector<std::unique_ptr<DiffractionResidentMatrix>> engines;
      for(int i=0;i<2;++i)engines.push_back(std::make_unique<DiffractionResidentMatrix>([base stringByAppendingString:extra],device));
#endif
      NSMutableArray *runs=[NSMutableArray array];
      // Warm each configuration, then alternate to reduce ordering bias.
      for(unsigned pass=0;pass<8;++pass) {
        unsigned workers=pass%2+1;
#ifdef DIFFRACTION_TEST_POOL
        workers=pass%2?6:1;
#endif
        std::atomic<size_t> next{0};
        std::vector<double> errors(fixtures.size(),0);
        std::vector<std::string> failures(workers);
        auto start=std::chrono::steady_clock::now();
        std::vector<std::thread> threads;
        for(unsigned worker=0;worker<workers;++worker)threads.emplace_back([&,worker] {
          try {
            for(size_t i;(i=next.fetch_add(1))<fixtures.size();) {
              @autoreleasepool {
                const auto &f=fixtures[i];
                std::vector<Pair> actual(f.expected.size());
#ifdef DIFFRACTION_TEST_POOL
                const auto &p=f.profile;
                ccl::DiffractionGratingProfile profile{p.pitch,p.depth,p.duty,p.incident,
                    {p.ridge_n,p.ridge_k},{p.groove_n,p.groove_k},{p.substrate_n,p.substrate_k}};
                ccl::DiffractionGratingBlock block;std::string error;
                if(!pool.solve(profile,p.wavelength,p.kx,p.ky,p.half_orders,p.retained,block,error))
                  throw std::runtime_error(error);
                if(block.matrix.size()!=actual.size())throw std::runtime_error("Pool shape mismatch");
                for(size_t j=0;j<actual.size();++j)actual[j]={float(block.matrix[j].real()),float(block.matrix[j].imag())};
#else
                auto &engine=*engines[worker];unsigned steps=0;
                auto response=diffraction_resident_response(engine,f.profile,steps);
                if(response.rows*response.cols!=f.expected.size())throw std::runtime_error("Shape mismatch");
                engine.download(response,actual.data());
#endif
                for(size_t j=0;j<actual.size();++j) {
                  double delta=std::hypot(double(actual[j].x)-f.expected[j].first,double(actual[j].y)-f.expected[j].second);
                  if(!std::isfinite(delta))throw std::runtime_error("Nonfinite response");
                  errors[i]=std::max(errors[i],delta);
                }
                if(errors[i]>=3e-4)throw std::runtime_error("Reference component gate failed");
              }
            }
          }catch(const std::exception &e){failures[worker]=e.what();}
        });
        for(auto &thread:threads)thread.join();
        for(const auto &failure:failures)if(!failure.empty())throw std::runtime_error(failure);
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        [runs addObject:@{@"workers":@(workers),@"warmup":@(pass<2),@"milliseconds":@(ms),
          @"max_component_error":@(*std::max_element(errors.begin(),errors.end())),@"cases":@(fixtures.size())}];
      }
      NSDictionary *report=@{@"device":device.name,@"runs":runs,
        @"scope":@"One GPU benchmark job; alternating whole-suite timings including thread startup and readback. Pool variant includes lazy initialization in warmup only."};
      NSData *json=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
      if(!json)return 2;fwrite(json.bytes,1,json.length,stdout);puts("");return 0;
    }catch(const std::exception &e){fprintf(stderr,"%s\n",e.what());return 2;}
  }
}
