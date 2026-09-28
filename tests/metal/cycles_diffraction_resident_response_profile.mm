/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "cycles_diffraction_resident_response.h"
#include <cstdio>
#include <limits>
#include <chrono>
struct Pair{float x,y;};
int main(int argc,const char **argv)
{
  if(argc!=4)return 2;
  @autoreleasepool {
    try {
      NSError *error=nil;
      NSString *base=[NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
      NSString *extra=[NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:&error];
      NSData *data=[NSData dataWithContentsOfFile:@(argv[3])];
      NSArray *fixtures=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
      if(!base||!extra||![fixtures isKindOfClass:[NSArray class]])return 2;
      DiffractionResidentMatrix engine([base stringByAppendingString:extra]);
      unsigned invalid_rejections=0;
      for(unsigned test=0;test<14;++test) {
        DiffractionResidentProfile p{16,3,1000,580,150,.41f,1,.9f,6,1,0,.9f,6,.42f,0};
        switch(test) {
          case 0:p.pitch=0;break;case 1:p.wavelength=0;break;case 2:p.depth=-1;break;
          case 3:p.duty=2;break;case 4:p.incident=0;break;case 5:p.ridge_k=-1;break;
          case 6:p.groove_n=0;break;case 7:p.substrate_n=-1;break;
          case 8:p.kx=std::numeric_limits<float>::quiet_NaN();break;
          case 9:p.depth=std::numeric_limits<float>::infinity();break;
          case 10:p.half_orders=0;break;case 11:p.half_orders=129;break;
          case 12:p.retained=17;break;case 13:p.retained=0;break;
        }
        bool rejected=false;
        try{unsigned steps=0;diffraction_resident_response(engine,p,steps);}
        catch(const std::runtime_error &){rejected=true;++invalid_rejections;}
        if(!rejected)throw std::runtime_error("Invalid profile accepted");
      }
      NSMutableArray *cases=[NSMutableArray array];unsigned failures=0,index=0;
      for(NSDictionary *fixture in fixtures) {
        NSArray *ridge=fixture[@"ridge"];
        DiffractionResidentProfile p{[fixture[@"half_orders"] unsignedIntValue],[fixture[@"retained_half_orders"] unsignedIntValue],
          [fixture[@"pitch_nm"] floatValue],[fixture[@"wavelength_nm"] floatValue],[fixture[@"depth_nm"] floatValue],
          [fixture[@"duty"] floatValue],[fixture[@"incident_ior"] floatValue],[ridge[0] floatValue],[ridge[1] floatValue],
          [fixture[@"incident_ior"] floatValue],0,[ridge[0] floatValue],[ridge[1] floatValue],
          [fixture[@"kx"] floatValue],[fixture[@"ky"] floatValue]};
        unsigned steps=0;NSArray *expected=fixture[@"matrix"];
        std::vector<Pair> actual(expected.count);
        NSMutableArray *times=[NSMutableArray array];double warmup=0;
        unsigned channels=0;double checked_max=0;
        for(unsigned run=0;run<4;++run) {
          engine.profiling=run>0;
          if(run==1){engine.metrics.clear();engine.allocation_count=0;engine.allocation_bytes=0;}
          auto start=std::chrono::steady_clock::now();
          @autoreleasepool {
            NSUInteger previous=engine.downloads;
            auto result=diffraction_resident_response(engine,p,steps);channels=result.rows;
            if(engine.downloads!=previous)throw std::runtime_error("Intermediate matrix readback");
            if(expected.count!=result.rows*result.cols)throw std::runtime_error("Reference dimensions disagree");
            engine.download(result,actual.data());
          }
          double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
          if(run==0)warmup=ms;else [times addObject:@(ms)];
          for(NSUInteger j=0;j<expected.count;++j) {
            NSArray *z=expected[j];double e=std::hypot(double(actual[j].x)-[z[0] doubleValue],double(actual[j].y)-[z[1] doubleValue]);
            if(!std::isfinite(e))throw std::runtime_error("Nonfinite benchmark response");
            checked_max=std::max(checked_max,e);
          }
        }
        double maximum=checked_max;
        for(NSUInteger i=0;i<expected.count;++i) {
          NSArray *z=expected[i];
          double e=std::hypot(double(actual[i].x)-[z[0] doubleValue],double(actual[i].y)-[z[1] doubleValue]);
          if(!std::isfinite(e))throw std::runtime_error("Nonfinite response");maximum=std::max(maximum,e);
        }
        if(maximum>=3e-4)++failures;
        fprintf(stderr,"Resident response %u N%u error %.9g\n",index,p.half_orders,maximum);
        NSMutableDictionary *metrics=[NSMutableDictionary dictionary];
        for(const auto &entry:engine.metrics) {
          metrics[@(entry.first.c_str())]=@{@"commands":@(entry.second.first),@"gpu_command_ms":@(entry.second.second)};
        }
        [cases addObject:@{@"case":@(index++),@"half_orders":@(p.half_orders),@"operation_metrics":metrics,@"allocated_bytes":@(engine.allocation_bytes),@"allocation_count":@(engine.allocation_count),@"channels":@(channels),@"warmup_ms":@(warmup),@"measured_ms":times,@"max_component_error":@(maximum),@"doublings":@(steps)}];
      }
      NSDictionary *report=@{@"scope":@"Instrumented standalone response profile; per-operation GPU command spans include scheduling effects; not a speedup benchmark",
        @"invalid_profile_rejections":@(invalid_rejections),@"device":engine.device.name,@"failures":@(failures),@"component_gate":@0.0003,@"matrix_input_uploads":@(engine.uploads),
        @"final_downloads":@(engine.downloads),@"scalar_norm_readbacks":@(engine.scalar_downloads),@"intermediate_payload_downloads":@0,@"cases":cases};
      NSData *json=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
      if(!json)return 2;fwrite(json.bytes,1,json.length,stdout);puts("");return failures?1:0;
    }
    catch(const std::exception &e){fprintf(stderr,"%s\n",e.what());return 2;}
  }
}
