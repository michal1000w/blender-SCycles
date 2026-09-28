/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "cycles_diffraction_resident_propagation.h"
#include <complex>
#include <cstdio>
struct Pair {float x,y;};
int main(int argc,const char **argv)
{
  if(argc!=4)return 2;
  @autoreleasepool {
    try {
      NSError *error=nil;
      NSString *base=[NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
      NSString *extra=[NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:&error];
      if(!base||!extra)return 2;
      DiffractionResidentMatrix engine([base stringByAppendingString:extra]);
      FILE *file=fopen(argv[3],"rb");if(!file)return 2;
      unsigned count=0;if(fread(&count,sizeof(unsigned),1,file)!=1||count>1024)return 2;
      NSMutableArray *cases=[NSMutableArray array];unsigned failures=0;
      for(unsigned i=0;i<count;++i) {
        unsigned m=0;float thickness=0;
        if(fread(&m,sizeof(unsigned),1,file)!=1||fread(&thickness,sizeof(float),1,file)!=1||!m||m>1024)return 2;
        std::vector<Pair> p(m*m),q(m*m),reference(4*m*m),actual(4*m*m);
        if(fread(p.data(),sizeof(Pair),p.size(),file)!=p.size()||fread(q.data(),sizeof(Pair),q.size(),file)!=q.size()||
           fread(reference.data(),sizeof(Pair),reference.size(),file)!=reference.size())return 2;
        auto gp=engine.upload(p.data(),m,m),gq=engine.upload(q.data(),m,m);
        NSUInteger before=engine.downloads,scalar_before=engine.scalar_downloads;
        unsigned steps=0;
        auto output=diffraction_resident_propagation(engine,gp,gq,thickness,steps);
        if(engine.downloads!=before)throw std::runtime_error("Intermediate propagation payload readback");
        engine.download(output,actual.data());
        double maximum=0;
        for(size_t j=0;j<actual.size();++j) {
          double e=std::hypot(double(actual[j].x)-reference[j].x,double(actual[j].y)-reference[j].y);
          if(!std::isfinite(e))throw std::runtime_error("Nonfinite propagation output");
          maximum=std::max(maximum,e);
        }
        if(maximum>=3e-4)++failures;
        fprintf(stderr,"Resident case %u dimension %u error %.9g\n",i,m,maximum);
        [cases addObject:@{@"case":@(i),@"dimension":@(m),@"doublings":@(steps),@"max_component_error":@(maximum),
                          @"scalar_norm_readbacks":@(engine.scalar_downloads-scalar_before)}];
      }
      if(fgetc(file)!=EOF)return 2;fclose(file);
      NSDictionary *report=@{@"scope":@"Resident propagation of preassembled transformed material matrices; no exterior/cache integration",
                             @"device":engine.device.name,@"failures":@(failures),@"component_gate":@0.0003,
                             @"uploads":@(engine.uploads),@"final_downloads":@(engine.downloads),@"intermediate_payload_downloads":@0,
                             @"scalar_norm_readbacks":@(engine.scalar_downloads),@"cases":cases};
      NSData *json=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
      if(!json)return 2;fwrite(json.bytes,1,json.length,stdout);puts("");return failures?1:0;
    }
    catch(const std::exception &e){fprintf(stderr,"%s\n",e.what());return 2;}
  }
}
