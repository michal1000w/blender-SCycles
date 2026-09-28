/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include <cstring>
#include "cycles_diffraction_resident_matrix.h"
#include <complex>
#include <cmath>
#include <cstdio>
struct Pair {float x,y;};
static std::complex<double> value(Pair p) {return {p.x,p.y};}
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
      unsigned count=0;if(fread(&count,sizeof(unsigned),1,file)!=1 || count>10000)return 2;
      NSMutableArray *cases=[NSMutableArray array];unsigned failures=0;
      for(unsigned record=0;record<count;++record) {
        unsigned n=0,rhs=0;
        if(fread(&n,sizeof(unsigned),1,file)!=1||fread(&rhs,sizeof(unsigned),1,file)!=1||!n||n>1024||!rhs||rhs>2048)return 2;
        std::vector<Pair> joined(n*(n+rhs)),reference(n*rhs),a(n*n),b(n*rhs),x(n*rhs),ax(n*rhs);
        if(fread(joined.data(),sizeof(Pair),joined.size(),file)!=joined.size()||fread(reference.data(),sizeof(Pair),reference.size(),file)!=reference.size())return 2;
        for(unsigned row=0;row<n;++row) {
          memcpy(a.data()+row*n,joined.data()+row*(n+rhs),n*sizeof(Pair));
          memcpy(b.data()+row*rhs,joined.data()+row*(n+rhs)+n,rhs*sizeof(Pair));
        }
        NSUInteger before_downloads=engine.downloads;
        auto ga=engine.upload(a.data(),n,n),gb=engine.upload(b.data(),n,rhs);
        auto gx=engine.refined_solve(ga,gb,2);
        auto gax=engine.multiply(ga,gx);
        if(engine.downloads!=before_downloads)throw std::runtime_error("Unexpected intermediate readback");
        engine.download(gx,x.data());engine.download(gax,ax.data());
        double maximum=0,residual2=0,b2=0,product_error=0;
        for(unsigned row=0;row<n;++row)for(unsigned col=0;col<rhs;++col) {
          const unsigned i=row*rhs+col;
          double e=std::abs(value(x[i])-value(reference[i]));maximum=std::max(maximum,e);
          if(!std::isfinite(e)||e>2e-5)++failures;
          std::complex<double> expected(0,0);
          for(unsigned k=0;k<n;++k)expected+=value(a[row*n+k])*value(x[k*rhs+col]);
          residual2+=std::norm(expected-value(b[i]));b2+=std::norm(value(b[i]));
          double pe=std::abs(value(ax[i])-expected);product_error=std::max(product_error,pe);
          if(!std::isfinite(pe)||pe>2e-6+1e-6*std::abs(expected))++failures;
        }
        double residual=std::sqrt(residual2/b2);
        if(!std::isfinite(residual)||residual>2e-6)++failures;
        [cases addObject:@{@"record":@(record),@"rows":@(n),@"rhs":@(rhs),@"max_solution_error":@(maximum),
                          @"relative_residual":@(residual),@"max_product_error":@(product_error)}];
      }
      if(fgetc(file)!=EOF)return 2;fclose(file);
      NSDictionary *report=@{@"scope":@"Device-resident refined solves and products on captured propagation systems; not complete propagation",
                             @"device":engine.device.name,@"failures":@(failures),@"uploads":@(engine.uploads),
                             @"final_downloads":@(engine.downloads),@"intermediate_payload_downloads":@0,@"cases":cases};
      NSData *json=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
      if(!json)return 2;fwrite(json.bytes,1,json.length,stdout);puts("");return failures?1:0;
    }
    catch(const std::exception &e) {fprintf(stderr,"%s\n",e.what());return 2;}
  }
}
