/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "cycles_diffraction_resident_matrix.h"
#include <cstdio>
#include <limits>
struct Pair {float x,y;};
int main(int argc,const char **argv) {
  if(argc!=3)return 2;
  @autoreleasepool {
    NSError *error=nil;
    NSString *base=[NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
    NSString *extra=[NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:&error];
    if(!base||!extra)return 2;
    unsigned failures=0;
    for(unsigned test=0;test<5;++test) {
      DiffractionResidentMatrix engine([base stringByAppendingString:extra]);
      Pair a[4]={{1,0},{0,0},{0,0},{1,0}},b[2]={{1,0},{2,0}},out[2]={{123,456},{123,456}};
      if(test==0)a[3]={0,0};
      if(test==1)a[0].x=std::numeric_limits<float>::quiet_NaN();
      if(test==2)a[3].y=std::numeric_limits<float>::infinity();
      if(test==3)b[0].x=std::numeric_limits<float>::quiet_NaN();
      if(test==4)b[1].y=std::numeric_limits<float>::infinity();
      bool rejected=false;
      try {auto x=engine.refined_solve(engine.upload(a,2,2),engine.upload(b,2,1),2);engine.download(x,out);}
      catch(const std::runtime_error &){rejected=true;}
      bool untouched=out[0].x==123&&out[0].y==456&&out[1].x==123&&out[1].y==456;
      if(!rejected||!untouched||engine.downloads!=0)++failures;
      printf("case %u rejected %u untouched %u downloads %lu\n",test,rejected,untouched,(unsigned long)engine.downloads);
    }
    printf("failures %u\n",failures);return failures?1:0;
  }
}
