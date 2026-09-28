/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "cycles_diffraction_resident_response.h"
#include <cstdio>
struct Pair {float x,y;};
int main(int argc,const char **argv) {
  if(argc!=4)return 2;
  @autoreleasepool {
    try {
      NSError *error=nil;
      NSString *base=[NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
      NSString *extra=[NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:&error];
      if(!base||!extra)return 2;
      DiffractionResidentMatrix engine([base stringByAppendingString:extra]);
      DiffractionResidentProfile p{64,6,1160,600,750,.55f,1,.9f,6,1,0,.9f,6,.07f,.09f};
      unsigned steps=0;
      auto capture=[&](const char *name,DiffractionResidentMatrix::Matrix matrix) {
        std::vector<Pair> data(matrix.rows*matrix.cols);engine.download(matrix,data.data());
        std::string path=std::string(argv[3])+"/"+name+".bin";
        FILE *file=fopen(path.c_str(),"wb");if(!file)throw std::runtime_error("Cannot create stage capture");
        unsigned shape[2]={matrix.rows,matrix.cols};
        bool ok=fwrite(shape,sizeof(unsigned),2,file)==2&&fwrite(data.data(),sizeof(Pair),data.size(),file)==data.size();
        if(fclose(file)!=0||!ok)throw std::runtime_error("Stage capture write failed");
      };
      auto result=diffraction_resident_response(engine,p,steps,capture);capture("retained",result);
      printf("Captured %lu matrices; steps %u; diagnostic synchronizations enabled\n",(unsigned long)engine.downloads,steps);
      return 0;
    }catch(const std::exception &e){fprintf(stderr,"%s\n",e.what());return 2;}
  }
}
