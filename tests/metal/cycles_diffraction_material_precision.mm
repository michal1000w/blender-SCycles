/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "cycles_diffraction_resident_response.h"
#include <complex>
#include <cstdio>
#include <Eigen/Dense>
struct Pair {float x,y;};
int main(int argc,const char **argv) {
  if(argc!=3)return 2;
  @autoreleasepool {
    try {
      NSError *error=nil;
      NSString *base=[NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
      NSString *extra=[NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:&error];
      if(!base||!extra)return 2;
      DiffractionResidentMatrix engine([base stringByAppendingString:extra]);
      DiffractionResidentProfile p{64,6,1160,600,750,.55f,1,.9f,6,1,0,.9f,6,.07f,.09f};
      const unsigned n=2*p.half_orders+1;
      NSMutableArray *reports=[NSMutableArray array];
      for(unsigned kind=0;kind<2;++kind) {
        auto matrix=engine.material(n,n,kind,&p,sizeof(p));
        std::vector<Pair> values(n*n);engine.download(matrix,values.data());
        std::complex<double> er=std::pow(std::complex<double>(p.ridge_n,p.ridge_k),2);
        std::complex<double> eg=std::pow(std::complex<double>(p.groove_n,p.groove_k),2);
        if(kind==1){er=1.0/er;eg=1.0/eg;}
        Eigen::MatrixXcd exact_material(n,n);
        double maximum=0,error2=0,reference2=0;
        for(unsigned row=0;row<n;++row)for(unsigned col=0;col<n;++col) {
          int order=int(row)-int(col);
          double coefficient=order?std::sin(3.14159265358979323846*order*double(p.duty))/(3.14159265358979323846*order):double(p.duty);
          auto expected=(er-eg)*coefficient+(row==col?eg:std::complex<double>(0));
          exact_material(row,col)=expected;
          auto actual=std::complex<double>(values[row*n+col].x,values[row*n+col].y);
          double e=std::abs(actual-expected);if(!std::isfinite(e))return 1;
          maximum=std::max(maximum,e);error2+=e*e;reference2+=std::norm(expected);
        }
        auto inverse=engine.refined_solve(matrix,engine.identity(n),2);
        std::vector<Pair> inverse_values(n*n);engine.download(inverse,inverse_values.data());
        Eigen::MatrixXcd rounded(n,n),actual_inverse(n,n);
        for(unsigned row=0;row<n;++row)for(unsigned col=0;col<n;++col) {
          Pair a=values[row*n+col],b=inverse_values[row*n+col];
          rounded(row,col)={a.x,a.y};actual_inverse(row,col)={b.x,b.y};
        }
        Eigen::MatrixXcd oracle=rounded.partialPivLu().solve(Eigen::MatrixXcd::Identity(n,n));
        Eigen::MatrixXcd exact_inverse=exact_material.partialPivLu().solve(Eigen::MatrixXcd::Identity(n,n));
        double coefficient_effect=(oracle-exact_inverse).norm()/exact_inverse.norm();
        double total_inverse_error=(actual_inverse-exact_inverse).norm()/exact_inverse.norm();
        double inverse_forward=(actual_inverse-oracle).norm()/oracle.norm();
        double condition1=rounded.cwiseAbs().colwise().sum().maxCoeff()*oracle.cwiseAbs().colwise().sum().maxCoeff();
        double residual_max=0,residual2=0;
        for(unsigned row=0;row<n;++row)for(unsigned col=0;col<n;++col) {
          std::complex<double> sum(0);
          for(unsigned k=0;k<n;++k) {
            Pair a=values[row*n+k],b=inverse_values[k*n+col];
            sum+=std::complex<double>(a.x,a.y)*std::complex<double>(b.x,b.y);
          }
          double error=std::abs(sum-double(row==col));
          if(!std::isfinite(error))return 1;
          residual_max=std::max(residual_max,error);residual2+=error*error;
        }
        [reports addObject:@{@"kind":@(kind),@"max_component_error":@(maximum),
          @"relative_frobenius_error":@(std::sqrt(error2/reference2)),
          @"inverse_residual_max":@(residual_max),
          @"coefficient_induced_inverse_error":@(coefficient_effect),
          @"inverse_error_to_exact_material":@(total_inverse_error),
          @"inverse_relative_forward_error":@(inverse_forward),@"condition_1":@(condition1),
          @"inverse_relative_frobenius_residual":@(std::sqrt(residual2/n))}];
      }
      NSData *json=[NSJSONSerialization dataWithJSONObject:reports options:NSJSONWritingPrettyPrinted error:&error];
      fwrite(json.bytes,1,json.length,stdout);puts("");return 0;
    }catch(const std::exception &e){fprintf(stderr,"%s\n",e.what());return 2;}
  }
}
