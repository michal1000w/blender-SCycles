/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
/* Raw-response reproduction of the full-domain application rejection. */
#include "device/metal/diffraction/response.h"
#include "diffraction_shader_source.h"
#include "scene/diffraction.h"
#include <Eigen/Dense>
#include <iomanip>
#include <iostream>

int main()
{
  @autoreleasepool {
    DiffractionResidentMatrix engine(
        [NSString stringWithUTF8String:diffraction_metal_shader_source], MTLCreateSystemDefaultDevice());
    const double wavelength=446.66666666666669, kx=-0.30180180180180183,
                 ky=-0.83333333333333337;
    ccl::DiffractionGratingProfile profile{740,150,double(float(.41)),1,1.5,1,1};
    std::cout << std::setprecision(17)
              << "orders,raw_unitarity,cpu_unitarity,raw_cpu_error,rounded_input_error,raw_rounded_cpu_error,polar_cpu_error,polar_change,polar_unitarity\n";
    for (unsigned orders : {16,32,64}) {
      DiffractionResidentProfile p{orders,2,740,float(wavelength),150,float(.41),
                                  1,1.5,0,1,0,1,0,float(kx),float(ky)};
      unsigned steps=0;
      auto raw=diffraction_resident_response(engine,p,steps);
      struct Pair {float x,y;};
      std::vector<Pair> values(raw.rows*raw.cols);
      engine.download(raw,values.data());
      ccl::DiffractionGratingBlock cpu,rounded;std::string error;
      if (!ccl::diffraction_grating_solve_reference(profile,wavelength,kx,ky,orders,2,cpu,error) ||
          !ccl::diffraction_grating_solve_reference(profile,p.wavelength,p.kx,p.ky,orders,2,rounded,error)) {
        std::cerr<<error;return 1;
      }
      const int n=raw.rows;
      if(raw.cols!=raw.rows||cpu.matrix.size()!=values.size()||rounded.matrix.size()!=values.size())return 2;
      Eigen::MatrixXcd g(n,n),c(n,n),r(n,n);
      for(int i=0;i<n;++i)for(int j=0;j<n;++j) {
        const int k=i*n+j;g(i,j)={values[k].x,values[k].y};c(i,j)=cpu.matrix[k];r(i,j)=rounded.matrix[k];
      }
      const auto identity=Eigen::MatrixXcd::Identity(n,n).eval();
      /* Diagnostic only: report an unconstrained polar result without using
       * it to accept the response or changing the production bound. */
      Eigen::MatrixXcd polar=g;
      for(int i=0;i<2;++i)polar=(.5*polar*(3.*identity-polar.adjoint()*polar)).eval();
      std::cout<<orders<<','<<(g.adjoint()*g-identity).norm()<<','
               <<(c.adjoint()*c-identity).norm()<<','<<(g-c).cwiseAbs().maxCoeff()<<','
               <<(r-c).cwiseAbs().maxCoeff()<<','<<(g-r).cwiseAbs().maxCoeff()<<','
               <<(polar-c).cwiseAbs().maxCoeff()<<','<<(polar-g).cwiseAbs().maxCoeff()<<','
               <<(polar.adjoint()*polar-identity).norm()<<'\n';
    }
  }
}
