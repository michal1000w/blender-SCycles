/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf.h"
#include "scene/diffraction_albedo.h"
#include "util/profiling.h"
#include <cmath>
#include <cstdio>
#include <string>

using namespace ccl;
static int checks = 0, failures = 0;
static void check(bool condition, const char *name)
{
  checks++;
  if (!condition) { failures++; std::fprintf(stderr,"FAIL %s\n",name); }
}
static float fraction(float x) { return x-floorf(x); }

int main()
{
  DiffractionTwoSidedAlbedoRequest request;
  request.alpha_x = request.alpha_y = 0.3f;
  request.inside_ior = 1.5f; request.inv_abbe = 1.0f/40.0f;
  request.generalized_f0_count = 16;
  request.film_ior = 1.32f; request.film_thickness_nm = 250.0f;
  request.mu_count = 4; request.phi_count = 8; request.wavelength_count = 16;
  request.facet_samples = 512;
  DiffractionTwoSidedAlbedoTable table;
  std::string error;
  check(diffraction_two_sided_albedo_build_cpu(request,table,error),"production coated nonlinear basis cache");
  if (table.deficits.empty()) { std::fprintf(stderr,"%s\n",error.c_str()); return 2; }
  const int4 descriptor=make_int4(0,0,0,4 | (15<<8));
  const float4 domain=make_float4(380,780,16,8);
  KernelGlobalsCPU base{};
  base.diffraction_two_sided_values.data=table.deficits.data();
  base.diffraction_two_sided_values.width=int(table.deficits.size());
  base.diffraction_two_sided_integrals.data=table.integrals.data();
  base.diffraction_two_sided_integrals.width=int(table.integrals.size());
  base.diffraction_two_sided_cross.data=table.cross_fractions.data();
  base.diffraction_two_sided_cross.width=int(table.cross_fractions.size());
  base.diffraction_two_sided_descriptors.data=&descriptor;
  base.diffraction_two_sided_domains.data=&domain;
  base.diffraction_two_sided_descriptors.width=base.diffraction_two_sided_domains.width=1;
  base.data.tables.num_diffraction_two_sided_caches=1;
  KernelObject object{}; base.objects.data=&object; base.objects.width=1;
  Profiler profiler; ThreadKernelGlobalsCPU globals(base,nullptr,profiler,0); KernelGlobals kg=&globals;
  for (const float nd : {1.0f, 1.0001f, 1.5f, 3.0f, 4.0f, 10.0f}) {
    float previous = -1;
    for (int node=0; node<16; ++node) {
      const float value=dielectric_f0_cache_node(nd,16,node);
      check(value>previous && value>=0 && value<=1,"coated knots strictly ordered at high/near-unit IOR");
      check(fabsf(dielectric_f0_cache_coordinate(nd,16,value)-node)<2e-5f,"coated knot lookup inverse");
      previous=value;
    }
  }
  float max_pdf=0,max_eval=0,max_reciprocity=0,max_row=0,max_energy=0;
  for(const float wavelength_sample : {0.2f,0.67f}) {
    const float wavelength_um=sample_wavelength(wavelength_sample);
    const float n=dielectric_ior_at_wavelength(request.inside_ior,request.inv_abbe,wavelength_um);
    const double ld=.5876,lc=.6563,lf=.4861;
    const double B=(double(request.inside_ior)-1)*request.inv_abbe/(1/(lf*lf)-1/(lc*lc));
    const double nd=double(request.inside_ior)-B/(ld*ld)+B/double(wavelength_um*wavelength_um);
    check(fabs(double(n)-nd)<5e-7,"shared Cauchy agrees independent double");
    for(const Spectrum tint : {one_spectrum(),make_float3(0.2f,0.7f,1.3f)}) {
      ShaderData front{},back{};
      const auto setup=[&](ShaderData &sd,bool reversed,float3 wi,int capacity) {
        sd.N=sd.Ng=make_float3(0,0,reversed?-1.0f:1.0f);sd.wi=wi;
        sd.runtime_flag=reversed?SR_BACKFACING:0;sd.shader_flag=SD_REQUIRES_WAVELENGTH;
        sd.rand_wavelength=wavelength_sample;sd.num_closure_left=capacity;
        return bsdf_diffraction_principled_transmission_two_sided_setup(
            kg,&sd,one_spectrum(),one_spectrum(),sd.N,make_float3(1,0,0),sqrtf(.3f),
            request.inside_ior,request.pitch_nm,request.depth_nm,request.duty,tint,
            request.inv_abbe,request.film_ior,request.film_thickness_nm,0);
      };
      check(setup(front,false,normalize(make_float3(.35f,.12f,1)),MAX_CLOSURE),"front generalized setup");
      const float3 transmitted=normalize(make_float3(.25f,-.15f,-1));
      check(setup(back,true,transmitted,MAX_CLOSURE),"back generalized setup");
      check(front.num_closure==2 && back.num_closure==2,"generalized first event and separate return");
      const auto *extra=(const DiffractionDielectricGeneralizedExtra *)
          ((const DiffractionDielectricBsdf *)&front.closure[0])->extra;
      check(fabsf(extra->base.param.facet.transmitted_ior-n)<1e-7f,
            "first event uses native spectral IOR");
      check(reduce_max(fabs(extra->generalized_f0-F0_from_ior(request.inside_ior)*tint))<1e-7f,
            "first event keeps native d-line F0 rather than replacing with spectral Fresnel");
      const int slots=MAX_CLOSURE-front.num_closure_left;
      ShaderData scarce{};
      check(!setup(scarce,false,front.wi,slots-1) && scarce.num_closure==0 &&
            scarce.num_closure_left==slots-1,"near-capacity setup is transactional");
      for(ShaderData *sd : {&front,&back}) {
        Spectrum total=zero_spectrum();
        for(int c=0;c<sd->num_closure;c++) {
          const ShaderClosure *sc=&sd->closure[c];
          Spectrum energy=zero_spectrum();
          for(int i=0;i<4096;i++) {
            Spectrum sampled;float3 wo;float pdf,eta;float2 rough;
            const int label=bsdf_sample(kg,sd,sc,make_float3((i+.5f)/4096,
                fraction((i+.5f)*.61803398875f),fraction((i+.5f)*.75487766625f)),
                &sampled,&wo,&pdf,&rough,&eta);
            if(label==LABEL_NONE || !(pdf>0)) continue;
            float epdf;const Spectrum evaluated=bsdf_eval(kg,sd,sc,wo,&epdf);
            max_pdf=max(max_pdf,fabsf(pdf-epdf));max_eval=max(max_eval,reduce_max(fabs(sampled-evaluated)));
            check(fabsf(pdf-epdf)<5e-5f*max(1.0f,pdf),"generalized sample/eval pdf");
            check(reduce_max(fabs(sampled-evaluated))<5e-5f*max(1.0f,reduce_max(sampled)),"generalized sample/eval value");
            if(label & LABEL_TRANSMIT) {
              check(fabsf(eta-(sd==&front?n:1/n))<3e-7f,"sample eta uses runtime spectral IOR");
            }
            energy+=sampled/pdf;
          }
          total+=energy/4096;
        }
        max_energy=max(max_energy,reduce_max(fabs(total-one_spectrum())));
        check(reduce_min(total)>0.95f && reduce_max(total)<1.05f,"colored lossless basis furnace");
        /* Independent quadrature of only the interpolated return. Its row
         * must equal the lookup deficit even BETWEEN wavelength nodes. */
        const ShaderClosure *returned=&sd->closure[1];
        const Spectrum expected=bsdf_albedo(kg,sd,returned,true,true);
        Spectrum integral=zero_spectrum();
        for(int side : {-1,1}) for(int i=0;i<128;i++) for(int j=0;j<64;j++) {
          const float mu=(i+.5f)/128,r=sqrtf(1-mu*mu),phi=M_2PI_F*(j+.5f)/64;
          const float3 wo=make_float3(r*cosf(phi),r*sinf(phi),side*mu);
          float pdf;integral+=bsdf_eval(kg,sd,returned,wo,&pdf)*(M_2PI_F/(128*64));
        }
        const float row=reduce_max(fabs(integral-expected));max_row=max(max_row,row);
        check(row<.002f,"off-grid wavelength return row matches angular deficit integral");
      }
      for(int c=0;c<2;c++) {
        float fp,rp;const Spectrum f=bsdf_eval(kg,&front,&front.closure[c],transmitted,&fp);
        const Spectrum r=bsdf_eval(kg,&back,&back.closure[c],front.wi,&rp);
        const Spectrum e=f/(n*n*fabsf(transmitted.z))-r/fabsf(front.wi.z);
        max_reciprocity=max(max_reciprocity,reduce_max(fabs(e)));
        check(reduce_max(fabs(e))<5e-4f*max(1.0f,reduce_max(f)),"paired reverse etendue reciprocity");
      }
    }
  }
  /* Allocation and first-event identity at the film's near-index limit.
   * Cache energy here is deliberately not tested with the different-IOR table. */
  for (const float nd : {1.0f, 1.001f}) {
    ShaderData reference{}, integrated{};
    const auto setup_data=[](ShaderData &sd) {
      sd.N=sd.Ng=make_float3(0,0,1); sd.wi=normalize(make_float3(.3f,.1f,1));
      sd.shader_flag=SD_REQUIRES_WAVELENGTH; sd.rand_wavelength=.2f;
      sd.num_closure_left=MAX_CLOSURE;
    };
    setup_data(reference); setup_data(integrated);
    const Spectrum tint=make_float3(.2f,.7f,1.3f);
    check(bsdf_diffraction_principled_transmission_setup(&reference,one_spectrum(),
        one_spectrum(),reference.N,make_float3(1,0,0),sqrtf(.3f),nd,
        request.pitch_nm,request.depth_nm,request.duty,tint,0,1.32f,250),
        "near-index first-event physical coating setup");
    check(bsdf_diffraction_principled_transmission_two_sided_setup(kg,&integrated,
        one_spectrum(),one_spectrum(),integrated.N,make_float3(1,0,0),sqrtf(.3f),nd,
        request.pitch_nm,request.depth_nm,request.duty,tint,0,1.32f,250,0),
        "near-index completed coated setup");
    check(integrated.num_closure==reference.num_closure+1,
        "near-index physical film closures plus separate return");
    for(int c=0;c<reference.num_closure;c++) {
      check(reference.closure[c].type==integrated.closure[c].type,
          "near-index coating/atom types preserved");
      for(int i=0;i<64;i++) {
        const float3 random=make_float3((i+.5f)/64,fraction(i*.618f),fraction(i*.754f));
        Spectrum a,b;float3 ao,bo;float ap,bp,ae,be;float2 ar,br;
        const int al=bsdf_sample(kg,&reference,&reference.closure[c],random,&a,&ao,&ap,&ar,&ae);
        const int bl=bsdf_sample(kg,&integrated,&integrated.closure[c],random,&b,&bo,&bp,&br,&be);
        check(al==bl && fabsf(ap-bp)<1e-6f && reduce_max(fabs(a-b))<1e-6f &&
              len(ao-bo)<1e-6f,"near-index film first-event atom sampling unchanged");
      }
    }
    const int slots=MAX_CLOSURE-integrated.num_closure_left;
    ShaderData scarce{}; setup_data(scarce); scarce.num_closure_left=slots-1;
    check(!bsdf_diffraction_principled_transmission_two_sided_setup(kg,&scarce,
        one_spectrum(),one_spectrum(),scarce.N,make_float3(1,0,0),sqrtf(.3f),nd,
        request.pitch_nm,request.depth_nm,request.duty,tint,0,1.32f,250,0) &&
        scarce.num_closure==0 && scarce.num_closure_left==slots-1,
        "near-index coating/atom/return allocation transactional");
  }
  /* Subcutoff film is a native identity, independent of this table's
   * energy values. Verify routing and first-event data with a bare descriptor. */
  const int4 bare_descriptor=make_int4(0,0,0,4 | (1<<8));
  globals.diffraction_two_sided_descriptors.data=&bare_descriptor;
  ShaderData bare{},subcutoff{};
  const auto setup_cutoff=[&](ShaderData &sd,float thickness) {
    sd.N=sd.Ng=make_float3(0,0,1);sd.wi=normalize(make_float3(.3f,.1f,1));
    sd.shader_flag=SD_REQUIRES_WAVELENGTH;sd.rand_wavelength=.2f;
    sd.num_closure_left=MAX_CLOSURE;
    return bsdf_diffraction_principled_transmission_two_sided_setup(kg,&sd,
        one_spectrum(),one_spectrum(),sd.N,make_float3(1,0,0),sqrtf(.3f),1.5f,
        request.pitch_nm,request.depth_nm,request.duty,make_float3(.2f,.7f,1.3f),
        request.inv_abbe,1.32f,thickness,0);
  };
  check(setup_cutoff(bare,0) && setup_cutoff(subcutoff,.05f),
        "subcutoff film accepts bare generalized descriptor");
  check(bare.num_closure==subcutoff.num_closure &&
        bare.num_closure_left==subcutoff.num_closure_left,"subcutoff allocation equals bare");
  const auto *cutoff_extra=(const DiffractionDielectricGeneralizedExtra *)
      ((const DiffractionDielectricBsdf *)&subcutoff.closure[0])->extra;
  check(cutoff_extra->film_thickness_over_wavelength==0,"subcutoff first event has no film");
  for(int c=0;c<bare.num_closure;c++) for(int i=0;i<64;i++) {
    const float3 random=make_float3((i+.5f)/64,fraction(i*.618f),fraction(i*.754f));
    Spectrum a,b;float3 ao,bo;float ap,bp,ae,be;float2 ar,br;
    const int al=bsdf_sample(kg,&bare,&bare.closure[c],random,&a,&ao,&ap,&ar,&ae);
    const int bl=bsdf_sample(kg,&subcutoff,&subcutoff.closure[c],random,&b,&bo,&bp,&br,&be);
    check(al==bl && ap==bp && isequal(a,b) && isequal(ao,bo),
          "subcutoff film sample equals bare exactly");
  }
  std::printf("checks=%d failures=%d max_pdf=%g max_eval=%g max_reciprocity=%g max_row=%g max_energy=%g\n",
              checks,failures,max_pdf,max_eval,max_reciprocity,max_row,max_energy);
  return failures!=0;
}
