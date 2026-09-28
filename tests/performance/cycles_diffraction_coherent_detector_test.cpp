/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf.h"
#include "kernel/svm/closure.h"
#include "kernel/light/coherent_path_field.h"
#include "util/profiling.h"
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
using namespace ccl;
static int checks=0, failures=0;
static void check(bool value,const char *name) {checks++;if(!value){failures++;std::fprintf(stderr,"FAIL %s\n",name);}}
int main()
{
  KernelGlobalsCPU base{};base.data.integrator.coherent_specular_enabled=1;
  KernelObject object{};base.objects.data=&object;base.objects.width=1;
  Profiler profiler;ThreadKernelGlobalsCPU globals(base,nullptr,profiler,0);KernelGlobals kg=&globals;
  const auto scalar=[](float v){return SVMInputFloat{__float_as_uint(v)};};
  const auto triple=[&](float3 v){return SVMInputFloat3{scalar(v.x),scalar(v.y),scalar(v.z)};};
  for(const bool principled:{false,true}) for(const float3 color:{zero_float3(),make_float3(.2f,.6f,1),
      make_float3(-2,4,.5f),make_float3(std::numeric_limits<float>::quiet_NaN(),
                                     std::numeric_limits<float>::infinity(),.5f)}) {
    SVMNodeDiffuseBsdfData diffuse{};diffuse.color=triple(color);diffuse.roughness=scalar(0);
    diffuse.normal_offset=SVM_STACK_INVALID;
    SVMNodePrincipledBsdfData data{};data.base_color=triple(color);data.base_color.x.bits=SVM_INPUT_STACK_OFFSET_MASK | 4u;data.ior=scalar(1.5f);
    data.alpha=scalar(1);data.roughness=scalar(.5f);data.normal_offset=data.coat_normal_offset=SVM_STACK_INVALID;
    data.tangent_offset=SVM_STACK_INVALID;data.specular_tint=triple(one_float3());
    data.diffraction_albedo_handle=data.diffraction_two_sided_handle=-1;
    data.distribution=CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID;
    const size_t bytes=principled?sizeof(data):sizeof(diffuse);
    std::vector<uint> words(bytes/sizeof(uint));std::memcpy(words.data(),principled?(void*)&data:(void*)&diffuse,bytes);
    globals.svm_nodes.data=words.data();globals.svm_nodes.width=int(words.size());
    ShaderData sd{};sd.N=sd.Ng=sd.wi=make_float3(0,0,1);sd.num_closure_left=MAX_CLOSURE;
    sd.object_flag=SD_OBJECT_COHERENT_DETECTOR;
    float stack[SVM_STACK_SIZE]={};stack[4]=color.x;stack[5]=color.y;stack[6]=color.z;
    const SVMNodeClosureBsdf node{principled?CLOSURE_BSDF_PRINCIPLED_ID:CLOSURE_BSDF_DIFFUSE_ID,SVM_STACK_INVALID,{0,0,0}};
    svm_node_closure_bsdf<~uint64_t(0),SHADER_TYPE_SURFACE>(kg,&sd,stack,
        principled?one_spectrum():color,node,PATH_RAY_VISIBILITY_CAMERA,0,0);
    check(coherent_detector_eligible(&sd),"actual SVM Lambertian receives owned paths");
    const float3 expected=coherent_detector_passive_color(color);
    if(sd.num_closure!=(is_zero(expected)?0:1)) std::fprintf(stderr,"case principled=%d color=%g,%g,%g closures=%d expected=%g,%g,%g\n",principled,color.x,color.y,color.z,sd.num_closure,expected.x,expected.y,expected.z);
    check(sd.num_closure==(is_zero(expected)?0:1),"black detector owns zero energy without closure");
    if(sd.num_closure) {
      coherent_detector_prepare(&sd);
      check(isequal(sd.closure[0].weight,expected),"SVM linked detector color passive before allocation");
      check(isfinite_safe(sd.closure[0].sample_weight) && sd.closure[0].sample_weight==average(expected),
            "sample/guiding weight matches passive color");
    }
    CoherentGeometryPath path{};path.count=0;path.spreading=1;
    CoherentGeometryInterface patches[4]{};bool mirror[4]{};
    CoherentPathDetectorFrame frame{make_float3(1,0,0),make_float3(0,1,0),make_float3(0,0,1)};
    CoherentCompletedPathField white{},colored{};
    check(coherent_path_field_transport(make_float3(0,0,1),zero_float3(),frame,&path,patches,
        mirror,one_float3(),one_float3(),0,&white),"white field");
    check(coherent_path_field_transport(make_float3(0,0,1),zero_float3(),frame,&path,patches,
        mirror,one_float3(),expected,0,&colored),"colored field");
    check(reduce_max(fabs(colored.physical_diagonal_rgb-white.physical_diagonal_rgb*expected))<1e-7,
          "field diagonal equals white times RGB response");
    const float3 wc=coherent_path_field_pair_cross(&white,&white,0,550e-9f,1);
    const float3 cc=coherent_path_field_pair_cross(&colored,&colored,0,550e-9f,1);
    check(reduce_max(fabs(cc-wc*expected))<1e-7,"all pairs equal white times RGB response");
  }
  ShaderData invalid{};invalid.object_flag=SD_OBJECT_COHERENT_DETECTOR;
  invalid.num_closure=1;invalid.closure[0].type=CLOSURE_NONE_ID;
  check(!coherent_detector_eligible(&invalid),"filtered detector does not suppress ordinary estimators");
  invalid.closure[0].type=CLOSURE_BSDF_MICROFACET_GGX_ID;
  check(!coherent_detector_eligible(&invalid),"glossy detector cannot own Lambertian estimator");
  invalid.num_closure=2;invalid.closure[0].type=CLOSURE_BSDF_DIFFUSE_ID;
  check(!coherent_detector_eligible(&invalid),"layered receiver cannot own Lambertian estimator");
  std::printf("checks=%d failures=%d\n",checks,failures);return failures!=0;
}
