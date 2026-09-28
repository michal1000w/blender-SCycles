/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf.h"
#include "kernel/svm/closure.h"
#include "scene/diffraction_albedo.h"
#include "scene/shader.tables"
#include "util/profiling.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace ccl;
static int checks = 0, failures = 0;
static void check(const bool valid, const char *name)
{
  checks++;
  if (!valid) { failures++; std::fprintf(stderr, "FAIL %s\n", name); }
}
static float fraction(const float x) { return x - floorf(x); }

static MicrofacetBsdf *reflective_layer(KernelGlobals kg, ShaderData *sd,
                                       const Spectrum weight, const Spectrum tint)
{
  auto *b = (MicrofacetBsdf *)bsdf_alloc(sd, sizeof(MicrofacetBsdf), weight);
  auto *f = b ? (FresnelGeneralizedSchlick *)closure_alloc_extra(
                   sd, sizeof(FresnelGeneralizedSchlick)) : nullptr;
  if (!b || !f) return nullptr;
  b->N = sd->N; b->T = make_float3(1,0,0); b->alpha_x = b->alpha_y = 0.3f;
  b->ior = 1.5f;
  f->f0 = 0.04f * tint; f->f90 = one_spectrum(); f->exponent = -1.5f;
  f->tint = {one_spectrum(), zero_spectrum()}; f->thin_film = {0, 0};
  bsdf_microfacet_ggx_setup(b);
  bsdf_microfacet_setup_fresnel_generalized_schlick(kg, b, sd->wi, f, false);
  return b;
}

static double integrate(KernelGlobals kg, ShaderData &sd, const ShaderClosure *sc)
{
  constexpr int mu_count = 64, phi_count = 128;
  double value = 0;
  for (int i = 0; i < mu_count; i++) {
    const float z = (i + 0.5f) / mu_count, radius = sqrtf(1-z*z);
    for (int j = 0; j < phi_count; j++) {
      const float phi = M_2PI_F * (j + 0.5f) / phi_count;
      float pdf;
      const Spectrum f = bsdf_eval(kg, &sd, sc,
          make_float3(radius*cosf(phi), radius*sinf(phi), z), &pdf);
      value += double(average(sc->weight * f));
    }
  }
  return value * double(M_2PI_F) / (mu_count * phi_count);
}

int main()
{
  DiffractionAlbedoRequest request;
  request.alpha_x = request.alpha_y = 0.3f;
  request.mu_count = 4; request.phi_count = 8; request.wavelength_count = 4;
  request.facet_samples = 512;
  DiffractionAlbedoTable table;
  std::string error;
  check(diffraction_albedo_build_cpu(request, table, error), "one-sided production cache");
  if (table.values.empty()) return 2;
  const int4 descriptor = make_int4(0,0,request.mu_count,request.phi_count);
  const float4 domain = make_float4(380,780,request.wavelength_count,1);
  KernelGlobalsCPU base{};
  std::vector<float> native_tables;
  const auto add_native_table = [&](const float *values, const int count) {
    const int offset = int(native_tables.size());
    native_tables.insert(native_tables.end(), values, values + count);
    return offset;
  };
  base.data.tables.ggx_E = add_native_table(table_ggx_E, 1024);
  base.data.tables.ggx_Eavg = add_native_table(table_ggx_Eavg, 32);
  base.data.tables.ggx_gen_schlick_ior_s = add_native_table(table_ggx_gen_schlick_ior_s, 32768);
  base.data.tables.ggx_gen_schlick_s = add_native_table(table_ggx_gen_schlick_s, 32768);
  base.data.tables.ggx_glass_E = add_native_table(table_ggx_glass_E, 32768);
  base.data.tables.ggx_glass_Eavg = add_native_table(table_ggx_glass_Eavg, 1024);
  base.data.tables.ggx_glass_inv_E = add_native_table(table_ggx_glass_inv_E, 32768);
  base.data.tables.ggx_glass_inv_Eavg = add_native_table(table_ggx_glass_inv_Eavg, 1024);
  base.data.tables.sheen_ltc = add_native_table(table_sheen_ltc, 3072);
  base.data.tables.thin_film_table = add_native_table(&table_thin_film_cmf[0][0], 3072);
  base.lookup_table.data = native_tables.data();
  base.lookup_table.width = int(native_tables.size());
  base.diffraction_albedo_values.data = table.values.data();
  base.diffraction_albedo_values.width = int(table.values.size());
  base.diffraction_albedo_averages.data = table.averages.data();
  base.diffraction_albedo_averages.width = int(table.averages.size());
  base.diffraction_albedo_descriptors.data = &descriptor;
  base.diffraction_albedo_domains.data = &domain;
  base.diffraction_albedo_descriptors.width = base.diffraction_albedo_domains.width = 1;
  base.data.tables.num_diffraction_albedo_caches = 1;
  KernelObject object{}; base.objects.data = &object; base.objects.width = 1;
  Profiler profiler; ThreadKernelGlobalsCPU globals(base,nullptr,profiler,0);
  KernelGlobals kg = &globals;
  float largest_pdf_error = 0, largest_eval_error = 0, largest_reciprocity = 0;
  double largest_layer_error = 0;
  for (const float coverage : {0.0f, 0.7f, 1.0f}) {
    ShaderData sd{};
    sd.N = sd.Ng = make_float3(0,0,1);
    sd.wi = normalize(make_float3(0.35f,0.12f,1));
    sd.rand_wavelength = 0.5f; sd.num_closure_left = MAX_CLOSURE;
    MicrofacetBsdf *b = reflective_layer(kg, &sd, one_spectrum(),
                                        rgb_to_spectrum(make_float3(0.5f,0.8f,1)));
    check(b != nullptr, "generalized reflective layer allocation");
    if (!b) return 3;
    const Spectrum fss = diffraction_conductor_average_fresnel(kg,b,false);
    double independent_fss = 0;
    for (int i = 0; i < 2048; i++) {
      const float mu = (i + 0.5f) / 2048;
      independent_fss += double(2*mu*average(microfacet_fresnel(kg,b,mu,nullptr).reflectance));
    }
    independent_fss /= 2048;
    check(fabs(average(fss)-independent_fss) < 0.015,
          "generalized Fresnel average matches independent projected integral");
    check(average(fss) > 0 && average(fss) < 0.2f,
          "dielectric return remains absorptive below unit Fresnel");
    check(bsdf_diffraction_conductor_multiggx_split_setup(
        kg,&sd,b,make_float3(1,0,0),request.pitch_nm,request.depth_nm,request.duty,
        coverage,0), "generalized reflective MultiGGX split");
    double energy = 0; Spectrum estimated = zero_spectrum();
    for (int closure = 0; closure < sd.num_closure; closure++) {
      const ShaderClosure *sc = &sd.closure[closure];
      energy += integrate(kg,sd,sc);
      estimated += closure_albedo(kg,&sd,sc,true,false);
      for (int i=0;i<2048;i++) {
        const float3 random = make_float3((i+0.5f)/2048,
            fraction((i+0.5f)*0.61803398875f),fraction((i+0.5f)*0.75487766625f));
        Spectrum sampled; float3 wo; float pdf,eta; float2 roughness;
        const int label = bsdf_sample(kg,&sd,sc,random,&sampled,&wo,&pdf,&roughness,&eta);
        if (label==LABEL_NONE) continue;
        float queried_pdf; const Spectrum queried = bsdf_eval(kg,&sd,sc,wo,&queried_pdf);
        largest_pdf_error=max(largest_pdf_error,fabsf(pdf-queried_pdf));
        largest_eval_error=max(largest_eval_error,reduce_max(fabs(sampled-queried)));
        check(fabsf(pdf-queried_pdf)<5e-5f*max(1.0f,pdf),"reflection sample/eval PDF");
        check(reduce_max(fabs(sampled-queried))<5e-5f*max(1.0f,reduce_max(sampled)),
              "reflection sample/eval value");
        if (bsdf_is_diffraction_conductor(sc->type)) {
          float rp; const Spectrum reverse = bsdf_diffraction_conductor_eval(kg,sc,wo,sd.wi,&rp);
          const float e=reduce_max(fabs(sampled/wo.z-reverse/sd.wi.z));
          largest_reciprocity=max(largest_reciprocity,e);
          check(e<5e-4f*max(1.0f,reduce_max(sampled/wo.z)),"reflective lobe reciprocity");
        }
      }
    }
    const double layer_error=fabs(energy-double(average(estimated)));
    largest_layer_error=std::max(largest_layer_error,layer_error);
    check(energy>=0 && energy<=1.01,"reflective dielectric energy bound");
    check(layer_error<0.06,"layer albedo tracks complete reflected energy");
  }
  DiffractionTwoSidedAlbedoRequest transmitted_request;
  transmitted_request.alpha_x = transmitted_request.alpha_y = 0.3f;
  transmitted_request.mu_count = 4; transmitted_request.phi_count = 8;
  transmitted_request.wavelength_count = 4; transmitted_request.facet_samples = 512;
  DiffractionTwoSidedAlbedoTable transmitted_table;
  check(diffraction_two_sided_albedo_build_cpu(transmitted_request, transmitted_table, error),
        "physical transmission cache for mixed Principled");
  if (transmitted_table.deficits.empty()) return 4;
  const int4 transmitted_descriptor = make_int4(0,0,0,4);
  const float4 transmitted_domain = make_float4(380,780,4,8);
  globals.diffraction_two_sided_values.data = transmitted_table.deficits.data();
  globals.diffraction_two_sided_values.width = int(transmitted_table.deficits.size());
  globals.diffraction_two_sided_integrals.data = transmitted_table.integrals.data();
  globals.diffraction_two_sided_integrals.width = int(transmitted_table.integrals.size());
  globals.diffraction_two_sided_cross.data = transmitted_table.cross_fractions.data();
  globals.diffraction_two_sided_cross.width = int(transmitted_table.cross_fractions.size());
  globals.diffraction_two_sided_descriptors.data = &transmitted_descriptor;
  globals.diffraction_two_sided_domains.data = &transmitted_domain;
  globals.diffraction_two_sided_descriptors.width = globals.diffraction_two_sided_domains.width = 1;
  globals.data.tables.num_diffraction_two_sided_caches = 1;
  ShaderData mixed{};
  mixed.N = mixed.Ng = make_float3(0,0,1);
  mixed.wi = normalize(make_float3(0.35f,0.12f,1));
  mixed.rand_wavelength = 0.5f; mixed.num_closure_left = MAX_CLOSURE;
  /* Actual Principled weights: metallic .3, solid transmission .4 of the
   * remaining .7, followed by reflection over the remaining .42 diffuse. */
  auto *metal = (MicrofacetBsdf *)bsdf_alloc(&mixed,sizeof(MicrofacetBsdf),make_spectrum(0.3f));
  auto *metal_f = (FresnelF82Tint *)closure_alloc_extra(&mixed,sizeof(FresnelF82Tint));
  metal->N = mixed.N; metal->T = make_float3(1,0,0);
  metal->alpha_x = metal->alpha_y = 0.3f; metal->ior = 1;
  metal_f->f0 = one_spectrum(); metal_f->thin_film = {0,0};
  bsdf_microfacet_ggx_setup(metal);
  bsdf_microfacet_setup_fresnel_f82_tint(kg,metal,mixed.wi,metal_f,one_spectrum(),false);
  check(bsdf_diffraction_conductor_multiggx_split_setup(kg,&mixed,metal,make_float3(1,0,0),
      request.pitch_nm,request.depth_nm,request.duty,1,0), "mixed metallic completion");
  check(bsdf_diffraction_principled_transmission_two_sided_setup(kg,&mixed,
      make_spectrum(0.28f),make_spectrum(0.28f),mixed.N,make_float3(1,0,0),sqrtf(0.3f),
      1.5f,transmitted_request.pitch_nm,transmitted_request.depth_nm,
      transmitted_request.duty,one_spectrum(),0,1,0,0), "mixed transmission completion");
  const int first_reflection = mixed.num_closure;
  auto *reflected = reflective_layer(kg,&mixed,make_spectrum(0.42f),one_spectrum());
  check(reflected != nullptr && bsdf_diffraction_conductor_multiggx_split_setup(
      kg,&mixed,reflected,make_float3(1,0,0),request.pitch_nm,request.depth_nm,
      request.duty,1,0), "mixed reflective dielectric completion");
  Spectrum layer = zero_spectrum();
  for (int i=first_reflection;i<mixed.num_closure;i++) {
    layer += closure_layer_albedo(kg,&mixed,&mixed.closure[i]);
  }
  bsdf_diffuse_setup(&mixed,mixed.N,closure_layering_weight(layer,make_spectrum(0.42f)));
  const int mixed_slots = MAX_CLOSURE - mixed.num_closure_left;
  check(mixed_slots <= 22, "mixed Principled fits native12 plus diffraction10 reservation");
  double mixed_energy = 0;
  for (int c=0;c<mixed.num_closure;c++) {
    const ShaderClosure *sc=&mixed.closure[c];
    double subtotal=0;
    constexpr int samples=8192;
    for(int i=0;i<samples;i++) {
      Spectrum value;float3 wo;float pdf,eta;float2 roughness;
      const int label=bsdf_sample(kg,&mixed,sc,make_float3((i+0.5f)/samples,
          fraction((i+0.5f)*0.61803398875f),fraction((i+0.5f)*0.75487766625f)),
          &value,&wo,&pdf,&roughness,&eta);
      if(label!=LABEL_NONE && pdf>0) subtotal += double(average(sc->weight*value)/pdf);
    }
    mixed_energy += subtotal/samples;
  }
  check(fabs(mixed_energy-1.0)<0.04,"mixed Principled lossless white-world energy");
  /* Exercise the actual SVM node with every native layer enabled. The graph
   * reserves native12 + diffraction10 slots; compare that allocation with a
   * roomy reference, including partial coverage and coated first events.
   * Film cases here test allocation only: their reuse of the uncoated table
   * is not an energy validation of coated cache construction. */
  const auto scalar = [](const float value) { return SVMInputFloat{__float_as_uint(value)}; };
  const auto triple = [&](const float x,const float y,const float z) {
    return SVMInputFloat3{scalar(x),scalar(y),scalar(z)};
  };
  for (const float coverage : {0.0f,0.5f,1.0f}) {
    for (const float film : {0.0f,250.0f}) {
      SVMNodePrincipledBsdfData data{};
      data.distribution=CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID;
      data.ior=scalar(1.5f);data.roughness=scalar(sqrtf(0.3f));
      data.sheen_weight=scalar(0.2f);data.coat_weight=scalar(0.2f);
      data.metallic=scalar(0.3f);data.transmission_weight=scalar(0.4f);
      data.subsurface_weight=scalar(0.2f);data.base_color=triple(1,1,1);
      data.alpha=scalar(1);data.diffuse_roughness=scalar(0.2f);
      data.normal_offset=data.coat_normal_offset=SVM_STACK_INVALID;data.tangent_offset=0;
      data.specular_tint=triple(1,1,1);data.specular_ior_level=scalar(0.5f);
      data.anisotropic=scalar(0);data.anisotropic_rotation=scalar(0);
      data.transmission_dispersion_scale=scalar(0);
      data.transmission_dispersion_abbe_number=scalar(20);
      data.emission_color=triple(0,0,0);data.emission_strength=scalar(0);
      data.sheen_tint=triple(1,1,1);data.sheen_roughness=scalar(0.5f);
      data.coat_tint=triple(1,1,1);data.coat_roughness=scalar(0.2f);data.coat_ior=scalar(1.5f);
      data.subsurface_method=CLOSURE_BSSRDF_RANDOM_WALK_ID;
      data.subsurface_radius=triple(1,1,1);data.subsurface_scale=scalar(0.1f);
      data.subsurface_ior=scalar(1.5f);data.subsurface_anisotropy=scalar(0);
      data.thin_film_thickness=scalar(film);data.thin_film_ior=scalar(1.32f);
      data.thin_wall={0,SVM_STACK_INVALID,{0,0,0}};
      data.diffraction_weight=scalar(coverage);data.diffraction_pitch=scalar(request.pitch_nm);
      data.diffraction_depth=scalar(request.depth_nm);data.diffraction_duty=scalar(request.duty);
      data.diffraction_albedo_handle=data.diffraction_two_sided_handle=0;
      std::vector<uint> node_words(sizeof(data)/sizeof(uint));
      std::memcpy(node_words.data(),&data,sizeof(data));
      globals.svm_nodes.data=node_words.data();globals.svm_nodes.width=int(node_words.size());
      const SVMNodeClosureBsdf node{CLOSURE_BSDF_PRINCIPLED_ID,SVM_STACK_INVALID,{0,0,0}};
      ShaderData roomy{},reserved{};
      for(ShaderData *sd:{&roomy,&reserved}) {
        sd->N=sd->Ng=make_float3(0,0,1);sd->wi=mixed.wi;sd->rand_wavelength=0.5f;
        sd->num_closure_left=sd==&roomy?MAX_CLOSURE:22;
        float stack[SVM_STACK_SIZE]={1,0,0};
        svm_node_closure_bsdf<~uint64_t(0),SHADER_TYPE_SURFACE>(
            kg,sd,stack,one_spectrum(),node,PATH_RAY_VISIBILITY_CAMERA,0,0);
      }
      check(roomy.num_closure==reserved.num_closure,
            "graph reservation preserves every native and grating closure");
      const int required=MAX_CLOSURE-roomy.num_closure_left;
      check(required<=22 && reserved.num_closure_left==22-required,
            "partial coated all-layer Principled fits actual graph reservation");
      for(int i=0;i<roomy.num_closure;i++) {
        check(roomy.closure[i].type==reserved.closure[i].type &&
              reduce_max(fabs(roomy.closure[i].weight-reserved.closure[i].weight))<1e-7f,
              "reserved SVM allocation loses no closure contribution");
      }
    }
  }
  /* Glass graph reservation is native2 + diffraction5 = 7 after the
   * generalized return storage extension. Prove it on actual SVM routing,
   * including partial coated coverage and a filtered unequal R/T pair. */
  for (const float coverage : {0.0f,0.5f,1.0f}) for (const float film : {0.0f,250.0f}) {
    for (const bool transmission_only : {false,true}) {
      SVMNodeGlassBsdfData data{};
      data.color=triple(1,1,1);data.roughness=scalar(sqrtf(0.3f));data.ior=scalar(1.5f);
      data.thin_film_thickness=scalar(film);data.thin_film_ior=scalar(1.32f);
      data.diffraction_weight=scalar(coverage);data.diffraction_pitch=scalar(request.pitch_nm);
      data.diffraction_depth=scalar(request.depth_nm);data.diffraction_duty=scalar(request.duty);
      data.diffraction_two_sided_handle=0;data.normal_offset=SVM_STACK_INVALID;data.tangent_offset=0;
      std::vector<uint> words(sizeof(data)/sizeof(uint));std::memcpy(words.data(),&data,sizeof(data));
      globals.svm_nodes.data=words.data();globals.svm_nodes.width=int(words.size());
      globals.data.integrator.caustics_reflective=!transmission_only;
      globals.data.integrator.caustics_refractive=true;
      const SVMNodeClosureBsdf node{CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID,
                                   SVM_STACK_INVALID,{0,0,0}};
      ShaderData roomy{},reserved{};
      for (ShaderData *sd : {&roomy,&reserved}) {
        sd->N=sd->Ng=make_float3(0,0,1);sd->wi=mixed.wi;sd->rand_wavelength=.5f;
        sd->num_closure_left=sd==&roomy?MAX_CLOSURE:7;
        float stack[SVM_STACK_SIZE]={1,0,0};
        svm_node_closure_bsdf<~uint64_t(0),SHADER_TYPE_SURFACE>(
            kg,sd,stack,one_spectrum(),node,
            transmission_only?PATH_RAY_VISIBILITY_DIFFUSE:PATH_RAY_VISIBILITY_CAMERA,0,0);
      }
      const int required=MAX_CLOSURE-roomy.num_closure_left;
      check(required<=7 && reserved.num_closure_left==7-required &&
            reserved.num_closure==roomy.num_closure,"actual Glass graph7 preserves all partial/coated closures");
      for (int c=0;c<roomy.num_closure;c++) {
        check(reserved.closure[c].type==roomy.closure[c].type &&
              reduce_max(fabs(reserved.closure[c].weight-roomy.closure[c].weight))<1e-7f,
              "actual Glass reservation preserves weights under unequal R/T filtering");
      }
    }
  }
  std::printf("mixed_energy=%g mixed_slots=%d ",mixed_energy,mixed_slots);
  std::printf("checks=%d failures=%d max_pdf_error=%g max_eval_error=%g "
              "max_reciprocity=%g max_layer_error=%g\n",checks,failures,
              largest_pdf_error,largest_eval_error,largest_reciprocity,largest_layer_error);
  return failures!=0;
}
