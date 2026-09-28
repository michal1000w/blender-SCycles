/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "kernel/closure/bsdf_diffraction_dielectric.h"
#include "kernel/util/diffraction_albedo.h"

CCL_NAMESPACE_BEGIN

/* Keep the original Fresnel payload alive in ShaderData auxiliary storage.
 * Nonzero orders use min(F_i,F_o) componentwise. Their sum is bounded by F_i;
 * the residual mirror carries the remainder. This is the reciprocal Fast
 * scalar model, not an electromagnetic solution of corrugated metal. */
struct DiffractionConductorExtra {
  MicrofacetBsdf carrier;
  DiffractionReflection param;
  int albedo_handle;
  float wavelength_nm;
  Spectrum multiscatter_color;
};
struct DiffractionConductorBsdf {
  SHADER_CLOSURE_BASE;
  packed_float3 T;
  ccl_private DiffractionConductorExtra *extra;
};
static_assert(sizeof(DiffractionConductorBsdf) <= sizeof(ShaderClosure));

ccl_device_inline bool bsdf_is_diffraction_conductor(const ClosureType type)
{
  return type == CLOSURE_BSDF_DIFFRACTION_CONDUCTOR_GGX_ID ||
         type == CLOSURE_BSDF_DIFFRACTION_CONDUCTOR_BECKMANN_ID;
}

ccl_device_inline Spectrum diffraction_conductor_nonzero(
    KernelGlobals kg, ccl_private const DiffractionConductorExtra *e,
    const int order, const float ci, const float co, const Spectrum Fi)
{
  if (!(ci > 0 && co > 0) || order == 0) return zero_spectrum();
  const Spectrum Fo = microfacet_fresnel(kg, &e->carrier, co, nullptr).reflectance;
  return min(Fi, Fo) * diffraction_reflection_nonzero_power(&e->param, order, ci, co);
}

ccl_device_inline Spectrum diffraction_conductor_power(
    KernelGlobals kg, ccl_private const DiffractionConductorExtra *e,
    const int order, const float3 wi, const float3 wo, const float3 h,
    const Spectrum Fi)
{
  const float ci = dot(wi, h);
  if (order != 0) return diffraction_conductor_nonzero(kg, e, order, ci, dot(wo,h), Fi);
  Spectrum other = zero_spectrum();
  const int bound = diffraction_reflection_max_order(&e->param);
  for (int m=-bound; m<=bound; ++m) {
    float3 candidate;
    if (m != 0 && diffraction_facet_reflect(wi,h,make_float3(1,0,0),
                                          m*e->param.wavelength_over_pitch,&candidate)) {
      other += diffraction_conductor_nonzero(kg,e,m,ci,dot(candidate,h),Fi);
    }
  }
  return max(Fi-other, zero_spectrum());
}

/* Proposal probabilities are normalized by the incident Fresnel mean. The
 * physical BSDF retains the full spectral power, including absorption. */
template<MicrofacetType m_type>
ccl_device_inline Spectrum diffraction_conductor_eval(
    KernelGlobals kg, ccl_private const DiffractionConductorExtra *e,
    const float3 wi, const float3 wo, ccl_private float *pdf)
{
  *pdf=0;
  const ccl_private DiffractionReflection *p=&e->param;
  if (!(wi.z>0 && wo.z>0) || roughness_is_almost_specular(p->alpha_x,p->alpha_y))
    return zero_spectrum();
  Spectrum density=zero_spectrum();
  float proposal=0;
  const int bound=diffraction_reflection_max_order(p);
  for (int m=-bound;m<=bound;++m) {
    const float delta=m*p->wavelength_over_pitch;
    DiffractionReflectionRoot roots[2];
    const int count=diffraction_reflection_half_vectors(wi,wo,make_float3(1,0,0),delta,roots);
    for (int r=0;r<count;++r) {
      const float3 h=roots[r].h;
      if (!(h.z>0)) continue;
      const float ci=dot(wi,h);
      const Spectrum Fi=microfacet_fresnel(kg,&e->carrier,ci,nullptr).reflectance;
      const float budget=average(Fi);
      if (!(budget>0)) continue;
      const Spectrum power=diffraction_conductor_power(kg,e,m,wi,wo,h,Fi);
      const float J=diffraction_reflection_jacobian(wi,wo,h,make_float3(1,0,0),delta);
      const float factor=bsdf_aniso_D<m_type>(p->alpha_x,p->alpha_y,h)*ci*J/wi.z;
      density+=power*factor;
      proposal+=average(power)*factor/budget;
    }
  }
  const float li=diffraction_dielectric_lambda<m_type>(p->alpha_x,p->alpha_y,wi);
  const float lo=diffraction_dielectric_lambda<m_type>(p->alpha_x,p->alpha_y,wo);
  *pdf=proposal/(1+li);
  return density/(1+li+lo);
}

ccl_device_inline Spectrum bsdf_diffraction_conductor_eval(
    KernelGlobals kg, ccl_private const ShaderClosure *sc,
    const float3 wi, const float3 wo, ccl_private float *pdf)
{
  const ccl_private DiffractionConductorBsdf *b=(const ccl_private DiffractionConductorBsdf *)sc;
  float3 X,Y;make_orthonormals_safe_tangent(b->N,b->T,&X,&Y);
  const float3 I=make_float3(dot(wi,X),dot(wi,Y),dot(wi,b->N));
  const float3 O=make_float3(dot(wo,X),dot(wo,Y),dot(wo,b->N));
  Spectrum value = b->type==CLOSURE_BSDF_DIFFRACTION_CONDUCTOR_BECKMANN_ID ?
      diffraction_conductor_eval<BECKMANN>(kg,b->extra,I,O,pdf) :
      diffraction_conductor_eval<GGX>(kg,b->extra,I,O,pdf);
  if (b->extra->albedo_handle >= 0 && I.z > 0 && O.z > 0) {
    const float2 incident = diffraction_albedo_lookup(
        kg, b->extra->albedo_handle, b->extra->wavelength_nm, I);
    const float2 outgoing = diffraction_albedo_lookup(
        kg, b->extra->albedo_handle, b->extra->wavelength_nm, O);
    const float missing = 1.0f - incident.x;
    const float average_missing = 1.0f - incident.y;
    if (average_missing > 1e-7f) {
      /* Reciprocal unit-reflector missing-energy lobe. Cosine mixture sampling
       * avoids variable-length rejection; both branches evaluate this PDF. */
      const float cosine_pdf = O.z * M_1_PI_F;
      /* The closure weight already contains one factor of the Glossy color.
       * Additional facet interactions absorb that color again. */
      const Spectrum color = b->extra->multiscatter_color;
      Spectrum multiple_color = safe_divide(color * incident.y,
          one_spectrum() - color * average_missing);
      /* Glossy carries one color factor in closure weight. Physical conductor
       * and F82 closures carry Fresnel inside the BSDF, so retain both factors. */
      if (b->extra->carrier.fresnel_type != MicrofacetFresnel::NONE) {
        multiple_color *= color;
      }
      value += multiple_color * (missing * (1.0f - outgoing.x) * cosine_pdf / average_missing);
      *pdf = (1.0f - missing) * *pdf + missing * cosine_pdf;
    }
  }
  return value;
}

ccl_device_inline int bsdf_diffraction_conductor_label(ccl_private const ShaderClosure *sc)
{
  const ccl_private DiffractionConductorBsdf *b=(const ccl_private DiffractionConductorBsdf *)sc;
  const ccl_private DiffractionReflection *p=&b->extra->param;
  return LABEL_REFLECT | (roughness_is_almost_specular(p->alpha_x,p->alpha_y) ?
                         LABEL_SINGULAR : LABEL_GLOSSY);
}

ccl_device_inline int bsdf_diffraction_conductor_sample(
    KernelGlobals kg, ccl_private const ShaderClosure *sc, const float3 Ng,
    const float3 wi, const float3 random, ccl_private Spectrum *eval,
    ccl_private float3 *wo, ccl_private float *pdf,
    ccl_private float2 *roughness, ccl_private float *eta)
{
  const ccl_private DiffractionConductorBsdf *b=(const ccl_private DiffractionConductorBsdf *)sc;
  const ccl_private DiffractionConductorExtra *e=b->extra;
  const ccl_private DiffractionReflection *p=&e->param;
  *eval=zero_spectrum();*pdf=0;*eta=1;*roughness=make_float2(p->alpha_x,p->alpha_y);
  float3 X,Y;make_orthonormals_safe_tangent(b->N,b->T,&X,&Y);
  const float3 I=make_float3(dot(wi,X),dot(wi,Y),dot(wi,b->N));
  if (!(I.z>0)) return LABEL_NONE;
  const bool singular=roughness_is_almost_specular(p->alpha_x,p->alpha_y);
  float order_random = random.z;
  if (!singular && e->albedo_handle >= 0) {
    const float2 albedo = diffraction_albedo_lookup(kg, e->albedo_handle, e->wavelength_nm, I);
    const float missing = albedo.y < 1.0f - 1e-7f ? 1.0f - albedo.x : 0.0f;
    if (random.z < missing) {
      const float radius = sqrtf(random.x), phi = M_2PI_F * random.y;
      const float3 O = make_float3(radius * cosf(phi), radius * sinf(phi),
                                   safe_sqrtf(1.0f - random.x));
      *wo = O.x * X + O.y * Y + O.z * b->N;
      if (dot(Ng, *wo) <= 0) return LABEL_NONE;
      *eval = bsdf_diffraction_conductor_eval(kg, sc, wi, *wo, pdf);
      return *pdf > 0 ? LABEL_REFLECT | LABEL_GLOSSY : LABEL_NONE;
    }
    order_random = missing < 1 ? (random.z - missing) / (1.0f - missing) : 0.0f;
  }

  const float3 h=singular ? make_float3(0,0,1) :
      b->type==CLOSURE_BSDF_DIFFRACTION_CONDUCTOR_BECKMANN_ID ?
        microfacet_beckmann_sample_vndf(I,p->alpha_x,p->alpha_y,make_float2(random)) :
        microfacet_ggx_sample_vndf(I,p->alpha_x,p->alpha_y,make_float2(random));
  const float ci=dot(I,h);
  const Spectrum Fi=microfacet_fresnel(kg,&e->carrier,ci,nullptr).reflectance;
  const float budget=average(Fi);
  if (!(budget>0)) return LABEL_NONE;
  float u=order_random*budget;
  Spectrum other=zero_spectrum(),power=zero_spectrum();
  float3 O=2*ci*h-I;
  bool selected=false;
  const int bound=diffraction_reflection_max_order(p);
  for (int m=-bound;m<=bound;++m) {
    float3 candidate;
    if (m==0 || !diffraction_facet_reflect(I,h,make_float3(1,0,0),
                                         m*p->wavelength_over_pitch,&candidate)) continue;
    const Spectrum value=diffraction_conductor_nonzero(kg,e,m,ci,dot(candidate,h),Fi);
    const float mass=average(value);
    if (u<mass) {O=candidate;power=value;selected=true;break;}
    u-=mass;other+=value;
  }
  if (!selected) power=max(Fi-other,zero_spectrum());
  *wo=O.x*X+O.y*Y+O.z*b->N;
  if (!(O.z>0 && dot(Ng,*wo)>0)) return LABEL_NONE;
  if (singular) {
    *pdf=average(power)/budget*1e6f;*eval=power*1e6f;*roughness=zero_float2();
  }
  else *eval=bsdf_diffraction_conductor_eval(kg,sc,wi,*wo,pdf);
  return *pdf>0 ? bsdf_diffraction_conductor_label(sc) : LABEL_NONE;
}

ccl_device_inline Spectrum bsdf_diffraction_conductor_delta(
    KernelGlobals kg, ccl_private const ShaderClosure *sc,
    const float3 wi, const float3 wo, ccl_private float *pdf)
{
  *pdf=0;
  if (!(bsdf_diffraction_conductor_label(sc)&LABEL_SINGULAR)) return zero_spectrum();
  const ccl_private DiffractionConductorBsdf *b=(const ccl_private DiffractionConductorBsdf *)sc;
  const ccl_private DiffractionConductorExtra *e=b->extra;
  float3 X,Y;make_orthonormals_safe_tangent(b->N,b->T,&X,&Y);
  const float3 I=make_float3(dot(wi,X),dot(wi,Y),dot(wi,b->N));
  const float3 O=make_float3(dot(wo,X),dot(wo,Y),dot(wo,b->N));
  if (!(I.z>0 && O.z>0)) return zero_spectrum();
  const float order=roundf((I.x+O.x)/e->param.wavelength_over_pitch);
  if (fabsf(order)>diffraction_reflection_max_order(&e->param) ||
      sqr(I.x+O.x-order*e->param.wavelength_over_pitch)+sqr(I.y+O.y)>16e-12f)
    return zero_spectrum();
  const Spectrum Fi=microfacet_fresnel(kg,&e->carrier,I.z,nullptr).reflectance;
  const float budget=average(Fi);
  if (!(budget>0)) return zero_spectrum();
  const Spectrum power=diffraction_conductor_power(kg,e,int(order),I,O,make_float3(0,0,1),Fi);
  *pdf=average(power)/budget*1e6f;
  return power*1e6f;
}

/* Convert an already configured single-scattering reflection closure. Its
 * original Fresnel auxiliary block remains owned by the same ShaderData. */
ccl_device_inline bool bsdf_diffraction_conductor_setup(
    ccl_private ShaderData *sd, ccl_private MicrofacetBsdf *source,
    const float3 tangent, const float pitch, const float depth, const float duty,
    const float medium_ior = 1.0f)
{
  if (depth == 0.0f) return false; /* Caller retains exact native setup. */
  if (!(pitch>0 && pitch<=1000000 && depth>=0 && medium_ior>0) ||
      !isfinite_safe(depth) || !isfinite_safe(medium_ior) ||
      (source->type!=CLOSURE_BSDF_MICROFACET_GGX_ID &&
       source->type!=CLOSURE_BSDF_MICROFACET_BECKMANN_ID) ||
      (source->fresnel_type!=MicrofacetFresnel::NONE &&
       source->fresnel_type!=MicrofacetFresnel::CONDUCTOR &&
       source->fresnel_type!=MicrofacetFresnel::F82_TINT &&
       source->fresnel_type!=MicrofacetFresnel::GENERALIZED_SCHLICK)) return false;
  const float wavelength=1000*sample_wavelength(sd->rand_wavelength)/medium_ior;
  const DiffractionReflection p={source->alpha_x,source->alpha_y,wavelength/pitch,
                                depth/wavelength,saturatef(duty)};
  if (!(p.wavelength_over_pitch>0.0f) || !isfinite_safe(p.wavelength_over_pitch)) return false;
  ccl_private DiffractionConductorExtra *e=(ccl_private DiffractionConductorExtra *)
      closure_alloc_extra(sd,sizeof(DiffractionConductorExtra));
  if (!e) return false;
  source->diffraction_wavelength_nm = 1000 * sample_wavelength(sd->rand_wavelength);
  e->carrier=*source;e->param=p;
  e->albedo_handle=-1;e->wavelength_nm=1000*sample_wavelength(sd->rand_wavelength);
  e->multiscatter_color=one_spectrum();
  const bool beckmann=source->type==CLOSURE_BSDF_MICROFACET_BECKMANN_ID;
  ccl_private DiffractionConductorBsdf *b=(ccl_private DiffractionConductorBsdf *)source;
  b->T=tangent;b->extra=e;
  b->type=beckmann ? CLOSURE_BSDF_DIFFRACTION_CONDUCTOR_BECKMANN_ID :
                    CLOSURE_BSDF_DIFFRACTION_CONDUCTOR_GGX_ID;
  sd->runtime_flag|=SR_BSDF_HAS_DISPERSION;
  return true;
}

ccl_device_inline bool bsdf_diffraction_conductor_split_setup(
    ccl_private ShaderData *sd, ccl_private MicrofacetBsdf *source,
    const float3 tangent, const float pitch, const float depth, const float duty,
    const float coverage)
{
  const float fraction=depth > 0.0f ? saturatef(coverage) : 0.0f;
  if (fraction==0) return true;
  const int extra=(sizeof(DiffractionConductorExtra)+sizeof(ShaderClosure)-1)/sizeof(ShaderClosure);
  if (sd->num_closure_left<extra+int(fraction<1)) return false;
  if (fraction==1) return bsdf_diffraction_conductor_setup(sd,source,tangent,pitch,depth,duty);
  ccl_private MicrofacetBsdf *grating=(ccl_private MicrofacetBsdf *)bsdf_alloc(
      sd,sizeof(MicrofacetBsdf),source->weight*fraction);
  if (!grating) return false;
  *grating=*source;
  grating->weight*=fraction;grating->sample_weight*=fraction;
  if (!bsdf_diffraction_conductor_setup(sd,grating,tangent,pitch,depth,duty)) {
    grating->type=CLOSURE_NONE_ID;grating->sample_weight=0;grating->weight=zero_spectrum();
    return false;
  }
  /* Both covered and uncovered portions travel on the same spectral path. */
  source->diffraction_wavelength_nm =
      ((ccl_private DiffractionConductorBsdf *)grating)->extra->wavelength_nm;
  source->weight*=1-fraction;source->sample_weight*=1-fraction;
  return true;
}

/* Average Fresnel used only by the additional scattering lobe. Uncoated
 * materials use the native approximation; film uses 16-point Gauss quadrature
 * of the actual carrier response against projected solid angle. */
ccl_device_inline Spectrum diffraction_conductor_average_fresnel(
    KernelGlobals kg, ccl_private const MicrofacetBsdf *bsdf, const bool include_film)
{
  Spectrum average_fresnel = one_spectrum();
  bool has_film = false;
  if (bsdf->fresnel_type == MicrofacetFresnel::CONDUCTOR) {
    ccl_private const FresnelConductor *f = (ccl_private const FresnelConductor *)bsdf->fresnel;
    average_fresnel = fresnel_conductor_Fss(f->ior);
    has_film = f->thin_film.thickness > THINFILM_THICKNESS_CUTOFF;
  }
  else if (bsdf->fresnel_type == MicrofacetFresnel::F82_TINT) {
    ccl_private const FresnelF82Tint *f = (ccl_private const FresnelF82Tint *)bsdf->fresnel;
    average_fresnel = fresnel_f82_Fss(f->f0, f->b);
    has_film = f->thin_film.thickness > THINFILM_THICKNESS_CUTOFF;
  }
  else if (bsdf->fresnel_type == MicrofacetFresnel::GENERALIZED_SCHLICK) {
    ccl_private const FresnelGeneralizedSchlick *f =
        (ccl_private const FresnelGeneralizedSchlick *)bsdf->fresnel;
    has_film = f->thin_film.thickness > THINFILM_THICKNESS_CUTOFF;
    /* Match the native reflective-layer average when no film is included.
     * This lobe is separate from Principled transmission and has zero
     * transmissive tint. */
    float s;
    if (f->exponent < 0.0f) {
      const float F0 = F0_from_ior(bsdf->ior);
      s = saturatef(inverse_lerp(F0, 1.0f, fresnel_dielectric_Fss(bsdf->ior)));
    }
    else {
      s = 2.0f / ((f->exponent + 3.0f) * f->exponent + 2.0f);
    }
    average_fresnel = f->tint.reflectance * mix(f->f0, f->f90, s);
  }
  if (!include_film || !has_film) return saturate(average_fresnel);
  const float nodes[8] = {0.095012510f, 0.281603551f, 0.458016778f, 0.617876244f,
                          0.755404408f, 0.865631202f, 0.944575023f, 0.989400935f};
  const float weights[8] = {0.189450610f, 0.182603415f, 0.169156519f, 0.149595989f,
                            0.124628971f, 0.095158512f, 0.062253524f, 0.027152459f};
  average_fresnel = zero_spectrum();
  for (int i = 0; i < 8; ++i) {
    const float lo = 0.5f * (1.0f - nodes[i]);
    const float hi = 0.5f * (1.0f + nodes[i]);
    average_fresnel += weights[i] * (
        lo * microfacet_fresnel(kg, bsdf, lo, nullptr).reflectance +
        hi * microfacet_fresnel(kg, bsdf, hi, nullptr).reflectance);
  }
  return saturate(average_fresnel);
}

/* The source must be initialized without native planar energy preservation.
 * Split first, then compensate the two physical models independently. */
ccl_device_inline bool bsdf_diffraction_conductor_multiggx_split_setup(
    KernelGlobals kg, ccl_private ShaderData *sd, ccl_private MicrofacetBsdf *source,
    const float3 tangent, const float pitch, const float depth, const float duty,
    const float coverage, const int albedo_handle)
{
  kernel_assert(source->energy_scale == 1.0f);
  const float fraction = depth > 0.0f ? saturatef(coverage) : 0.0f;
  Spectrum native_fss = diffraction_conductor_average_fresnel(kg, source, false);
  const Spectrum grating_fss = albedo_handle >= 0 && fraction > 0 ?
      diffraction_conductor_average_fresnel(kg, source, true) : one_spectrum();
  const int grating_index = sd->num_closure;
  if (!bsdf_diffraction_conductor_split_setup(sd, source, tangent, pitch, depth, duty, fraction)) {
    microfacet_ggx_preserve_energy(kg, source, sd->wi, native_fss);
    return false;
  }
  if (fraction > 0) {
    ccl_private DiffractionConductorBsdf *grating = fraction == 1.0f ?
        (ccl_private DiffractionConductorBsdf *)source :
        (ccl_private DiffractionConductorBsdf *)&sd->closure[grating_index];
    grating->extra->albedo_handle = albedo_handle;
    grating->extra->multiscatter_color = microfacet_diffraction_spectral_reflectance(
        kg, grating_fss, grating->extra->wavelength_nm);
  }
  if (fraction < 1.0f) {
    native_fss = microfacet_diffraction_spectral_reflectance(
        kg, native_fss, source->diffraction_wavelength_nm);
    microfacet_ggx_preserve_energy(kg, source, sd->wi, native_fss);
  }
  return true;
}

/* Glossy retains its existing wavelength-dependent color basis. Its uncoated
 * reflector has unit facet Fresnel; the closure weight carries the color. */
ccl_device_inline bool bsdf_diffraction_glossy_setup(
    KernelGlobals kg, ccl_private ShaderData *sd, const Spectrum weight,
    const float3 N, const float3 T, const float alpha_x, const float alpha_y,
    const float pitch, const float depth, const float duty, const float medium_ior,
    const bool beckmann, const int albedo_handle = -1,
    const Spectrum multiscatter_color = one_spectrum())
{
  ccl_private MicrofacetBsdf *source=(ccl_private MicrofacetBsdf *)bsdf_alloc(
      sd,sizeof(MicrofacetBsdf), depth == 0.0f ? weight :
          bsdf_spectral_transmission_color(kg,sd,weight));
  if (!source) return false;
  source->N=N;source->T=T;source->ior=1.0f;
  source->alpha_x=saturatef(alpha_x);source->alpha_y=saturatef(alpha_y);
  sd->runtime_flag |= beckmann ? bsdf_microfacet_beckmann_setup(source) :
                               bsdf_microfacet_ggx_setup(source);
  if (depth == 0.0f) return true;
  if (bsdf_diffraction_conductor_setup(sd,source,T,pitch,depth,duty,medium_ior)) {
    ccl_private DiffractionConductorExtra *extra =
        ((ccl_private DiffractionConductorBsdf *)source)->extra;
    extra->albedo_handle = albedo_handle;
    extra->multiscatter_color = saturate(bsdf_spectral_transmission_color(
        kg, sd, multiscatter_color));
    return true;
  }
  source->type=CLOSURE_NONE_ID;source->weight=zero_spectrum();source->sample_weight=0;
  return false;
}

CCL_NAMESPACE_END
