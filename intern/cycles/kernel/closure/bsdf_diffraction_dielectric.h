/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/closure/bsdf_diffraction.h"
#include "kernel/util/diffraction_two_sided.h"
#include "kernel/util/diffraction_two_sided_lookup.h"

CCL_NAMESPACE_BEGIN

/* Intensity-only Fast scalar approximation.
 * Returns the same transport convention as Cycles' microfacet dielectric;
 * the integrator is responsible for its existing adjoint/eta conversion. */
struct DiffractionRoughDielectric {
  DiffractionDielectricFacet facet;
  float alpha_x, alpha_y;
};

/* Convert physical node inputs to the Fast scalar relief model. Indices are
 * absolute and already oriented to the incident side. The transmission phase
 * is the optical-path difference of the two relief heights, independent of
 * facet angle in this approximation. Reversing the interface reverses phase;
 * binary intensity coefficients are invariant under that sign change. */
ccl_device_inline bool diffraction_dielectric_parameters(
    const float wavelength_nm, const float pitch_nm, const float depth_nm,
    const float duty, const float incident_ior, const float transmitted_ior,
    const float alpha_x, const float alpha_y,
    ccl_private DiffractionRoughDielectric *result)
{
  if (!(wavelength_nm>0 && pitch_nm>0 && depth_nm>=0 &&
        incident_ior>0 && transmitted_ior>0 && duty>=0 && duty<=1 &&
        alpha_x>=0 && alpha_x<=1 && alpha_y>=0 && alpha_y<=1) ||
      !isfinite_safe(wavelength_nm+pitch_nm+depth_nm+incident_ior+transmitted_ior))
    return false;
  const float depth_over_wavelength=depth_nm/wavelength_nm;
  DiffractionRoughDielectric p;
  p.facet.wavelength_over_pitch=(wavelength_nm/pitch_nm)/incident_ior;
  p.facet.height_over_wavelength=depth_over_wavelength*incident_ior;
  p.facet.duty=duty;
  p.facet.incident_ior=incident_ior;
  p.facet.transmitted_ior=transmitted_ior;
  p.facet.transmission_phase=M_2PI_F*depth_over_wavelength*(incident_ior-transmitted_ior);
  p.alpha_x=alpha_x;p.alpha_y=alpha_y;
  if (!isfinite_safe(p.facet.height_over_wavelength) ||
      !isfinite_safe(p.facet.transmission_phase) ||
      diffraction_dielectric_facet_order_bound(&p.facet,false)==0 ||
      diffraction_dielectric_facet_order_bound(&p.facet,true)==0) return false;
  *result=p;
  return true;
}

/* The straight-through mass is independent of the chosen visible facet.
 * It can therefore be split before facet sampling without changing the NDF. */
ccl_device_inline float diffraction_dielectric_straight_mass(
    ccl_private const DiffractionRoughDielectric *p)
{
  return p->facet.incident_ior==p->facet.transmitted_ior ?
             diffraction_binary_power(0,p->facet.transmission_phase,p->facet.duty):0.0f;
}

ccl_device_inline bool diffraction_dielectric_facet_sample(
    ccl_private const DiffractionDielectricFacet *p, const float3 wi, const float3 h,
    const float random, ccl_private float3 *wo, ccl_private float *mass,
    ccl_private int *order, ccl_private bool *transmission,
    const bool exclude_straight = false, const float film_ior=1.0f,
    const float film_thickness_over_wavelength=0.0f, const bool pure_refraction=false)
{
  *mass = 0.0f;
  *wo = zero_float3();
  if (!(random >= 0.0f && random < 1.0f && dot(wi,h) > 0.0f)) return false;
  const float3 axis = make_float3(1,0,0);
  const float eta = p->incident_ior / p->transmitted_ior;
  const float budget = exclude_straight ?
      diffraction_dielectric_facet_nonstraight_budget(
          p,dot(wi,h),film_ior,film_thickness_over_wavelength):1.0f;
  if (!(budget>0) || (exclude_straight && p->incident_ior!=p->transmitted_ior)) return false;
  const float ci = dot(wi,h);
  if (pure_refraction) {
    float u=random*budget;
    const int bound=diffraction_dielectric_facet_active_order_bound(p,true);
    for (int m=-bound;m<=bound;++m) {
      if (exclude_straight && m==0) continue;
      float3 candidate;
      if (!diffraction_facet_transmit(wi,h,axis,eta,m*p->wavelength_over_pitch*eta,&candidate))
        continue;
      const float power=diffraction_binary_power(m,p->transmission_phase,p->duty);
      if (u<power) {
        *wo=candidate;*mass=power;*order=m;*transmission=true;
        return true;
      }
      u-=power;
    }
    return false;
  }
  const float Fi = film_thickness_over_wavelength > 0.0f ?
      diffraction_thin_film_reflectance(ci, p->incident_ior, p->transmitted_ior,
                                      film_ior, film_thickness_over_wavelength) :
      fresnel_dielectric(ci, p->transmitted_ior / p->incident_ior, nullptr);
  float u = random*budget, other = 0.0f;
  for (int side=0; side<2; ++side) {
    const bool trans = side != 0;
    const int bound = diffraction_dielectric_facet_active_order_bound(p,trans);
    for (int m=-bound; m<=bound; ++m) {
      if (m==0 && (!trans || exclude_straight)) continue;
      float3 candidate;
      const float delta = m*p->wavelength_over_pitch;
      const bool valid = trans ?
          diffraction_facet_transmit(wi,h,axis,eta,delta*eta,&candidate) :
          diffraction_facet_reflect(wi,h,axis,delta,&candidate);
      if (!valid) continue;
      const float power = diffraction_dielectric_facet_power(
          p,m,trans,ci,dot(candidate,h),film_ior,film_thickness_over_wavelength,Fi);
      if (u < power) {
        *wo=candidate; *mass=power; *order=m; *transmission=trans;
        return true;
      }
      u-=power;
      other+=power;
    }
  }
  *mass=budget-other;
  *wo=2*dot(wi,h)*h-wi;
  *order=0; *transmission=false;
  return *mass>0.0f;
}

/* Beckmann's rational Lambda fit can dip below zero near its cutoff.
 * Lambda represents a nonnegative masked-area ratio. Bound that approximation
 * before forming G1/G2, rather than clipping an evaluated BSDF or its PDF. */
template<MicrofacetType m_type>
ccl_device_inline float diffraction_dielectric_lambda(const float alpha_x,
                                                       const float alpha_y,
                                                       const float3 w)
{
  const float lambda=bsdf_aniso_lambda<m_type>(alpha_x,alpha_y,w);
  return m_type==BECKMANN ? max(0.0f,lambda):lambda;
}

/* f*abs(cos_o), density per outgoing solid angle. Discrete events are handled
 * by the sampler; their masses must never be mixed into this continuous PDF. */
template<MicrofacetType m_type = GGX>
ccl_device_inline float diffraction_dielectric_eval(
    ccl_private const DiffractionRoughDielectric *p, const float3 wi, const float3 wo,
    ccl_private float *pdf, const float film_ior=1.0f,
    const float film_thickness_over_wavelength=0.0f, const bool exclude_straight=false, const bool pure_refraction=false)
{
  *pdf=0.0f;
  if (!(wi.z>0) || wo.z==0 || roughness_is_almost_specular(p->alpha_x,p->alpha_y)) return 0;
  const bool trans=wo.z<0;
  if (pure_refraction && !trans) return 0;
  const float eta=p->facet.incident_ior/p->facet.transmitted_ior;
  const float3 axis=make_float3(1,0,0);
  const int bound=diffraction_dielectric_facet_active_order_bound(&p->facet,trans);
  float density=0,proposal_density=0;
  for (int m=-bound; m<=bound; ++m) {
    if (trans && m==0 && eta==1.0f) continue;
    const float delta=m*p->facet.wavelength_over_pitch*(trans?eta:1.0f);
    float3 normals[2];
    int count;
    if (trans) {
      DiffractionTransmissionRoot roots[2];
      const float ni=p->facet.incident_ior,no=p->facet.transmitted_ior;
      count=diffraction_transmission_half_vectors_momentum(
          wi,wo,axis,diffraction_symmetric_momentum(ni,wi,no,wo),m*p->facet.wavelength_over_pitch*ni,roots);
      for (int r=0;r<count;++r) normals[r]=roots[r].h;
    }
    else {
      DiffractionReflectionRoot roots[2];
      count=diffraction_reflection_half_vectors(wi,wo,axis,delta,roots);
      for (int r=0;r<count;++r) normals[r]=roots[r].h;
    }
    for (int r=0;r<count;++r) {
      const float3 h=normals[r];
      if (!(h.z>0)) continue;
      const float power=(!trans && m==0)?
          diffraction_dielectric_facet_residual(&p->facet,wi,h,axis,film_ior,film_thickness_over_wavelength):
          diffraction_dielectric_facet_power(&p->facet,m,trans,dot(wi,h),dot(wo,h),film_ior,film_thickness_over_wavelength,-1.0f,pure_refraction);
      const float jacobian=trans?
          diffraction_transmission_jacobian_indices(wi,wo,h,axis,p->facet.incident_ior,
              p->facet.transmitted_ior,m*p->facet.wavelength_over_pitch*p->facet.incident_ior):
          diffraction_reflection_jacobian(wi,wo,h,axis,delta);
      const float contribution=power*bsdf_aniso_D<m_type>(p->alpha_x,p->alpha_y,h)*dot(wi,h)*jacobian/wi.z;
      density+=contribution;
      if (exclude_straight) {
        const float budget=diffraction_dielectric_facet_nonstraight_budget(
            &p->facet,dot(wi,h),film_ior,film_thickness_over_wavelength);
        if (budget>0) proposal_density+=contribution/budget;
      }
    }
  }
  const float li=diffraction_dielectric_lambda<m_type>(p->alpha_x,p->alpha_y,wi);
  const float lo=diffraction_dielectric_lambda<m_type>(p->alpha_x,p->alpha_y,wo);
  *pdf=(exclude_straight?proposal_density:density)/(1+li);
  return density/(1+li+lo);
}

/* Direction sampling is separate so the closure can evaluate its PDF after
 * the final local-to-world rounding, without duplicating the expensive order
 * sum. The public math sampler below evaluates in its own local frame. */
template<MicrofacetType m_type = GGX>
ccl_device_inline bool diffraction_dielectric_sample_direction(
    ccl_private const DiffractionRoughDielectric *p, const float3 wi, const float3 random,
    ccl_private float3 *wo, ccl_private float *mass, ccl_private bool *singular,
    const bool exclude_straight = false, const float film_ior=1.0f,
    const float film_thickness_over_wavelength=0.0f, const bool pure_refraction=false)
{
  *mass=0;
  *wo=zero_float3();
  *singular=false;
  if (!(wi.z>0)) return false;
  const bool smooth=roughness_is_almost_specular(p->alpha_x,p->alpha_y);
  const float3 h=smooth?make_float3(0,0,1):
      (m_type==BECKMANN ?
       microfacet_beckmann_sample_vndf(wi,p->alpha_x,p->alpha_y,make_float2(random)) :
       microfacet_ggx_sample_vndf(wi,p->alpha_x,p->alpha_y,make_float2(random)));
  int order; bool trans;
  if (!diffraction_dielectric_facet_sample(&p->facet,wi,h,random.z,wo,mass,&order,&trans,
                                        exclude_straight,film_ior,film_thickness_over_wavelength,pure_refraction) ||
      ((wo->z<0)!=trans) || wo->z==0) return false;
  const bool straight=trans && order==0 && p->facet.incident_ior==p->facet.transmitted_ior;
  *singular=smooth || straight;
  return *mass>0;
}

template<MicrofacetType m_type = GGX>
ccl_device_inline bool diffraction_dielectric_sample(
    ccl_private const DiffractionRoughDielectric *p, const float3 wi, const float3 random,
    ccl_private float3 *wo, ccl_private float *value, ccl_private float *pdf,
    ccl_private bool *singular)
{
  *value=*pdf=0;
  float mass;
  if (!diffraction_dielectric_sample_direction<m_type>(p,wi,random,wo,&mass,singular)) return false;
  if (*singular) *value=*pdf=mass;
  else *value=diffraction_dielectric_eval<m_type>(p,wi,*wo,pdf);
  return *pdf>0;
}

/* Conditional continuous sampler for the split closure. Its closure weight
 * must be multiplied by (1-straight_mass); the separate atom gets straight_mass.
 * No rejection loop, random-number recycling, or lost atom probability. */
template<MicrofacetType m_type = GGX>
ccl_device_inline bool diffraction_dielectric_sample_continuous(
    ccl_private const DiffractionRoughDielectric *p, const float3 wi, const float3 random,
    ccl_private float3 *wo, ccl_private float *value, ccl_private float *pdf)
{
  *value=*pdf=0;
  const float atom=diffraction_dielectric_straight_mass(p), budget=1-atom;
  if (!(budget>0) || roughness_is_almost_specular(p->alpha_x,p->alpha_y)) return false;
  float mass;bool singular;
  if (!diffraction_dielectric_sample_direction<m_type>(p,wi,random,wo,&mass,&singular,atom>0) || singular)
    return false;
  *value=diffraction_dielectric_eval<m_type>(p,wi,*wo,pdf)/budget;
  *pdf/=budget;
  return *pdf>0;
}

/* A coating makes the matched-index atom depend on the visible facet. Draw
 * the ordinary VNDF, then condition the order selection at that facet. The
 * evaluator sums the resulting per-root proposal densities. The returned
 * value is the physical continuous BSDF, not divided by an atom mixture weight;
 * a future mixed closure must account for its own discrete/continuous choice. */
template<MicrofacetType m_type = GGX>
ccl_device_inline bool diffraction_dielectric_sample_coated_continuous(
    ccl_private const DiffractionRoughDielectric *p, const float3 wi, const float3 random,
    const float film_ior, const float film_thickness_over_wavelength,
    ccl_private float3 *wo, ccl_private float *value, ccl_private float *pdf)
{
  *value=*pdf=0;
  *wo=zero_float3();
  if (roughness_is_almost_specular(p->alpha_x,p->alpha_y)) return false;
  const bool exclude_straight=p->facet.incident_ior==p->facet.transmitted_ior;
  float mass;bool singular;
  if (!diffraction_dielectric_sample_direction<m_type>(
          p,wi,random,wo,&mass,&singular,exclude_straight,film_ior,film_thickness_over_wavelength) ||
      singular) return false;
  *value=diffraction_dielectric_eval<m_type>(
      p,wi,*wo,pdf,film_ior,film_thickness_over_wavelength,exclude_straight);
  return *pdf>0;
}

/* Deterministic VNDF quadrature for the coated straight-through atom. This is
 * an approximation to a facet integral, unlike the exact uncoated constant.
 * The caller chooses a power-of-two work bound; no quality default is selected
 * here. Canonicalizing the tangential signs preserves reversal symmetry for
 * the symmetric anisotropic GGX/Beckmann normal distributions. */
template<MicrofacetType m_type = GGX>
ccl_device_inline bool diffraction_dielectric_coated_straight_mass_quadrature(
    ccl_private const DiffractionRoughDielectric *p, const float3 wi,
    const float film_ior, const float film_thickness_over_wavelength,
    const int samples, ccl_private float *mass)
{
  *mass=0;
  if (!(wi.z>0) || samples<1 || samples>4096 || (samples&(samples-1))!=0 ||
      !(film_ior>0 && film_thickness_over_wavelength>=0) || !isfinite_safe(wi) ||
      !isfinite_safe(film_ior+film_thickness_over_wavelength)) return false;
  if (p->facet.incident_ior!=p->facet.transmitted_ior) return true;
  const float coefficient=diffraction_dielectric_straight_mass(p);
  if (film_thickness_over_wavelength==0) {*mass=coefficient;return true;}
  if (roughness_is_almost_specular(p->alpha_x,p->alpha_y)) {
    *mass=coefficient*(1-diffraction_thin_film_pair_reflectance(
        wi.z,wi.z,p->facet.incident_ior,p->facet.transmitted_ior,
        film_ior,film_thickness_over_wavelength));
    return true;
  }
  const float3 incident=make_float3(fabsf(wi.x),fabsf(wi.y),wi.z);
  float total=0;
  for(int i=0;i<samples;++i) {
    const float2 point=make_float2((i+.5f)/samples,
        float(reverse_integer_bits(uint(i))>>8)*0x1p-24f+.5f/samples);
    const float3 h=m_type==BECKMANN ?
        microfacet_beckmann_sample_vndf(incident,p->alpha_x,p->alpha_y,point):
        microfacet_ggx_sample_vndf(incident,p->alpha_x,p->alpha_y,point);
    const float ci=dot(incident,h);
    total+=1-diffraction_thin_film_pair_reflectance(
        ci,ci,p->facet.incident_ior,p->facet.transmitted_ior,
        film_ior,film_thickness_over_wavelength);
  }
  *mass=coefficient*(total/samples);
  return isfinite_safe(*mass);
}

/* Reference mixture for validating the complete coated local model. Renderer
 * closures still need separate measures for guiding. `atom_mass` is the bounded
 * quadrature result for this incident direction. Keep both proposal branches
 * alive: quadrature rounding must never remove support from a real lobe. */
ccl_device_inline float diffraction_dielectric_coated_atom_probability(
    ccl_private const DiffractionRoughDielectric *p,const float atom_mass)
{
  return p->facet.incident_ior==p->facet.transmitted_ior ? clamp(atom_mass,.05f,.95f):0;
}

template<MicrofacetType m_type = GGX>
ccl_device_inline float diffraction_dielectric_eval_coated_mixture(
    ccl_private const DiffractionRoughDielectric *p,const float3 wi,const float3 wo,
    const float film_ior,const float film_thickness_over_wavelength,const float atom_mass,
    ccl_private float *pdf)
{
  const bool matched=p->facet.incident_ior==p->facet.transmitted_ior;
  const float value=diffraction_dielectric_eval<m_type>(
      p,wi,wo,pdf,film_ior,film_thickness_over_wavelength,matched);
  *pdf*=1-diffraction_dielectric_coated_atom_probability(p,atom_mass);
  return value;
}

template<MicrofacetType m_type = GGX>
ccl_device_inline bool diffraction_dielectric_sample_coated_mixture(
    ccl_private const DiffractionRoughDielectric *p,const float3 wi,const float3 random,
    const float film_ior,const float film_thickness_over_wavelength,const float atom_mass,
    ccl_private float3 *wo,ccl_private float *value,ccl_private float *pdf,
    ccl_private bool *singular)
{
  *wo=zero_float3();*value=*pdf=0;*singular=false;
  if (!(wi.z>0 && atom_mass>=0 && atom_mass<=1 && random.z>=0 && random.z<1)) return false;
  if (roughness_is_almost_specular(p->alpha_x,p->alpha_y)) {
    if (!diffraction_dielectric_sample_direction<m_type>(
            p,wi,random,wo,value,singular,false,film_ior,film_thickness_over_wavelength)) return false;
    *pdf=*value;
    return *pdf>0;
  }
  const float probability=diffraction_dielectric_coated_atom_probability(p,atom_mass);
  if (random.z<probability) {
    *wo=-wi;*value=atom_mass;*pdf=probability;*singular=true;
    return true;
  }
  const float3 remapped=make_float3(random.x,random.y,
      min((random.z-probability)/(1-probability),0x1.fffffep-1f));
  if (!diffraction_dielectric_sample_coated_continuous<m_type>(
          p,wi,remapped,film_ior,film_thickness_over_wavelength,wo,value,pdf))return false;
  *pdf*=1-probability;
  return *pdf>0;
}

/* Compact inline storage for shared tints; differing tints use auxiliary data
 * owned by the same ShaderData allocation. Such closures must not outlive that
 * ShaderData or be copied without their auxiliary storage. Front/back material
 * indices are oriented by shader setup, as for the ordinary Glass closure. */
enum { DIFFRACTION_DIELECTRIC_TINT_EXTRA = 1 << 30,
       DIFFRACTION_DIELECTRIC_COATING_EXTRA = 1 << 29,
       DIFFRACTION_DIELECTRIC_PURE_REFRACTION = 1 << 28,
       DIFFRACTION_DIELECTRIC_GENERALIZED = 1 << 27,
       DIFFRACTION_DIELECTRIC_TWO_SIDED_MS = 1 << 26,
       DIFFRACTION_DIELECTRIC_CACHE_BASE = 1 << 25,
       DIFFRACTION_DIELECTRIC_POLARIZER = 1 << 24 };
struct DiffractionDielectricTintExtra {
  DiffractionRoughDielectric param;
  Spectrum reflection,transmission;
};
struct DiffractionDielectricGeneralizedExtra {
  DiffractionDielectricTintExtra base;
  Spectrum generalized_f0;
  float generalized_reference_f0;
  float film_ior,film_thickness_over_wavelength;
};
struct DiffractionDielectricCoatingExtra {
  DiffractionDielectricTintExtra base;
  float film_ior,film_thickness_over_wavelength;
};
struct DiffractionDielectricTwoSidedExtra {
  DiffractionDielectricTintExtra base;
  int cache_handle;
  int incoming_side;
  float inside_ior;
  float wavelength_nm;
  Spectrum generalized_f0; /* ignored by single-basis physical caches */
  float ior_d, inv_abbe;
};
struct DiffractionDielectricBsdf {
  SHADER_CLOSURE_BASE;
  packed_float3 T;
  int disabled_lobes; /* LABEL_REFLECT / LABEL_TRANSMIT; zero enables both. */
  union {
    DiffractionRoughDielectric param;
    ccl_private DiffractionDielectricTintExtra *extra;
  };
};
ccl_device_inline ccl_private const DiffractionRoughDielectric *diffraction_dielectric_param(
    ccl_private const DiffractionDielectricBsdf *bsdf)
{
  return (bsdf->disabled_lobes & DIFFRACTION_DIELECTRIC_TINT_EXTRA)?&bsdf->extra->param:&bsdf->param;
}
ccl_device_inline Spectrum diffraction_dielectric_tint(
    ccl_private const DiffractionDielectricBsdf *bsdf,const bool transmission)
{
  if (!(bsdf->disabled_lobes & DIFFRACTION_DIELECTRIC_TINT_EXTRA)) return one_spectrum();
  return transmission?bsdf->extra->transmission:bsdf->extra->reflection;
}
ccl_device_inline ccl_private const DiffractionDielectricCoatingExtra *diffraction_dielectric_coating(
    ccl_private const DiffractionDielectricBsdf *bsdf)
{
  return (bsdf->disabled_lobes & DIFFRACTION_DIELECTRIC_COATING_EXTRA)?
      (ccl_private const DiffractionDielectricCoatingExtra *)bsdf->extra:nullptr;
}
static_assert(sizeof(ShaderClosure) >= sizeof(DiffractionDielectricBsdf));

/* Generalized Schlick with wavelength-dependent IOR and optional thin film. The reference F0
 * is the ordinary interface value; generalized_f0 retains native Specular Tint.
 * Proposal masses use the spectral mean of these powers, so tint does not
 * introduce unsupported directions or change the sample/evaluate measure. */
ccl_device_inline float diffraction_dielectric_generalized_budget(
    ccl_private const DiffractionDielectricGeneralizedExtra *e)
{
  return e->base.param.facet.incident_ior==e->base.param.facet.transmitted_ior ?
             average(e->generalized_f0):1.0f;
}

ccl_device_inline Spectrum diffraction_dielectric_generalized_fresnel(
    ccl_private const DiffractionDielectricGeneralizedExtra *e, const float physical)
{
  if(e->film_thickness_over_wavelength>0.0f) {
    /* Native Principled film tint rule, bounded for passive order powers.
     * The physical near-matched branch is handled by coated Glass setup. */
    if(e->generalized_reference_f0<=1e-5f)return make_spectrum(physical);
    const float s=saturatef(inverse_lerp(1.0f,e->generalized_reference_f0,physical));
    return saturate(physical*mix(one_spectrum(),e->generalized_f0/e->generalized_reference_f0,s));
  }
  const float s=saturatef(inverse_lerp(e->generalized_reference_f0,1.0f,physical));
  return mix(e->generalized_f0,one_spectrum(),s);
}

ccl_device_inline float diffraction_dielectric_generalized_interface(
    ccl_private const DiffractionDielectricGeneralizedExtra *e,const float cosine,
    const bool reverse)
{
  const ccl_private DiffractionDielectricFacet *p=&e->base.param.facet;
  const float ni=reverse?p->transmitted_ior:p->incident_ior;
  const float no=reverse?p->incident_ior:p->transmitted_ior;
  return e->film_thickness_over_wavelength>0.0f ?
      diffraction_thin_film_reflectance(cosine,ni,no,e->film_ior,e->film_thickness_over_wavelength) :
      fresnel_dielectric(cosine,no/ni,nullptr);
}

ccl_device_inline Spectrum diffraction_dielectric_generalized_nonzero(
    ccl_private const DiffractionDielectricGeneralizedExtra *e, const int order,
    const bool transmission, const float ci, const float co)
{
  ccl_private const DiffractionDielectricFacet *p=&e->base.param.facet;
  if (!(ci>0) || (transmission ? !(co<0) : !(co>0))) return zero_spectrum();
  const float eta=p->incident_ior/p->transmitted_ior;
  const Spectrum Fi=diffraction_dielectric_generalized_fresnel(
      e,diffraction_dielectric_generalized_interface(e,ci,false));
  if (transmission) {
    Spectrum power;
    if(order==0) {
      const float rs=(eta*ci+co)/(eta*ci-co),rp=(ci+eta*co)/(ci-eta*co);
      const float physical=e->film_thickness_over_wavelength>0.0f ?
          diffraction_thin_film_pair_reflectance(ci,-co,p->incident_ior,p->transmitted_ior,
                                                e->film_ior,e->film_thickness_over_wavelength) :
          .5f*(rs*rs+rp*rp);
      power=one_spectrum()-diffraction_dielectric_generalized_fresnel(e,physical);
    }
    else {
      const Spectrum Fo=diffraction_dielectric_generalized_fresnel(
          e,diffraction_dielectric_generalized_interface(e,-co,true));
      power=min(one_spectrum()-Fi,one_spectrum()-Fo);
    }
    return power*diffraction_binary_power(order,p->transmission_phase,p->duty);
  }
  const Spectrum Fo=diffraction_dielectric_generalized_fresnel(
      e,diffraction_dielectric_generalized_interface(e,co,false));
  return min(Fi,Fo)*diffraction_binary_power(
      order,M_2PI_F*p->height_over_wavelength*(ci+co),p->duty);
}

ccl_device_inline Spectrum diffraction_dielectric_generalized_power(
    ccl_private const DiffractionDielectricGeneralizedExtra *e, const int order,
    const bool transmission,const float3 wi,const float3 wo,const float3 h)
{
  if(transmission || order!=0)
    return diffraction_dielectric_generalized_nonzero(e,order,transmission,dot(wi,h),dot(wo,h));
  ccl_private const DiffractionDielectricFacet *p=&e->base.param.facet;
  Spectrum other=zero_spectrum();
  for(int side=0;side<2;++side) {
    const bool trans=side!=0;const float eta=p->incident_ior/p->transmitted_ior;
    const int bound=diffraction_dielectric_facet_active_order_bound(p,trans);
    for(int m=-bound;m<=bound;++m) {
      if(!trans && m==0)continue;
      float3 candidate;const float delta=m*p->wavelength_over_pitch;
      const bool valid=trans?
          diffraction_facet_transmit(wi,h,make_float3(1,0,0),eta,delta*eta,&candidate):
          diffraction_facet_reflect(wi,h,make_float3(1,0,0),delta,&candidate);
      if(valid)other+=diffraction_dielectric_generalized_nonzero(e,m,trans,dot(wi,h),dot(candidate,h));
    }
  }
  return one_spectrum()-other;
}

template<MicrofacetType m_type = GGX>
ccl_device_inline Spectrum diffraction_dielectric_generalized_eval(
    ccl_private const DiffractionDielectricGeneralizedExtra *e, const float3 wi, const float3 wo,
    ccl_private float *pdf)
{
  ccl_private const DiffractionRoughDielectric *p=&e->base.param;
  *pdf=0.0f;
  if (!(wi.z>0) || wo.z==0 || roughness_is_almost_specular(p->alpha_x,p->alpha_y)) return zero_spectrum();
  const bool trans=wo.z<0;
  const float eta=p->facet.incident_ior/p->facet.transmitted_ior;
  const float3 axis=make_float3(1,0,0);
  const int bound=diffraction_dielectric_facet_active_order_bound(&p->facet,trans);
  Spectrum density=zero_spectrum();
  for (int m=-bound; m<=bound; ++m) {
    if (trans && m==0 && eta==1.0f) continue;
    const float delta=m*p->facet.wavelength_over_pitch*(trans?eta:1.0f);
    float3 normals[2];
    int count;
    if (trans) {
      DiffractionTransmissionRoot roots[2];
      const float ni=p->facet.incident_ior,no=p->facet.transmitted_ior;
      count=diffraction_transmission_half_vectors_momentum(
          wi,wo,axis,diffraction_symmetric_momentum(ni,wi,no,wo),m*p->facet.wavelength_over_pitch*ni,roots);
      for (int r=0;r<count;++r) normals[r]=roots[r].h;
    }
    else {
      DiffractionReflectionRoot roots[2];
      count=diffraction_reflection_half_vectors(wi,wo,axis,delta,roots);
      for (int r=0;r<count;++r) normals[r]=roots[r].h;
    }
    for (int r=0;r<count;++r) {
      const float3 h=normals[r];
      if (!(h.z>0)) continue;
      const Spectrum power=diffraction_dielectric_generalized_power(e,m,trans,wi,wo,h);
      const float jacobian=trans?
          diffraction_transmission_jacobian_indices(wi,wo,h,axis,p->facet.incident_ior,
              p->facet.transmitted_ior,m*p->facet.wavelength_over_pitch*p->facet.incident_ior):
          diffraction_reflection_jacobian(wi,wo,h,axis,delta);
      const Spectrum contribution=power*bsdf_aniso_D<m_type>(p->alpha_x,p->alpha_y,h)*dot(wi,h)*jacobian/wi.z;
      density+=contribution;

    }
  }
  const float li=diffraction_dielectric_lambda<m_type>(p->alpha_x,p->alpha_y,wi);
  const float lo=diffraction_dielectric_lambda<m_type>(p->alpha_x,p->alpha_y,wo);
  *pdf=average(density)/(1+li);
  return density/(1+li+lo);
}


template<MicrofacetType m_type>
ccl_device_inline bool diffraction_dielectric_generalized_sample_direction(
    ccl_private const DiffractionDielectricGeneralizedExtra *e,const float3 wi,
    const float3 random,ccl_private float3 *wo,ccl_private float *mass,ccl_private bool *singular)
{
  ccl_private const DiffractionRoughDielectric *p=&e->base.param;*mass=0;*wo=zero_float3();
  if(!(wi.z>0))return false;
  *singular=roughness_is_almost_specular(p->alpha_x,p->alpha_y);
  const float3 h=*singular?make_float3(0,0,1):
      (m_type==BECKMANN?microfacet_beckmann_sample_vndf(wi,p->alpha_x,p->alpha_y,make_float2(random)):
                       microfacet_ggx_sample_vndf(wi,p->alpha_x,p->alpha_y,make_float2(random)));
  const bool matched=p->facet.incident_ior==p->facet.transmitted_ior;
  float u=random.z*diffraction_dielectric_generalized_budget(e);
  Spectrum other=matched?one_spectrum()-e->generalized_f0:zero_spectrum();
  for(int side=0;side<2;++side) {
    const bool trans=side!=0;
    if(trans && matched)continue;
    const float eta=p->facet.incident_ior/p->facet.transmitted_ior;
    const int bound=diffraction_dielectric_facet_active_order_bound(&p->facet,trans);
    for(int m=-bound;m<=bound;++m) {
      if(!trans && m==0)continue;
      float3 candidate;const float delta=m*p->facet.wavelength_over_pitch;
      const bool valid=trans?
          diffraction_facet_transmit(wi,h,make_float3(1,0,0),eta,delta*eta,&candidate):
          diffraction_facet_reflect(wi,h,make_float3(1,0,0),delta,&candidate);
      if(!valid)continue;
      const Spectrum power=diffraction_dielectric_generalized_nonzero(e,m,trans,dot(wi,h),dot(candidate,h));
      const float probability=average(power);
      if(u<probability) {
        *wo=candidate;*mass=probability;
        return (candidate.z<0)==trans && candidate.z!=0;
      }
      u-=probability;other+=power;
    }
  }
  *wo=2*dot(wi,h)*h-wi;*mass=average(one_spectrum()-other);
  return wo->z>0 && *mass>0;
}

struct DiffractionCoatedAtomExtra {
  DiffractionRoughDielectric param;
  float film_ior,film_thickness_over_wavelength;
  int samples;
  MicrofacetType distribution;
};
struct DiffractionCoatedAtomBsdf {
  SHADER_CLOSURE_BASE;
  packed_float3 T;
  int polarizer; /* Uses previous pointer-alignment padding; OFF size unchanged. */
  ccl_private DiffractionCoatedAtomExtra *extra;
};
struct DiffractionStraightPolarizerBsdf {
  SHADER_CLOSURE_BASE;
  packed_float3 polarizer_axis;
  int polarizer;
};
static_assert(sizeof(DiffractionStraightPolarizerBsdf)<=sizeof(ShaderClosure));
/* Suffixes occupy already allocated extra-block padding. Original allocations
 * and layouts are unchanged when the checkbox is OFF. */
template<typename Base> struct DiffractionPolarizerExtra {
  Base base;
  packed_float3 axis;
};
#define DIFFRACTION_POLARIZER_PADDING_CHECK(T) \
  static_assert(sizeof(DiffractionPolarizerExtra<T>) <= \
      ((sizeof(T)+sizeof(ShaderClosure)-1)/sizeof(ShaderClosure))*sizeof(ShaderClosure))
DIFFRACTION_POLARIZER_PADDING_CHECK(DiffractionDielectricTintExtra);
DIFFRACTION_POLARIZER_PADDING_CHECK(DiffractionDielectricCoatingExtra);
DIFFRACTION_POLARIZER_PADDING_CHECK(DiffractionDielectricTwoSidedExtra);
DIFFRACTION_POLARIZER_PADDING_CHECK(DiffractionCoatedAtomExtra);
#undef DIFFRACTION_POLARIZER_PADDING_CHECK
static_assert(sizeof(DiffractionCoatedAtomBsdf)<=sizeof(ShaderClosure));

ccl_device_inline float bsdf_diffraction_coated_atom_mass(
    ccl_private const ShaderClosure *sc,const float3 wi)
{
  const ccl_private DiffractionCoatedAtomBsdf *bsdf=(const ccl_private DiffractionCoatedAtomBsdf *)sc;
  const ccl_private DiffractionCoatedAtomExtra *extra=bsdf->extra;
  float3 X,Y;make_orthonormals_safe_tangent(bsdf->N,bsdf->T,&X,&Y);
  const float3 local=make_float3(dot(wi,X),dot(wi,Y),dot(wi,bsdf->N));
  float mass=0;
  if(extra->distribution==BECKMANN)
    diffraction_dielectric_coated_straight_mass_quadrature<BECKMANN>(
        &extra->param,local,extra->film_ior,extra->film_thickness_over_wavelength,extra->samples,&mass);
  else
    diffraction_dielectric_coated_straight_mass_quadrature<GGX>(
        &extra->param,local,extra->film_ior,extra->film_thickness_over_wavelength,extra->samples,&mass);
  return mass;
}

/* Keep the direction-dependent physical atom out of the stored color weight.
 * The auxiliary parameters belong to this ShaderData, just like Fresnel data.
 * Selection weight is an importance estimate; evaluation uses the actual wi. */
template<MicrofacetType m_type=GGX>
ccl_device_inline bool bsdf_diffraction_coated_atom_setup(
    ccl_private ShaderData *sd,const Spectrum weight,const float3 normal,const float3 tangent,
    ccl_private const DiffractionRoughDielectric *param,const float film_ior,
    const float film_thickness_over_wavelength,const int samples)
{
  if (!(param->facet.incident_ior>0 && param->facet.incident_ior==param->facet.transmitted_ior) ||
      !isfinite_safe(param->facet.incident_ior) || !isfinite_safe(param->facet.transmission_phase) ||
      !(param->facet.duty>=0 && param->facet.duty<=1) ||
      !(param->alpha_x>=0 && param->alpha_x<=1 && param->alpha_y>=0 && param->alpha_y<=1))return false;
  const float3 N=safe_normalize(normal),T=safe_normalize(tangent-dot(tangent,N)*N);
  if(is_zero(N)||is_zero(T)||!isfinite_safe(N)||!isfinite_safe(T))return false;
  float3 X,Y;make_orthonormals_safe_tangent(N,T,&X,&Y);
  float mass;
  if(!diffraction_dielectric_coated_straight_mass_quadrature<m_type>(
      param,make_float3(dot(sd->wi,X),dot(sd->wi,Y),dot(sd->wi,N)),
      film_ior,film_thickness_over_wavelength,samples,&mass))return false;
  const int extra_slots=(sizeof(DiffractionCoatedAtomExtra)+sizeof(ShaderClosure)-1)/sizeof(ShaderClosure);
  if(sd->num_closure_left<1+extra_slots)return false;
  ccl_private DiffractionCoatedAtomBsdf *bsdf=(ccl_private DiffractionCoatedAtomBsdf *)
      bsdf_alloc(sd,sizeof(DiffractionCoatedAtomBsdf),weight);
  if(!bsdf)return false;
  ccl_private DiffractionCoatedAtomExtra *extra=(ccl_private DiffractionCoatedAtomExtra *)
      closure_alloc_extra(sd,sizeof(DiffractionCoatedAtomExtra));
  kernel_assert(extra!=nullptr);
  extra->param=*param;extra->film_ior=film_ior;
  extra->film_thickness_over_wavelength=film_thickness_over_wavelength;
  extra->samples=samples;extra->distribution=m_type;
  bsdf->type=CLOSURE_BSDF_DIFFRACTION_COATED_STRAIGHT_ID;
  bsdf->N=N;bsdf->T=T;bsdf->extra=extra;bsdf->polarizer=0;
  bsdf->sample_weight*=max(.05f,mass);
  sd->runtime_flag|=SR_BSDF|SR_BSDF_HAS_DISPERSION|SR_BSDF_HAS_TRANSMISSION;
  return true;
}

ccl_device_inline bool bsdf_diffraction_dielectric_has_transmission(
    ccl_private const ShaderClosure *sc)
{
  if(sc->type==CLOSURE_BSDF_DIFFRACTION_STRAIGHT_ID ||
     sc->type==CLOSURE_BSDF_DIFFRACTION_COATED_STRAIGHT_ID) return true;
  if(sc->type!=CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID &&
     sc->type!=CLOSURE_BSDF_DIFFRACTION_BECKMANN_ID) return false;
  const ccl_private DiffractionDielectricBsdf *bsdf=(const ccl_private DiffractionDielectricBsdf *)sc;
  return !(bsdf->disabled_lobes & LABEL_TRANSMIT);
}

/* Closure filters act on transport lobes, not the broad glossy category. */
ccl_device_inline bool bsdf_diffraction_dielectric_filter(
    ccl_private ShaderClosure *sc,const bool reflection,const bool transmission)
{
  bool remove=false;
  if(sc->type==CLOSURE_BSDF_DIFFRACTION_STRAIGHT_ID ||
     sc->type==CLOSURE_BSDF_DIFFRACTION_COATED_STRAIGHT_ID) remove=transmission;
  else {
    ccl_private DiffractionDielectricBsdf *bsdf=(ccl_private DiffractionDielectricBsdf *)sc;
    if(reflection)bsdf->disabled_lobes|=LABEL_REFLECT;
    if(transmission)bsdf->disabled_lobes|=LABEL_TRANSMIT;
    remove=(bsdf->disabled_lobes&(LABEL_REFLECT|LABEL_TRANSMIT))==(LABEL_REFLECT|LABEL_TRANSMIT);
  }
  if(remove) {sc->type=CLOSURE_NONE_ID;sc->sample_weight=0;}
  return !remove;
}

ccl_device_inline void bsdf_diffraction_dielectric_frame(
    ccl_private const DiffractionDielectricBsdf *bsdf,
    ccl_private float3 *X, ccl_private float3 *Y)
{
  make_orthonormals_safe_tangent(bsdf->N,bsdf->T,X,Y);
}

/* Every spectral channel reconstructs its own reciprocal return from q/Q.
 * The direction proposal mixes their side probabilities; it never substitutes
 * mean RGB Fresnel into transport or blends completed return matrices. */
ccl_device_inline float4 diffraction_dielectric_ms_cache(
    KernelGlobals kg, ccl_private const DiffractionDielectricTwoSidedExtra *extra,
    const int side, const float3 local_direction, const int channel)
{
  return diffraction_two_sided_cache_lookup(
      kg, extra->cache_handle, extra->wavelength_nm, side, local_direction,
      GET_SPECTRUM_CHANNEL(extra->generalized_f0, channel),
      extra->ior_d, extra->inv_abbe, extra->inside_ior);
}

ccl_device_inline float2 diffraction_dielectric_ms_probabilities(
    KernelGlobals kg, ccl_private const DiffractionDielectricBsdf *bsdf, const float3 I)
{
  const ccl_private DiffractionDielectricTwoSidedExtra *extra =
      (const ccl_private DiffractionDielectricTwoSidedExtra *)bsdf->extra;
  float2 probability = zero_float2();
  FOREACH_SPECTRUM_CHANNEL(channel) {
    const float4 cache = diffraction_dielectric_ms_cache(kg, extra, extra->incoming_side, I, channel);
    DiffractionTwoSidedReturn returned;
    if (!diffraction_two_sided_return(cache.y, cache.z, cache.w, &returned)) continue;
    probability.x += diffraction_two_sided_side_probability(
        returned, extra->incoming_side, extra->incoming_side);
    probability.y += diffraction_two_sided_side_probability(
        returned, extra->incoming_side, 1 - extra->incoming_side);
  }
  probability *= 1.0f / SPECTRUM_CHANNELS;
  if (bsdf->disabled_lobes & LABEL_REFLECT) probability.x = 0.0f;
  if (bsdf->disabled_lobes & LABEL_TRANSMIT) probability.y = 0.0f;
  return probability;
}

ccl_device_inline Spectrum diffraction_dielectric_ms_albedo(
    KernelGlobals kg, ccl_private const DiffractionDielectricBsdf *bsdf, const float3 I,
    const bool reflection, const bool transmission)
{
  if (!(I.z > 0.0f)) return zero_spectrum();
  const ccl_private DiffractionDielectricTwoSidedExtra *extra =
      (const ccl_private DiffractionDielectricTwoSidedExtra *)bsdf->extra;
  Spectrum result = zero_spectrum();
  FOREACH_SPECTRUM_CHANNEL(channel) {
    const float4 cache = diffraction_dielectric_ms_cache(kg, extra, extra->incoming_side, I, channel);
    DiffractionTwoSidedReturn returned;
    if (!diffraction_two_sided_return(cache.y, cache.z, cache.w, &returned)) continue;
    if (reflection && !(bsdf->disabled_lobes & LABEL_REFLECT)) {
      GET_SPECTRUM_CHANNEL(result, channel) += GET_SPECTRUM_CHANNEL(extra->base.reflection, channel) *
          cache.x * diffraction_two_sided_side_probability(
              returned, extra->incoming_side, extra->incoming_side);
    }
    if (transmission && !(bsdf->disabled_lobes & LABEL_TRANSMIT)) {
      GET_SPECTRUM_CHANNEL(result, channel) += GET_SPECTRUM_CHANNEL(extra->base.transmission, channel) *
          cache.x * diffraction_two_sided_side_probability(
              returned, extra->incoming_side, 1 - extra->incoming_side);
    }
  }
  return result;
}

/* A separate continuous return closure complements the single-facet lobe.
 * Reflection and transmission weights stay outside the reciprocal neutral
 * kernel; scene validation permits only equal neutral weights for this mode. */
ccl_device_inline Spectrum bsdf_diffraction_dielectric_ms_eval(
    KernelGlobals kg,
    ccl_private const DiffractionDielectricBsdf *bsdf,
    const float3 wi,
    const float3 wo,
    ccl_private float *pdf)
{
  *pdf = 0.0f;
  const ccl_private DiffractionDielectricTwoSidedExtra *extra =
      (const ccl_private DiffractionDielectricTwoSidedExtra *)bsdf->extra;
  float3 X, Y;
  bsdf_diffraction_dielectric_frame(bsdf, &X, &Y);
  const float3 I = make_float3(dot(wi, X), dot(wi, Y), dot(wi, bsdf->N));
  const float3 O = make_float3(dot(wo, X), dot(wo, Y), dot(wo, bsdf->N));
  if (!(I.z > 0.0f && fabsf(O.z) > 0.0f)) return zero_spectrum();
  const bool transmission = O.z < 0.0f;
  if (bsdf->disabled_lobes & (transmission ? LABEL_TRANSMIT : LABEL_REFLECT)) {
    return zero_spectrum();
  }
  const int incoming_side = extra->incoming_side;
  const int outgoing_side = transmission ? 1 - incoming_side : incoming_side;
  const float3 outgoing_lookup = transmission ? make_float3(O.x, -O.y, -O.z) : O;
  const float2 probability = diffraction_dielectric_ms_probabilities(kg, bsdf, I);
  const float allowed_probability = probability.x + probability.y;
  if (!(allowed_probability > 0.0f)) return zero_spectrum();
  const float cosine = fabsf(O.z);
  *pdf = (transmission ? probability.y : probability.x) /
         allowed_probability * cosine * M_1_PI_F;
  const float outgoing_ior = outgoing_side == 0 ? 1.0f : extra->inside_ior;
  Spectrum value = zero_spectrum();
  FOREACH_SPECTRUM_CHANNEL(channel) {
    const float4 incoming = diffraction_dielectric_ms_cache(kg, extra, incoming_side, I, channel);
    const float4 outgoing = diffraction_dielectric_ms_cache(
        kg, extra, outgoing_side, outgoing_lookup, channel);
    DiffractionTwoSidedReturn returned;
    if (!diffraction_two_sided_return(incoming.y, incoming.z, incoming.w, &returned)) continue;
    GET_SPECTRUM_CHANNEL(value, channel) = diffraction_two_sided_eval(
        returned, incoming_side, outgoing_side, incoming.x, outgoing.x, outgoing_ior, cosine);
  }
  return value * (transmission ? extra->base.transmission : extra->base.reflection);
}

template<MicrofacetType m_type = GGX>
ccl_device_inline Spectrum bsdf_diffraction_dielectric_eval(
    KernelGlobals kg, ccl_private const ShaderClosure *sc, const float3 wi, const float3 wo,
    ccl_private float *pdf)
{
  const ccl_private DiffractionDielectricBsdf *bsdf=(const ccl_private DiffractionDielectricBsdf *)sc;
  *pdf=0;
  if (bsdf->disabled_lobes & DIFFRACTION_DIELECTRIC_TWO_SIDED_MS) {
    return bsdf_diffraction_dielectric_ms_eval(kg, bsdf, wi, wo, pdf);
  }
  const int side=dot(bsdf->N,wo)<0?LABEL_TRANSMIT:LABEL_REFLECT;
  if (bsdf->disabled_lobes & side) return zero_spectrum();
  float3 X,Y; bsdf_diffraction_dielectric_frame(bsdf,&X,&Y);
  const ccl_private DiffractionDielectricCoatingExtra *coating=diffraction_dielectric_coating(bsdf);
  if(bsdf->disabled_lobes & DIFFRACTION_DIELECTRIC_GENERALIZED) {
    const ccl_private DiffractionDielectricGeneralizedExtra *e=
        (const ccl_private DiffractionDielectricGeneralizedExtra *)bsdf->extra;
    const float budget=diffraction_dielectric_generalized_budget(e);
    if(!(budget>0))return zero_spectrum();
    const Spectrum value=diffraction_dielectric_generalized_eval<m_type>(e,
        make_float3(dot(wi,X),dot(wi,Y),dot(wi,bsdf->N)),
        make_float3(dot(wo,X),dot(wo,Y),dot(wo,bsdf->N)),pdf);
    *pdf/=budget;
    return value*diffraction_dielectric_tint(bsdf,side==LABEL_TRANSMIT);
  }
  if(coating) {
    const ccl_private DiffractionRoughDielectric *p=diffraction_dielectric_param(bsdf);
    const float value=diffraction_dielectric_eval<m_type>(p,
        make_float3(dot(wi,X),dot(wi,Y),dot(wi,bsdf->N)),
        make_float3(dot(wo,X),dot(wo,Y),dot(wo,bsdf->N)),pdf,
        coating->film_ior,coating->film_thickness_over_wavelength,
        p->facet.incident_ior==p->facet.transmitted_ior);
    return make_spectrum(value)*diffraction_dielectric_tint(bsdf,side==LABEL_TRANSMIT);
  }
  const float budget=1-diffraction_dielectric_straight_mass(diffraction_dielectric_param(bsdf));
  *pdf=0;
  if (!(budget>0)) return zero_spectrum();
  const float value=diffraction_dielectric_eval<m_type>(diffraction_dielectric_param(bsdf),
      make_float3(dot(wi,X),dot(wi,Y),dot(wi,bsdf->N)),
      make_float3(dot(wo,X),dot(wo,Y),dot(wo,bsdf->N)),pdf,1.0f,0.0f,false,
      (bsdf->disabled_lobes & DIFFRACTION_DIELECTRIC_PURE_REFRACTION)!=0);
  *pdf/=budget;
  return make_spectrum(value/budget)*diffraction_dielectric_tint(bsdf,side==LABEL_TRANSMIT);
}

template<MicrofacetType m_type = GGX>
ccl_device_inline int bsdf_diffraction_dielectric_sample(
    KernelGlobals kg, ccl_private const ShaderClosure *sc, const float3 Ng, const float3 wi,
    const float3 random, ccl_private Spectrum *eval, ccl_private float3 *wo,
    ccl_private float *pdf, ccl_private float2 *roughness, ccl_private float *eta)
{
  const ccl_private DiffractionDielectricBsdf *bsdf=(const ccl_private DiffractionDielectricBsdf *)sc;
  *eval=zero_spectrum(); *pdf=0; *wo=zero_float3(); *eta=1;
  *roughness=make_float2(diffraction_dielectric_param(bsdf)->alpha_x,diffraction_dielectric_param(bsdf)->alpha_y);
  if (bsdf->disabled_lobes & DIFFRACTION_DIELECTRIC_TWO_SIDED_MS) {
    const ccl_private DiffractionDielectricTwoSidedExtra *extra =
        (const ccl_private DiffractionDielectricTwoSidedExtra *)bsdf->extra;
    float3 X, Y;
    bsdf_diffraction_dielectric_frame(bsdf, &X, &Y);
    const float3 I = make_float3(dot(wi, X), dot(wi, Y), dot(wi, bsdf->N));
    if (!(I.z > 0.0f)) return LABEL_NONE;
    const float2 probability = diffraction_dielectric_ms_probabilities(kg, bsdf, I);
    const float allowed_probability = probability.x + probability.y;
    if (!(allowed_probability > 0.0f)) return LABEL_NONE;
    const bool transmission = random.z * allowed_probability >= probability.x;
    const float radius = sqrtf(random.x);
    const float angle = M_2PI_F * random.y;
    const float3 O = make_float3(radius * cosf(angle),
                                 radius * sinf(angle),
                                 (transmission ? -1.0f : 1.0f) * safe_sqrtf(1.0f - random.x));
    *wo = O.x * X + O.y * Y + O.z * bsdf->N;
    if ((dot(Ng, *wo) < 0.0f) != transmission) return LABEL_NONE;
    *eval = bsdf_diffraction_dielectric_ms_eval(kg, bsdf, wi, *wo, pdf);
    if (!(*pdf > 0.0f)) return LABEL_NONE;
    *eta = transmission ?
               (extra->incoming_side == 0 ? extra->inside_ior : 1.0f / extra->inside_ior) :
               1.0f;
    return (transmission ? LABEL_TRANSMIT : LABEL_REFLECT) | LABEL_GLOSSY;
  }
  float3 X,Y; bsdf_diffraction_dielectric_frame(bsdf,&X,&Y);
  const float3 local_wi=make_float3(dot(wi,X),dot(wi,Y),dot(wi,bsdf->N));
  float3 local_wo;float value;bool singular;
  const ccl_private DiffractionRoughDielectric *p=diffraction_dielectric_param(bsdf);
  const ccl_private DiffractionDielectricCoatingExtra *coating=diffraction_dielectric_coating(bsdf);
  const float film_ior=coating?coating->film_ior:1;
  const float film_thickness=coating?coating->film_thickness_over_wavelength:0;
  const bool generalized=(bsdf->disabled_lobes & DIFFRACTION_DIELECTRIC_GENERALIZED)!=0;
  const float budget=generalized?diffraction_dielectric_generalized_budget(
      (const ccl_private DiffractionDielectricGeneralizedExtra *)bsdf->extra):coating?
      diffraction_dielectric_facet_nonstraight_budget(&p->facet,local_wi.z,film_ior,film_thickness):
      1-diffraction_dielectric_straight_mass(p);
  if (!coating && !(budget>0)) return LABEL_NONE;
  const bool exclude=coating?p->facet.incident_ior==p->facet.transmitted_ior:budget<1;
  if (!(generalized ? diffraction_dielectric_generalized_sample_direction<m_type>(
                          (const ccl_private DiffractionDielectricGeneralizedExtra *)bsdf->extra,local_wi,random,&local_wo,&value,&singular) :
          diffraction_dielectric_sample_direction<m_type>(p,local_wi,random,&local_wo,&value,&singular,
              exclude,film_ior,film_thickness,
              (bsdf->disabled_lobes & DIFFRACTION_DIELECTRIC_PURE_REFRACTION)!=0)))
    return LABEL_NONE;
  *wo=local_wo.x*X+local_wo.y*Y+local_wo.z*bsdf->N;
  const bool transmission=local_wo.z<0;
  if (bsdf->disabled_lobes & (transmission?LABEL_TRANSMIT:LABEL_REFLECT)) return LABEL_NONE;
  if ((dot(Ng,*wo)<0)!=transmission) { *pdf=0; return LABEL_NONE; }
  if (singular) {
    *pdf=value/budget*1e6f;
    *eval=make_spectrum(coating?value*1e6f:*pdf)*diffraction_dielectric_tint(bsdf,transmission);
    if(generalized) {
      const float ratio=transmission?p->facet.transmitted_ior/p->facet.incident_ior:1;
      const int order=int(roundf((local_wi.x+ratio*local_wo.x)/p->facet.wavelength_over_pitch));
      *eval=1e6f*diffraction_dielectric_generalized_power((const ccl_private DiffractionDielectricGeneralizedExtra *)bsdf->extra,order,transmission,
          local_wi,local_wo,make_float3(0,0,1))*diffraction_dielectric_tint(bsdf,transmission);
    }
    *roughness=zero_float2();
  }
  else {
    *eval=bsdf_diffraction_dielectric_eval<m_type>(kg,sc,wi,*wo,pdf);
    if (!(*pdf>0)) return LABEL_NONE;
  }
  *eta=transmission?diffraction_dielectric_param(bsdf)->facet.transmitted_ior/diffraction_dielectric_param(bsdf)->facet.incident_ior:1;
  return (transmission?LABEL_TRANSMIT:LABEL_REFLECT)|(singular?LABEL_SINGULAR:LABEL_GLOSSY);
}

/* Atomic probability for mixture evaluation. Rough matched-index transmission
 * contains one straight-through atom; all its other events are continuous. */
ccl_device_inline float bsdf_diffraction_dielectric_delta_mass(
    ccl_private const ShaderClosure *sc, const float3 wi, const float3 wo,
    ccl_private float *value_mass=nullptr)
{
  if(value_mass)*value_mass=0;
  const ccl_private DiffractionDielectricBsdf *bsdf=(const ccl_private DiffractionDielectricBsdf *)sc;
  const ccl_private DiffractionDielectricFacet *p=&diffraction_dielectric_param(bsdf)->facet;
  if (!roughness_is_almost_specular(diffraction_dielectric_param(bsdf)->alpha_x,diffraction_dielectric_param(bsdf)->alpha_y)) {
    return 0; /* Matched-index atom lives in a separate closure. */
  }
  float3 X,Y;bsdf_diffraction_dielectric_frame(bsdf,&X,&Y);
  const float3 I=make_float3(dot(wi,X),dot(wi,Y),dot(wi,bsdf->N));
  const float3 O=make_float3(dot(wo,X),dot(wo,Y),dot(wo,bsdf->N));
  if (!(I.z>0) || O.z==0 || !isfinite_safe(I) || !isfinite_safe(O) ||
      fabsf(len_squared(I)-1)>8e-6f || fabsf(len_squared(O)-1)>8e-6f) return 0;
  const bool trans=O.z<0;
  if (bsdf->disabled_lobes & (trans?LABEL_TRANSMIT:LABEL_REFLECT)) return 0;
  const float ratio=trans?p->transmitted_ior/p->incident_ior:1;
  const float x=I.x+ratio*O.x,y=I.y+ratio*O.y;
  const float m=roundf(x/p->wavelength_over_pitch);
  if (fabsf(m)>diffraction_dielectric_facet_active_order_bound(p,trans) ||
      sqr(x-m*p->wavelength_over_pitch)+sqr(y)>sqr(4e-6f*max(1.0f,ratio))) return 0;
  const ccl_private DiffractionDielectricCoatingExtra *coating=diffraction_dielectric_coating(bsdf);
  const float film_ior=coating?coating->film_ior:1;
  const float film_thickness=coating?coating->film_thickness_over_wavelength:0;
  const float budget=(bsdf->disabled_lobes & DIFFRACTION_DIELECTRIC_GENERALIZED)?
      diffraction_dielectric_generalized_budget((const ccl_private DiffractionDielectricGeneralizedExtra *)bsdf->extra):coating?
      diffraction_dielectric_facet_nonstraight_budget(p,I.z,film_ior,film_thickness):
      1-diffraction_dielectric_straight_mass(diffraction_dielectric_param(bsdf));
  if (!(budget>0) || (trans && m==0 && p->incident_ior==p->transmitted_ior)) return 0;
  const float mass=(bsdf->disabled_lobes & DIFFRACTION_DIELECTRIC_GENERALIZED) ?
      average(diffraction_dielectric_generalized_power((const ccl_private DiffractionDielectricGeneralizedExtra *)bsdf->extra,int(m),trans,I,O,make_float3(0,0,1))) :
      !trans && m==0 ?
      diffraction_dielectric_facet_residual(p,I,make_float3(0,0,1),make_float3(1,0,0),film_ior,film_thickness):
      diffraction_dielectric_facet_power(p,int(m),trans,I.z,O.z,film_ior,film_thickness,-1.0f,
          (bsdf->disabled_lobes & DIFFRACTION_DIELECTRIC_PURE_REFRACTION)!=0);
  if(value_mass)*value_mass=coating?mass:mass/budget;
  return mass/budget;
}


/* Called only after the delta-direction support test succeeds. */
ccl_device_inline Spectrum bsdf_diffraction_dielectric_generalized_delta_value(
    ccl_private const ShaderClosure *sc,const float3 wi,const float3 wo)
{
  ccl_private const DiffractionDielectricBsdf *b=(const ccl_private DiffractionDielectricBsdf *)sc;
  float3 X,Y;bsdf_diffraction_dielectric_frame(b,&X,&Y);
  const float3 I=make_float3(dot(wi,X),dot(wi,Y),dot(wi,b->N));
  const float3 O=make_float3(dot(wo,X),dot(wo,Y),dot(wo,b->N));
  ccl_private const DiffractionDielectricFacet *p=&b->extra->param.facet;const bool trans=O.z<0;
  const float ratio=trans?p->transmitted_ior/p->incident_ior:1;
  const int order=int(roundf((I.x+ratio*O.x)/p->wavelength_over_pitch));
  return diffraction_dielectric_generalized_power((const ccl_private DiffractionDielectricGeneralizedExtra *)b->extra,order,trans,I,O,make_float3(0,0,1))*
         diffraction_dielectric_tint(b,trans);
}

ccl_device_inline bool diffraction_dielectric_parameters_valid(
    ccl_private const DiffractionRoughDielectric *param)
{
  const ccl_private DiffractionDielectricFacet *p=&param->facet;
  if (!(p->incident_ior>0 && p->transmitted_ior>0 && p->wavelength_over_pitch>0) ||
      !isfinite_safe(p->incident_ior) || !isfinite_safe(p->transmitted_ior) ||
      !isfinite_safe(p->wavelength_over_pitch) || !isfinite_safe(p->height_over_wavelength) ||
      !isfinite_safe(p->transmission_phase) || !(p->duty>=0 && p->duty<=1) ||
      !(param->alpha_x>=0 && param->alpha_x<=1 && param->alpha_y>=0 && param->alpha_y<=1) ||
      diffraction_dielectric_facet_order_bound(p,false)==0 ||
      diffraction_dielectric_facet_order_bound(p,true)==0) return false;
  return true;
}

/* At most two ordinary closure slots; shader compilation must reserve both.
 * The weights perform the mixture split, while each conditional closure keeps
 * its own normalized PDF. This avoids mixed measures inside guiding RIS. */
template<MicrofacetType m_type = GGX>
ccl_device_inline_outline_metal bool bsdf_diffraction_dielectric_setup(
    ccl_private ShaderData *sd, const Spectrum weight, const float3 normal,
    const float3 tangent, ccl_private const DiffractionRoughDielectric *param,
    const int disabled_lobes = 0)
{
  if ((disabled_lobes & (LABEL_REFLECT|LABEL_TRANSMIT)) == (LABEL_REFLECT|LABEL_TRANSMIT)) return false;
  if(!diffraction_dielectric_parameters_valid(param))return false;
  const float3 N=safe_normalize(normal);
  const float3 T=safe_normalize(tangent-dot(tangent,N)*N);
  if (is_zero(N) || is_zero(T) || !isfinite_safe(N) || !isfinite_safe(T)) return false;
  const float atom=diffraction_dielectric_straight_mass(param);
  const bool allocate_atom=atom>0 && !(disabled_lobes & LABEL_TRANSMIT);
  if (sd->num_closure_left < (int(allocate_atom)+(atom<1))) return false;
  bool allocated=false;
  if (allocate_atom) {
    ccl_private ShaderClosure *sc=bsdf_alloc(sd,sizeof(ShaderClosure),weight*atom);
    if (sc) {sc->type=CLOSURE_BSDF_DIFFRACTION_STRAIGHT_ID;sc->N=N;
      ((ccl_private DiffractionStraightPolarizerBsdf *)sc)->polarizer=0;allocated=true;}
  }
  if (atom<1) {
    ccl_private DiffractionDielectricBsdf *bsdf=(ccl_private DiffractionDielectricBsdf *)
        bsdf_alloc(sd,sizeof(DiffractionDielectricBsdf),weight*(1-atom));
    if (bsdf) {
      bsdf->type=m_type==BECKMANN?CLOSURE_BSDF_DIFFRACTION_BECKMANN_ID:CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID;
      bsdf->N=N;bsdf->T=T;bsdf->disabled_lobes=disabled_lobes;bsdf->param=*param;allocated=true;
      if (!roughness_is_almost_specular(param->alpha_x,param->alpha_y))
        sd->runtime_flag|=SR_BSDF_HAS_EVAL;
    }
  }
  if (allocated) {
    sd->runtime_flag|=SR_BSDF|SR_BSDF_HAS_DISPERSION;
    if (!(disabled_lobes & LABEL_TRANSMIT)) sd->runtime_flag|=SR_BSDF_HAS_TRANSMISSION;
  }
  return allocated;
}

/* Preserve separate spectral reflection/transmission tints without adding
 * per-ray storage. Identical weights share the ordinary closure pair. Different
 * weights use auxiliary colors with the joint directional proposal. Reserve
 * the complete allocation before mutating sd. */
template<MicrofacetType m_type = GGX>
ccl_device_inline bool bsdf_diffraction_dielectric_setup_tinted(
    ccl_private ShaderData *sd,const Spectrum reflection_weight,
    const Spectrum transmission_weight,const float3 normal,const float3 tangent,
    ccl_private const DiffractionRoughDielectric *param)
{
  if (is_zero(reflection_weight-transmission_weight))
    return bsdf_diffraction_dielectric_setup<m_type>(sd,reflection_weight,normal,tangent,param);
  Spectrum reflection_sample=reflection_weight,transmission_sample=transmission_weight;
  const bool reflection=closure_sample_weight(sd->runtime_flag,reflection_sample)>0;
  const bool transmission=closure_sample_weight(sd->runtime_flag,transmission_sample)>0;
  if (!reflection || !transmission)
    return bsdf_diffraction_dielectric_setup<m_type>(sd,reflection?reflection_weight:transmission_weight,
        normal,tangent,param,reflection?LABEL_TRANSMIT:LABEL_REFLECT);
  const float atom=diffraction_dielectric_straight_mass(param);
  if (!(atom<1))
    return bsdf_diffraction_dielectric_setup<m_type>(sd,transmission_weight,normal,tangent,param);
  const int extra_slots=(sizeof(DiffractionDielectricTintExtra)+sizeof(ShaderClosure)-1)/sizeof(ShaderClosure);
  if (sd->num_closure_left < 1+int(atom>0)+extra_slots) return false;
  const int first=sd->num_closure;
  if (!bsdf_diffraction_dielectric_setup<m_type>(sd,one_spectrum(),normal,tangent,param))return false;
  if(atom>0) {
    sd->closure[first].weight=transmission_weight*atom;
    Spectrum w=sd->closure[first].weight;
    sd->closure[first].sample_weight=closure_sample_weight(sd->runtime_flag,w);
  }
  ccl_private DiffractionDielectricBsdf *bsdf=(ccl_private DiffractionDielectricBsdf *)&sd->closure[first+int(atom>0)];
  ccl_private DiffractionDielectricTintExtra *extra=(ccl_private DiffractionDielectricTintExtra *)closure_alloc_extra(sd,sizeof(DiffractionDielectricTintExtra));
  kernel_assert(extra!=nullptr);
  extra->param=*param;extra->reflection=reflection_weight;extra->transmission=transmission_weight;
  bsdf->extra=extra;bsdf->disabled_lobes|=DIFFRACTION_DIELECTRIC_TINT_EXTRA;
  Spectrum sample=(reflection_weight+transmission_weight)*(.5f*(1-atom));
  bsdf->sample_weight=closure_sample_weight(sd->runtime_flag,sample);
  return true;
}

/* Two separate measures for guiding: a direction-dependent atom and the
 * conditional continuous/order lobe. Neither stored weight absorbs the atom
 * integral. Preflight every slot before creating either closure. */
template<MicrofacetType m_type=GGX>
ccl_device_inline bool bsdf_diffraction_dielectric_setup_coated(
    ccl_private ShaderData *sd,Spectrum reflection_weight,Spectrum transmission_weight,
    const float3 normal,const float3 tangent,ccl_private const DiffractionRoughDielectric *param,
    const float film_ior,const float film_thickness_over_wavelength,const int atom_samples)
{
  if(film_thickness_over_wavelength==0)
    return bsdf_diffraction_dielectric_setup_tinted<m_type>(
        sd,reflection_weight,transmission_weight,normal,tangent,param);
  if(!diffraction_dielectric_parameters_valid(param))return false;
  const bool reflection=closure_sample_weight(sd->runtime_flag,reflection_weight)>0;
  const bool transmission=closure_sample_weight(sd->runtime_flag,transmission_weight)>0;
  if(!reflection && !transmission)return false;
  const float3 N=safe_normalize(normal),T=safe_normalize(tangent-dot(tangent,N)*N);
  if(is_zero(N)||is_zero(T)||!isfinite_safe(N)||!isfinite_safe(T))return false;
  float3 X,Y;make_orthonormals_safe_tangent(N,T,&X,&Y);
  float atom;
  if(!diffraction_dielectric_coated_straight_mass_quadrature<m_type>(param,
      make_float3(dot(sd->wi,X),dot(sd->wi,Y),dot(sd->wi,N)),
      film_ior,film_thickness_over_wavelength,atom_samples,&atom))return false;
  const bool allocate_atom=transmission && param->facet.incident_ior==param->facet.transmitted_ior;
  const int main_extra=(sizeof(DiffractionDielectricCoatingExtra)+sizeof(ShaderClosure)-1)/sizeof(ShaderClosure);
  const int atom_extra=(sizeof(DiffractionCoatedAtomExtra)+sizeof(ShaderClosure)-1)/sizeof(ShaderClosure);
  if(sd->num_closure_left<1+main_extra+int(allocate_atom)*(1+atom_extra))return false;
  if(allocate_atom && !bsdf_diffraction_coated_atom_setup<m_type>(
      sd,transmission_weight,N,T,param,film_ior,film_thickness_over_wavelength,atom_samples))return false;
  ccl_private DiffractionDielectricBsdf *bsdf=(ccl_private DiffractionDielectricBsdf *)
      bsdf_alloc(sd,sizeof(DiffractionDielectricBsdf),one_spectrum());
  kernel_assert(bsdf!=nullptr);
  ccl_private DiffractionDielectricCoatingExtra *extra=(ccl_private DiffractionDielectricCoatingExtra *)
      closure_alloc_extra(sd,sizeof(DiffractionDielectricCoatingExtra));
  kernel_assert(extra!=nullptr);
  extra->base.param=*param;extra->base.reflection=reflection_weight;
  extra->base.transmission=transmission_weight;extra->film_ior=film_ior;
  extra->film_thickness_over_wavelength=film_thickness_over_wavelength;
  bsdf->extra=&extra->base;bsdf->N=N;bsdf->T=T;
  bsdf->type=m_type==BECKMANN?CLOSURE_BSDF_DIFFRACTION_BECKMANN_ID:CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID;
  bsdf->disabled_lobes=DIFFRACTION_DIELECTRIC_TINT_EXTRA|DIFFRACTION_DIELECTRIC_COATING_EXTRA|
      (reflection?0:LABEL_REFLECT)|(transmission?0:LABEL_TRANSMIT);
  Spectrum selection=(reflection_weight+transmission_weight)*(.5f*max(.05f,1-atom));
  bsdf->sample_weight=average(selection);
  sd->runtime_flag|=SR_BSDF|SR_BSDF_HAS_DISPERSION;
  if(transmission)sd->runtime_flag|=SR_BSDF_HAS_TRANSMISSION;
  if(!roughness_is_almost_specular(param->alpha_x,param->alpha_y))sd->runtime_flag|=SR_BSDF_HAS_EVAL;
  return true;
}

/* Shared physical-input conversion for SVM and OSL Glass nodes. */
ccl_device_inline_outline_metal bool bsdf_diffraction_glass_setup(
    ccl_private ShaderData *sd, const Spectrum reflection, const Spectrum transmission,
    const float3 normal, float3 tangent, const float roughness, const float ior,
    const float pitch, const float depth, const float duty,
    const float film_ior, const float film_thickness, const bool beckmann)
{
  const float3 N = safe_normalize_fallback(normal, sd->N);
  tangent -= dot(tangent, N) * N;
  if (len_squared(tangent) < 1e-12f) {
    float3 unused;
    make_orthonormals(N, &tangent, &unused);
  }
  const bool back = (sd->runtime_flag & SR_BACKFACING) != 0;
  const float wavelength = 1000.0f * sample_wavelength(sd->rand_wavelength);
  const float alpha = sqr(saturatef(roughness));
  DiffractionRoughDielectric param;
  if (!diffraction_dielectric_parameters(wavelength, pitch, depth, saturatef(duty),
                                        back ? ior : 1.0f, back ? 1.0f : ior,
                                        alpha, alpha, &param)) {
    return false;
  }
  /* Bounded work only matters for the matched-index coated atom. */
  if (beckmann) {
    return bsdf_diffraction_dielectric_setup_coated<BECKMANN>(
        sd, reflection, transmission, N, tangent, &param,
        film_ior, max(film_thickness, 0.0f) / wavelength, 64);
  }
  return bsdf_diffraction_dielectric_setup_coated<GGX>(
      sd, reflection, transmission, N, tangent, &param,
      film_ior, max(film_thickness, 0.0f) / wavelength, 64);
}

/* Approximate neutral, lossless MultiGGX completion. The first-event closure
 * and this reciprocal two-sided return closure have separate PDFs, so guiding
 * and MIS can select them without a hidden mixture inside either closure. */
ccl_device_inline_outline_metal bool bsdf_diffraction_glass_two_sided_setup(
    KernelGlobals kg,
    ccl_private ShaderData *sd,
    const Spectrum reflection,
    const Spectrum transmission,
    const float3 normal,
    float3 tangent,
    const float roughness,
    const float ior,
    const float pitch,
    const float depth,
    const float duty,
    const int cache_handle,
    const float film_ior = 1.0f,
    const float film_thickness = 0.0f)
{
  if (cache_handle < 0 || !(ior > 1.0f && depth > 0.0f) ||
      sqr(saturatef(roughness)) < 1.0e-4f || is_zero(reflection + transmission) ||
      !(film_ior > 0.0f && film_thickness >= 0.0f) ||
      !isfinite_safe(film_ior + film_thickness))
  {
    return false;
  }
  const float3 N = safe_normalize_fallback(normal, sd->N);
  tangent -= dot(tangent, N) * N;
  if (len_squared(tangent) < 1e-12f) {
    float3 unused;
    make_orthonormals(N, &tangent, &unused);
  }
  const bool back = (sd->runtime_flag & SR_BACKFACING) != 0;
  const float wavelength_nm = 1000.0f * sample_wavelength(sd->rand_wavelength);
  const float alpha = sqr(saturatef(roughness));
  DiffractionRoughDielectric param;
  if (!diffraction_dielectric_parameters(wavelength_nm,
                                        pitch,
                                        depth,
                                        saturatef(duty),
                                        back ? ior : 1.0f,
                                        back ? 1.0f : ior,
                                        alpha,
                                        alpha,
                                        &param))
  {
    return false;
  }
  const float3 X = safe_normalize(tangent);
  const float3 Y = normalize(cross(N, X));
  const float3 local_wi = make_float3(dot(sd->wi, X), dot(sd->wi, Y), dot(sd->wi, N));
  const float4 incident = diffraction_two_sided_cache_lookup(
      kg, cache_handle, wavelength_nm, int(back), local_wi);
  DiffractionTwoSidedReturn returned;
  const bool has_return = incident.x > 1.0e-7f &&
                          diffraction_two_sided_return(
                              incident.y, incident.z, incident.w, &returned);
  const int single_extra = film_thickness > 0.0f ?
      (sizeof(DiffractionDielectricCoatingExtra) + sizeof(ShaderClosure) - 1) /
          sizeof(ShaderClosure) :
      (is_zero(reflection - transmission) ? 0 :
           (sizeof(DiffractionDielectricTintExtra) + sizeof(ShaderClosure) - 1) /
               sizeof(ShaderClosure));
  const int ms_extra = has_return ?
      (sizeof(DiffractionDielectricTwoSidedExtra) + sizeof(ShaderClosure) - 1) /
          sizeof(ShaderClosure) :
      0;
  if (sd->num_closure_left < 1 + single_extra + int(has_return) * (1 + ms_extra)) {
    return false;
  }
  const int first_closure = sd->num_closure;
  if (!bsdf_diffraction_glass_setup(sd,
                                    reflection,
                                    transmission,
                                    N,
                                    tangent,
                                    roughness,
                                    ior,
                                    pitch,
                                    depth,
                                    duty,
                                    film_ior,
                                    film_thickness,
                                    false))
  {
    return false;
  }
  /* Filter Glossy must not roughen only the first-event term while the
   * complementary table still represents the original roughness. */
  ccl_private DiffractionDielectricBsdf *single =
      (ccl_private DiffractionDielectricBsdf *)&sd->closure[first_closure];
  kernel_assert(single->type == CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID);
  single->disabled_lobes |= DIFFRACTION_DIELECTRIC_CACHE_BASE;
  if (!has_return) return true;
  ccl_private DiffractionDielectricBsdf *bsdf =
      (ccl_private DiffractionDielectricBsdf *)bsdf_alloc(
          sd, sizeof(DiffractionDielectricBsdf), one_spectrum());
  kernel_assert(bsdf != nullptr);
  ccl_private DiffractionDielectricTwoSidedExtra *extra =
      (ccl_private DiffractionDielectricTwoSidedExtra *)closure_alloc_extra(
          sd, sizeof(DiffractionDielectricTwoSidedExtra));
  kernel_assert(extra != nullptr);
  extra->base.param = param;
  extra->base.reflection = reflection;
  extra->base.transmission = transmission;
  extra->cache_handle = cache_handle;
  extra->incoming_side = int(back);
  extra->inside_ior = ior;
  extra->wavelength_nm = wavelength_nm;
  extra->generalized_f0 = zero_spectrum();
  extra->ior_d = ior;
  extra->inv_abbe = 0.0f;
  bsdf->N = N;
  bsdf->T = tangent;
  bsdf->extra = &extra->base;
  bsdf->type = CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID;
  bsdf->disabled_lobes = DIFFRACTION_DIELECTRIC_TINT_EXTRA |
                         DIFFRACTION_DIELECTRIC_TWO_SIDED_MS |
                         (is_zero(reflection) ? int(LABEL_REFLECT) : 0) |
                         (is_zero(transmission) ? int(LABEL_TRANSMIT) : 0);
  const float reflected_probability = diffraction_two_sided_side_probability(
      returned, int(back), int(back));
  const float transmitted_probability = diffraction_two_sided_side_probability(
      returned, int(back), 1 - int(back));
  Spectrum selection =
      (reflection * reflected_probability + transmission * transmitted_probability) * incident.x;
  bsdf->sample_weight = closure_sample_weight(sd->runtime_flag, selection);
  sd->runtime_flag |= SR_BSDF | SR_BSDF_HAS_EVAL | SR_BSDF_HAS_DISPERSION;
  if (!(bsdf->disabled_lobes & LABEL_TRANSMIT)) sd->runtime_flag |= SR_BSDF_HAS_TRANSMISSION;
  return true;
}

ccl_device_inline_outline_metal bool bsdf_diffraction_principled_transmission_setup(
    ccl_private ShaderData *sd,const Spectrum reflection,const Spectrum transmission,
    const float3 normal,const float3 tangent,const float roughness,const float ior,
    const float pitch,const float depth,const float duty,const Spectrum specular_tint,
    const float inv_abbe=0.0f,const float film_ior=1.0f,const float film_thickness=0.0f)
{
  const bool back=(sd->runtime_flag & SR_BACKFACING)!=0;
  float physical_ior=ior;
  if(inv_abbe!=0.0f) {
    const float oriented=bsdf_glass_ior(sd,back?1.0f/ior:ior,inv_abbe);
    physical_ior=back?1.0f/oriented:oriented;
  }
  const bool film=film_thickness>THINFILM_THICKNESS_CUTOFF;
  /* Native film tint correction vanishes near matched indices. Reuse the
   * physical coated atom there, including its direction-dependent integral. */
  if(film && (F0_from_ior(physical_ior)<=1e-5f ||
              (isequal(specular_tint,one_spectrum()) && physical_ior==ior)))
    return bsdf_diffraction_glass_setup(sd,reflection,transmission,normal,tangent,roughness,
        physical_ior,pitch,depth,duty,film_ior,film_thickness,false);
  const Spectrum f0=saturate(F0_from_ior(ior)*specular_tint);
  if((isequal(specular_tint,one_spectrum()) && physical_ior==ior) ||
     (physical_ior==1.0f && is_zero(f0)))
    return bsdf_diffraction_glass_setup(sd,reflection,transmission,normal,tangent,roughness,
                                      physical_ior,pitch,depth,duty,1,0,false);
  const bool matched=physical_ior==1.0f;
  const Spectrum atom_weight=matched?transmission*(one_spectrum()-f0):zero_spectrum();
  const bool atom=!is_zero(atom_weight);
  const bool continuous=!matched || !is_zero(reflection*f0);
  const int slots=(sizeof(DiffractionDielectricGeneralizedExtra)+sizeof(ShaderClosure)-1)/sizeof(ShaderClosure);
  if(sd->num_closure_left<int(atom)+int(continuous)*(1+slots))return false;
  const float3 N=safe_normalize_fallback(normal,sd->N);
  float3 T=tangent-dot(tangent,N)*N;
  if(len_squared(T)<1e-12f) {float3 unused;make_orthonormals(N,&T,&unused);}
  T=safe_normalize(T);
  const float alpha=sqr(saturatef(roughness));
  DiffractionRoughDielectric param;
  if(!diffraction_dielectric_parameters(1000*sample_wavelength(sd->rand_wavelength),
      pitch,depth,saturatef(duty),back?physical_ior:1.0f,back?1.0f:physical_ior,
      alpha,alpha,&param))return false;
  if(atom && !bsdf_diffraction_dielectric_setup<GGX>(sd,atom_weight,N,T,&param,LABEL_REFLECT))
    return false;
  if(!continuous)return atom;
  const int disabled=(is_zero(reflection)?int(LABEL_REFLECT):0)|
                     ((matched || is_zero(transmission))?int(LABEL_TRANSMIT):0);
  if(disabled==(LABEL_REFLECT|LABEL_TRANSMIT))return atom;
  ccl_private DiffractionDielectricBsdf *b=(ccl_private DiffractionDielectricBsdf *)
      bsdf_alloc(sd,sizeof(DiffractionDielectricBsdf),one_spectrum());
  kernel_assert(b!=nullptr);
  b->N=N;b->T=T;b->type=CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID;
  b->disabled_lobes=disabled|DIFFRACTION_DIELECTRIC_TINT_EXTRA|DIFFRACTION_DIELECTRIC_GENERALIZED;
  ccl_private DiffractionDielectricGeneralizedExtra *e=(ccl_private DiffractionDielectricGeneralizedExtra *)
      closure_alloc_extra(sd,sizeof(DiffractionDielectricGeneralizedExtra));
  kernel_assert(e!=nullptr);
  e->base.param=param;e->base.reflection=reflection;
  e->base.transmission=matched?zero_spectrum():transmission;
  e->generalized_f0=f0;
  e->generalized_reference_f0=F0_from_ior(back?1.0f/physical_ior:physical_ior);
  e->film_ior=film_ior;
  e->film_thickness_over_wavelength=film?film_thickness/(1000*sample_wavelength(sd->rand_wavelength)):0.0f;
  b->extra=&e->base;
  Spectrum sample=matched?reflection*f0:(reflection+transmission)*.5f;
  b->sample_weight=closure_sample_weight(sd->runtime_flag,sample);
  sd->runtime_flag|=SR_BSDF|SR_BSDF_HAS_DISPERSION;
  if(!(disabled&LABEL_TRANSMIT))sd->runtime_flag|=SR_BSDF_HAS_TRANSMISSION;
  if(!roughness_is_almost_specular(alpha,alpha))sd->runtime_flag|=SR_BSDF_HAS_EVAL;
  return true;
}

/* The physical, solid Principled transmission lobe has the same first-event
 * dielectric transport as Glass when its generalized tint and dispersion are
 * disabled. Its external closure weights may still mix with other Principled
 * lobes; the neutral return cache is conditional on this lobe only. */
ccl_device_inline_outline_metal bool bsdf_diffraction_principled_transmission_two_sided_setup(
    KernelGlobals kg,
    ccl_private ShaderData *sd,
    const Spectrum reflection,
    const Spectrum transmission,
    const float3 normal,
    const float3 tangent,
    const float roughness,
    const float ior,
    const float pitch,
    const float depth,
    const float duty,
    const Spectrum specular_tint,
    const float inv_abbe,
    const float film_ior,
    const float film_thickness,
    const int cache_handle)
{
  if (cache_handle < 0 || cache_handle >= kernel_data.tables.num_diffraction_two_sided_caches) {
    return false;
  }
  const int4 descriptor = kernel_data_fetch(diffraction_two_sided_descriptors, cache_handle);
  const int basis_count = 1 + (descriptor.w >> 8);
  const bool film = film_thickness > THINFILM_THICKNESS_CUTOFF;
  const float effective_film_thickness = film ? film_thickness : 0.0f;
  if (basis_count == 1) {
    if (!isequal(specular_tint, one_spectrum()) || inv_abbe != 0.0f) return false;
    return bsdf_diffraction_glass_two_sided_setup(
        kg, sd, reflection, transmission, normal, tangent, roughness, ior, pitch, depth,
        duty, cache_handle, film ? film_ior : 1.0f, effective_film_thickness);
  }
  /* Bare powers use exact endpoint bases; coated powers use the host's
   * bounded, nonuniform F0 grid. Never apply the bare interpolant to a film. */
  if ((film ? basis_count != 16 : basis_count != 2) ||
      !(ior > 0.0f && depth > 0.0f) || sqr(saturatef(roughness)) < 1.0e-4f)
  {
    return false;
  }
  const bool back = (sd->runtime_flag & SR_BACKFACING) != 0;
  const float oriented_ior = bsdf_glass_ior(sd, back ? 1.0f / ior : ior, inv_abbe);
  const float physical_ior = back ? 1.0f / oriented_ior : oriented_ior;
  const float wavelength_nm = 1000.0f * sample_wavelength(sd->rand_wavelength);
  const float3 N = safe_normalize_fallback(normal, sd->N);
  float3 T = tangent - dot(tangent, N) * N;
  if (len_squared(T) < 1.0e-12f) {
    float3 unused;
    make_orthonormals(N, &T, &unused);
  }
  T = safe_normalize(T);
  float3 X, Y;
  make_orthonormals_safe_tangent(N, T, &X, &Y);
  const float3 I = make_float3(dot(sd->wi, X), dot(sd->wi, Y), dot(sd->wi, N));
  DiffractionDielectricTwoSidedExtra lookup{};
  if (!diffraction_dielectric_parameters(wavelength_nm, pitch, depth, saturatef(duty),
      back ? physical_ior : 1.0f, back ? 1.0f : physical_ior,
      sqr(saturatef(roughness)), sqr(saturatef(roughness)), &lookup.base.param)) return false;
  lookup.base.reflection = reflection;
  lookup.base.transmission = transmission;
  lookup.cache_handle = cache_handle;
  lookup.incoming_side = int(back);
  lookup.inside_ior = physical_ior;
  lookup.wavelength_nm = wavelength_nm;
  lookup.generalized_f0 = saturate(F0_from_ior(ior) * specular_tint);
  lookup.ior_d = ior;
  lookup.inv_abbe = inv_abbe;
  Spectrum selection = zero_spectrum();
  FOREACH_SPECTRUM_CHANNEL(channel) {
    const float4 cache = diffraction_dielectric_ms_cache(kg, &lookup, int(back), I, channel);
    DiffractionTwoSidedReturn returned;
    if (!(cache.x > 1.0e-7f) ||
        !diffraction_two_sided_return(cache.y, cache.z, cache.w, &returned)) continue;
    GET_SPECTRUM_CHANNEL(selection, channel) = cache.x * (
        GET_SPECTRUM_CHANNEL(reflection, channel) * diffraction_two_sided_side_probability(
            returned, int(back), int(back)) +
        GET_SPECTRUM_CHANNEL(transmission, channel) * diffraction_two_sided_side_probability(
            returned, int(back), 1 - int(back)));
  }
  const bool has_return = average(selection) > 0.0f;
  /* Match the first-event setup's near-index coating limit. Its atom is
   * facet-dependent and cannot be replaced by the bare 1-F0 atom. */
  const bool physical = (isequal(specular_tint, one_spectrum()) && physical_ior == ior) ||
      (film && F0_from_ior(physical_ior) <= 1.0e-5f);
  const bool matched = physical_ior == 1.0f;
  const bool atom = matched && !is_zero(transmission * (one_spectrum() - lookup.generalized_f0));
  const bool continuous = !matched || !is_zero(reflection * lookup.generalized_f0);
  const int physical_extra = is_zero(reflection - transmission) ? 0 :
      (sizeof(DiffractionDielectricTintExtra) + sizeof(ShaderClosure) - 1) / sizeof(ShaderClosure);
  const int generalized_extra =
      (sizeof(DiffractionDielectricGeneralizedExtra) + sizeof(ShaderClosure) - 1) /
          sizeof(ShaderClosure);
  const int coating_extra =
      (sizeof(DiffractionDielectricCoatingExtra) + sizeof(ShaderClosure) - 1) /
          sizeof(ShaderClosure);
  const int coating_atom_extra =
      (sizeof(DiffractionCoatedAtomExtra) + sizeof(ShaderClosure) - 1) /
          sizeof(ShaderClosure);
  const int first_slots = physical ?
      (film ? 1 + coating_extra + int(matched && !is_zero(transmission)) *
          (1 + coating_atom_extra) : 1 + physical_extra) :
      int(atom) + int(continuous) * (1 + generalized_extra);
  const int return_slots = 1 +
      (sizeof(DiffractionDielectricTwoSidedExtra) + sizeof(ShaderClosure) - 1) / sizeof(ShaderClosure);
  if (sd->num_closure_left < first_slots + int(has_return) * return_slots) return false;
  const int first_closure = sd->num_closure;
  if (!bsdf_diffraction_principled_transmission_setup(
      sd, reflection, transmission, N, T, roughness, ior, pitch, depth, duty,
      specular_tint, inv_abbe, film_ior, effective_film_thickness)) return false;
  for (int i = first_closure; i < sd->num_closure; i++) {
    if (sd->closure[i].type == CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID) {
      ((ccl_private DiffractionDielectricBsdf *)&sd->closure[i])->disabled_lobes |=
          DIFFRACTION_DIELECTRIC_CACHE_BASE;
    }
  }
  if (!has_return) return true;
  ccl_private DiffractionDielectricBsdf *bsdf =
      (ccl_private DiffractionDielectricBsdf *)bsdf_alloc(
          sd, sizeof(DiffractionDielectricBsdf), one_spectrum());
  ccl_private DiffractionDielectricTwoSidedExtra *extra =
      (ccl_private DiffractionDielectricTwoSidedExtra *)closure_alloc_extra(
          sd, sizeof(DiffractionDielectricTwoSidedExtra));
  kernel_assert(bsdf != nullptr && extra != nullptr);
  *extra = lookup;
  bsdf->N = N;
  bsdf->T = T;
  bsdf->extra = &extra->base;
  bsdf->type = CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID;
  bsdf->disabled_lobes = DIFFRACTION_DIELECTRIC_TINT_EXTRA | DIFFRACTION_DIELECTRIC_TWO_SIDED_MS |
      (is_zero(reflection) ? int(LABEL_REFLECT) : 0) |
      (is_zero(transmission) ? int(LABEL_TRANSMIT) : 0);
  bsdf->sample_weight = closure_sample_weight(sd->runtime_flag, selection);
  sd->runtime_flag |= SR_BSDF | SR_BSDF_HAS_EVAL | SR_BSDF_HAS_DISPERSION;
  if (!is_zero(transmission)) sd->runtime_flag |= SR_BSDF_HAS_TRANSMISSION;
  return true;
}

ccl_device_inline_outline_metal bool bsdf_diffraction_refraction_setup(
    ccl_private ShaderData *sd, const Spectrum weight, const float3 normal,
    float3 tangent, const float roughness, const float ior,
    const float pitch, const float depth, const float duty, const bool beckmann)
{
  const float3 N=safe_normalize_fallback(normal,sd->N);
  tangent-=dot(tangent,N)*N;
  if (len_squared(tangent)<1e-12f) {
    float3 unused;make_orthonormals(N,&tangent,&unused);
  }
  const bool back=(sd->runtime_flag & SR_BACKFACING)!=0;
  const float wavelength=1000.0f*sample_wavelength(sd->rand_wavelength);
  const float alpha=sqr(saturatef(roughness));
  DiffractionRoughDielectric param;
  if (!diffraction_dielectric_parameters(wavelength,pitch,depth,saturatef(duty),
                                        back?ior:1.0f,back?1.0f:ior,alpha,alpha,&param))
    return false;
  const int flags=int(LABEL_REFLECT)|DIFFRACTION_DIELECTRIC_PURE_REFRACTION;
  return beckmann ? bsdf_diffraction_dielectric_setup<BECKMANN>(sd,weight,N,tangent,&param,flags) :
                    bsdf_diffraction_dielectric_setup<GGX>(sd,weight,N,tangent,&param,flags);
}


/* Get the declared analyzer axis without changing the native scalar proposal.
 * The integrator supplies an explicit depolarizing Fast substrate for rough
 * grating carriers, and a preserving substrate for straight-through atoms. */
ccl_device_inline bool bsdf_diffraction_polarizer_axis(
    const ccl_private ShaderClosure *sc, ccl_private float3 *axis)
{
  if(sc->type==CLOSURE_BSDF_DIFFRACTION_STRAIGHT_ID) {
    const ccl_private DiffractionStraightPolarizerBsdf *b=(const ccl_private DiffractionStraightPolarizerBsdf *)sc;
    if(!b->polarizer)return false;*axis=b->polarizer_axis;return true;
  }
  if(sc->type==CLOSURE_BSDF_DIFFRACTION_COATED_STRAIGHT_ID) {
    const ccl_private DiffractionCoatedAtomBsdf *b=(const ccl_private DiffractionCoatedAtomBsdf *)sc;
    if(!b->polarizer)return false;
    *axis=((const ccl_private DiffractionPolarizerExtra<DiffractionCoatedAtomExtra> *)b->extra)->axis;
    return true;
  }
  if(sc->type!=CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID &&
     sc->type!=CLOSURE_BSDF_DIFFRACTION_BECKMANN_ID)return false;
  const ccl_private DiffractionDielectricBsdf *b=(const ccl_private DiffractionDielectricBsdf *)sc;
  if(!(b->disabled_lobes&DIFFRACTION_DIELECTRIC_POLARIZER))return false;
  if(b->disabled_lobes&DIFFRACTION_DIELECTRIC_TWO_SIDED_MS)
    *axis=((const ccl_private DiffractionPolarizerExtra<DiffractionDielectricTwoSidedExtra> *)b->extra)->axis;
  else if(b->disabled_lobes&DIFFRACTION_DIELECTRIC_COATING_EXTRA)
    *axis=((const ccl_private DiffractionPolarizerExtra<DiffractionDielectricCoatingExtra> *)b->extra)->axis;
  else *axis=((const ccl_private DiffractionPolarizerExtra<DiffractionDielectricTintExtra> *)b->extra)->axis;
  return true;
}

/* All-or-nothing metadata attachment. Check every required inline-carrier
 * allocation before changing any closure; existing extras use their padding. */
ccl_device_inline bool bsdf_diffraction_glass_set_polarizer(
    ccl_private ShaderData *sd,const int first_closure,const float3 axis)
{
  if(first_closure<0 || first_closure>sd->num_closure || !isfinite_safe(axis) ||
     !(len_squared(axis)>1e-12f))return false;
  int required=0;
  const int end=sd->num_closure;
  for(int i=first_closure;i<end;i++) {
    const ccl_private ShaderClosure *sc=&sd->closure[i];
    if(sc->type==CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID ||
       sc->type==CLOSURE_BSDF_DIFFRACTION_BECKMANN_ID) {
      const ccl_private DiffractionDielectricBsdf *b=(const ccl_private DiffractionDielectricBsdf *)sc;
      if(b->disabled_lobes&DIFFRACTION_DIELECTRIC_GENERALIZED)return false;
      required+=!(b->disabled_lobes&DIFFRACTION_DIELECTRIC_TINT_EXTRA);
    }
  }
  if(sd->num_closure_left<required)return false;
  for(int i=first_closure;i<end;i++) {
    ccl_private ShaderClosure *sc=&sd->closure[i];
    if(sc->type==CLOSURE_BSDF_DIFFRACTION_STRAIGHT_ID) {
      ccl_private DiffractionStraightPolarizerBsdf *b=(ccl_private DiffractionStraightPolarizerBsdf *)sc;
      b->polarizer_axis=axis;b->polarizer=1;
    }
    else if(sc->type==CLOSURE_BSDF_DIFFRACTION_COATED_STRAIGHT_ID) {
      ccl_private DiffractionCoatedAtomBsdf *b=(ccl_private DiffractionCoatedAtomBsdf *)sc;
      ((ccl_private DiffractionPolarizerExtra<DiffractionCoatedAtomExtra> *)b->extra)->axis=axis;
      b->polarizer=1;
    }
    else if(sc->type==CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID ||
            sc->type==CLOSURE_BSDF_DIFFRACTION_BECKMANN_ID) {
      ccl_private DiffractionDielectricBsdf *b=(ccl_private DiffractionDielectricBsdf *)sc;
      if(!(b->disabled_lobes&DIFFRACTION_DIELECTRIC_TINT_EXTRA)) {
        const auto param=b->param;
        ccl_private DiffractionDielectricTintExtra *extra=(ccl_private DiffractionDielectricTintExtra *)closure_alloc_extra(
            sd,sizeof(DiffractionPolarizerExtra<DiffractionDielectricTintExtra>));
        kernel_assert(extra!=nullptr);extra->param=param;
        extra->reflection=extra->transmission=one_spectrum();
        b->extra=extra;b->disabled_lobes|=DIFFRACTION_DIELECTRIC_TINT_EXTRA;
      }
      if(b->disabled_lobes&DIFFRACTION_DIELECTRIC_TWO_SIDED_MS)
        ((ccl_private DiffractionPolarizerExtra<DiffractionDielectricTwoSidedExtra> *)b->extra)->axis=axis;
      else if(b->disabled_lobes&DIFFRACTION_DIELECTRIC_COATING_EXTRA)
        ((ccl_private DiffractionPolarizerExtra<DiffractionDielectricCoatingExtra> *)b->extra)->axis=axis;
      else ((ccl_private DiffractionPolarizerExtra<DiffractionDielectricTintExtra> *)b->extra)->axis=axis;
      b->disabled_lobes|=DIFFRACTION_DIELECTRIC_POLARIZER;
    }
  }
  return true;
}

CCL_NAMESPACE_END
