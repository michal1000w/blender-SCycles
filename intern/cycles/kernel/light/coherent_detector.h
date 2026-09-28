/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

CCL_NAMESPACE_BEGIN

/* Host certification admits only a Lambertian graph. Runtime filters can
 * still remove it, so ownership must use this same predicate at eye and light
 * endpoints. A black Lambertian may allocate no closure and owns zero energy. */
ccl_device_inline bool coherent_detector_eligible(ccl_private const ShaderData *sd)
{
  return (sd->object_flag & SD_OBJECT_COHERENT_DETECTOR) &&
         (sd->num_closure == 0 ||
          (sd->num_closure == 1 && sd->closure[0].type == CLOSURE_BSDF_DIFFUSE_ID));
}

ccl_device_inline Spectrum coherent_detector_passive_color(Spectrum color)
{
  FOREACH_SPECTRUM_CHANNEL(channel) {
    const float value = GET_SPECTRUM_CHANNEL(color, channel);
    /* Integer exponent classification survives finite-only GPU fast math. */
    const bool finite = (__float_as_uint(value) & 0x7f800000u) != 0x7f800000u;
    GET_SPECTRUM_CHANNEL(color, channel) = finite ? saturatef(value) : 0.0f;
  }
  return color;
}

/* Linked RGB detector colors are passive at this declared material point.
 * Apply the same bounded response to native scattering and coherent fields;
 * clamping only the connector would change the estimator's owned class. */
ccl_device_inline void coherent_detector_prepare(ccl_private ShaderData *sd)
{
  if (!coherent_detector_eligible(sd) || sd->num_closure == 0) return;
  ccl_private ShaderClosure *sc = &sd->closure[0];
  sc->weight = coherent_detector_passive_color(sc->weight);
  /* One closure: scaling its weight cannot change the directional PDF. */
  sc->sample_weight = average(sc->weight);
}

CCL_NAMESPACE_END
