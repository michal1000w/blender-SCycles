/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf.h"
#include "kernel/integrator/path_state.h"
#include "kernel/light/coherent_detector.h"
#include "kernel/light/coherent_history.h"
#include "util/profiling.h"
#include <cstdio>
using namespace ccl;
int main() {
  int checks=0, failures=0;
  auto check=[&](bool result){checks++;failures+=!result;};
  KernelGlobalsCPU base{};
  base.data.integrator.max_bounce=4;
  base.data.integrator.max_diffuse_bounce=1; // UI zero + native upload offset.
  base.data.integrator.max_glossy_bounce=3;
  Profiler profiler;ThreadKernelGlobalsCPU globals(base,nullptr,profiler,0);KernelGlobals kg=&globals;
  IntegratorStateCPU state{};
  ShaderData sd{};sd.object_flag=SD_OBJECT_COHERENT_DETECTOR;sd.num_closure=1;
  sd.closure[0].type=CLOSURE_BSDF_DIFFUSE_ID;sd.closure[0].weight=one_spectrum();
  check(coherent_detector_eligible(&sd));
  KernelCoherentCandidate direct{};direct.light=3;
  check(coherent_history_owns_direct_source(&direct,3,true));
  check(coherent_history_matches_source(coherent_history_begin(),&direct,9,9));
  check(coherent_history_candidate_within_budget(&direct,0,0,0,4,3,1,true,3,100));
  KernelCoherentCandidate mirrors{};mirrors.count=2;mirrors.event[0]=mirrors.event[1]=0;
  check(coherent_history_candidate_within_budget(&mirrors,0,0,0,4,3,1,true,3,100));
  check(!(state.path.flag & PATH_RAY_TERMINATE_AFTER_TRANSPARENT));
  path_state_next(kg,&state,LABEL_REFLECT|LABEL_DIFFUSE,0);
  check(state.path.diffuse_bounce==1);
  check(state.path.flag & PATH_RAY_TERMINATE_AFTER_TRANSPARENT);
  check(state.path.bounce==1);
  // Contrast UI one: first diffuse continuation survives, allowing interreflection.
  globals.data.integrator.max_diffuse_bounce=2;state={};
  path_state_next(kg,&state,LABEL_REFLECT|LABEL_DIFFUSE,0);
  check(!(state.path.flag & PATH_RAY_TERMINATE_AFTER_TRANSPARENT));
  path_state_next(kg,&state,LABEL_REFLECT|LABEL_GLOSSY|LABEL_SINGULAR,0);
  check(state.path.bounce==2 && state.path.diffuse_bounce==1);
  check(coherent_history_candidate_within_budget(&direct,2,1,0,4,3,1,true,3,100));
  std::printf("checks=%d failures=%d\n",checks,failures);return failures!=0;
}
