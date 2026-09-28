/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/device/cpu/compat.h"
#include "kernel/types.h"
#include <cstdio>
/* Test only constant-data reads; avoid instantiating a stale ABI constructor
 * before the cohesive build. Production kernel helpers remain unmodified. */
#define __KERNEL_GPU__
namespace ccl {
struct PatchTestGlobals {
  KernelData data{};
  KernelCoherentPatch *coherent_patches;
  int *coherent_patch_primitives;
  KernelCoherentCandidate *coherent_candidates;
  KernelLight *lights;
};
using KernelGlobals = const PatchTestGlobals *;
}
#define kernel_data (kg->data)
#define kernel_data_fetch(name,index) (kg->name[index])
#include "kernel/light/coherent_patch_membership.h"
#include "kernel/light/coherent_history_kernel.h"
#include "kernel/light/coherent_geometry.h"
using namespace ccl;
static int checks=0,failures=0;
static void check(bool value,const char *name){checks++;if(!value){failures++;std::fprintf(stderr,"FAIL %s\n",name);}}
int main()
{
  KernelCoherentPatch patches[3]{};
  int primitives[]={0,2,3,4,0,2};
  patches[0].object=patches[1].object=7;patches[2].object=8;
  for(int i=0;i<3;i++){patches[i].primitive_offset=2*i;patches[i].primitive_count=2;patches[i].mode=1;patches[i].primitive_type=PRIMITIVE_TRIANGLE;}
  KernelLight light{};light.object_id=12;
  KernelCoherentCandidate candidate{};candidate.count=2;candidate.patch[0]=0;candidate.patch[1]=1;
  PatchTestGlobals globals{};globals.coherent_patches=patches;globals.coherent_patch_primitives=primitives;
  globals.coherent_candidates=&candidate;globals.lights=&light;
  globals.data.integrator.coherent_patch_count=3;globals.data.integrator.coherent_candidate_count=1;
  KernelGlobals kg=&globals;
  check(coherent_patch_for_hit(kg,7,0,PRIMITIVE_POINT)==-1,"point ID cannot alias triangle ID");
  patches[2].primitive_type=PRIMITIVE_POINT;
  check(coherent_patch_for_hit(kg,8,0,PRIMITIVE_POINT)==2 && coherent_patch_for_hit(kg,8,0)==-1,"sphere lookup type qualified");
  patches[2].primitive_type=PRIMITIVE_TRIANGLE;
  check(coherent_patch_for_hit(kg,7,0)==0 && coherent_patch_for_hit(kg,7,3)==1,
        "same mesh's two faces map to distinct patches");
  check(coherent_patch_for_hit(kg,8,0)==2,"instance sharing primitive IDs maps by object");
  check(coherent_patch_for_hit(kg,9,0)==-1,"unmarked instance retains ordinary estimator");
  check(coherent_patch_for_hit(kg,7,1)==-1,"hole or undeclared triangle is not a patch");
  check(coherent_patch_transition_valid(kg,0,1,7,3),"same-object next distinct face accepted");
  check(!coherent_patch_transition_valid(kg,0,1,7,2),"previous face self-hit cannot masquerade as next face");
  check(!coherent_patch_transition_valid(kg,0,0,7,2),"same-face immediate repeat rejected");
  check(!coherent_patch_transition_valid(kg,-1,0,7,1),"bounding rectangle hole remains blocked");
  check(!coherent_patch_transition_valid(kg,0,1,8,3),"different instance cannot satisfy endpoint");
  auto history=coherent_history_after_scatter(coherent_history_begin(),coherent_patch_for_hit(kg,7,0),1,LABEL_REFLECT|LABEL_SINGULAR,0);
  history=coherent_history_after_scatter(history,coherent_patch_for_hit(kg,7,3),1,LABEL_REFLECT|LABEL_SINGULAR,0);
  check(coherent_history_owned_candidate(kg,history,12),"same-object mirror prefix owned exactly");
  check(!coherent_history_owned_candidate(kg,history,13),"source identity retained");
  const auto wrong=coherent_history_after_scatter(coherent_history_begin(),coherent_patch_for_hit(kg,8,0),1,LABEL_REFLECT|LABEL_SINGULAR,0);
  check(!coherent_history_owned_candidate(kg,wrong,12),"instance prefix does not match another object's class");
  const auto hole=coherent_history_after_scatter(coherent_history_begin(),coherent_patch_for_hit(kg,7,1),1,LABEL_REFLECT|LABEL_SINGULAR,0);
  check(!coherent_history_valid(hole),"nonparticipating hit invalidates source class");
  candidate.event[0]=candidate.event[1]=1;candidate.expected_incident_side[0]=1;candidate.expected_incident_side[1]=-1;
  auto slab=coherent_history_after_scatter(coherent_history_begin(),coherent_patch_for_hit(kg,7,0),2,LABEL_TRANSMIT|LABEL_SINGULAR,1);
  slab=coherent_history_after_scatter(slab,coherent_patch_for_hit(kg,7,3),2,LABEL_TRANSMIT|LABEL_SINGULAR,-1);
  check(coherent_history_owned_candidate(kg,slab,12),"same-object slab entry/exit signature and medium sides owned");
  /* A Glass point has one primitive and two interfaces. Every isolated TT
   * root belongs to the same native prefix class: ownership is a boolean
   * termination decision, not a sum over the three deterministic roots. */
  KernelCoherentCandidate sphere_roots[3]{};
  patches[2].shape=1; patches[2].mode=2; patches[2].primitive_type=PRIMITIVE_POINT;
  for(int root=0;root<3;root++) {
    auto &c=sphere_roots[root];c.count=2;c.sphere_branch=root;
    c.patch[0]=c.patch[1]=2;c.event[0]=c.event[1]=1;
    c.expected_incident_side[0]=1;c.expected_incident_side[1]=-1;
  }
  const int sphere_patch=coherent_patch_for_hit(kg,8,0,PRIMITIVE_POINT);
  auto sphere_tt=coherent_history_after_scatter(coherent_history_begin(),sphere_patch,2,
                                               LABEL_TRANSMIT|LABEL_SINGULAR,1);
  sphere_tt=coherent_history_after_scatter(sphere_tt,sphere_patch,2,
                                          LABEL_TRANSMIT|LABEL_SINGULAR,-1);
  check(coherent_history_valid(sphere_tt) && coherent_history_count(sphere_tt)==2,
        "same sphere primitive retains both TT interfaces");
  check(coherent_patch_transition_valid(kg,2,2,8,0,PRIMITIVE_POINT),
        "repeated Glass sphere exact primitive membership accepted");
  check(!coherent_patch_transition_valid(kg,2,2,7,0,PRIMITIVE_POINT),
        "repeated sphere cannot cross instance identity");
  globals.coherent_candidates=sphere_roots;globals.data.integrator.coherent_candidate_count=3;
  check(coherent_history_owned_candidate(kg,sphere_tt,12),
        "three root inventory suppresses native TT prefix once through boolean ownership");
  check(!coherent_history_owned_candidate(kg,sphere_tt,13),"TT source identity mismatch rejected");
  for(int root=0;root<3;root++) {
    check(coherent_history_matches(sphere_tt,&sphere_roots[root]),
          "branch ID does not split the native TT ownership class");
    globals.coherent_candidates=&sphere_roots[root];globals.data.integrator.coherent_candidate_count=1;
    check(coherent_history_owned_candidate(kg,sphere_tt,12),
          "each isolated root has identical prefix ownership");
  }
  globals.coherent_candidates=sphere_roots;globals.data.integrator.coherent_candidate_count=3;
  for(int entry_side : {-1,1}) for(int exit_side : {-1,1}) {
    auto wrong_tt=coherent_history_after_scatter(coherent_history_begin(),sphere_patch,2,
                                                LABEL_TRANSMIT|LABEL_SINGULAR,entry_side);
    wrong_tt=coherent_history_after_scatter(wrong_tt,sphere_patch,2,
                                           LABEL_TRANSMIT|LABEL_SINGULAR,exit_side);
    check(coherent_history_owned_candidate(kg,wrong_tt,12)==(entry_side==1 && exit_side==-1),
          "reversed and wrong-side sphere histories rejected");
  }
  check(!coherent_history_owned_candidate(kg,coherent_history_invalidate(sphere_tt),12),
        "invalid TT prefix retains ordinary transport");
  globals.coherent_candidates=&candidate;globals.data.integrator.coherent_candidate_count=1;
  CoherentGeometryInterface mirror[4]{};
  mirror[0]={make_float3(0,.5f,1),make_float3(0,1,0),make_float3(0,0,1),1,1,1,1,1,0,1};
  mirror[1]={make_float3(.5f,0,1),make_float3(1,0,0),make_float3(0,0,-1),1,1,1,1,1,0,1};
  CoherentGeometryPath path{};
  check(coherent_geometry_connect(make_float3(.2f,.8f,1),make_float3(.8f,.2f,1),
        normalize(make_float3(-1,-1,0)),mirror,2,&path),"two-face mirror stationary route solved");
  check(len(path.point[0]-make_float3(0,.6f,1))<1e-5f && len(path.point[1]-make_float3(.6f,0,1))<1e-5f,
        "two-face mirror matches independent unfolded route");
  mirror[0]={make_float3(.3f,0,0),make_float3(0,1,0),make_float3(0,0,-1),1,1,1,1.5f,1.5f,1,1};
  mirror[1]={make_float3(.6f,0,0),make_float3(0,1,0),make_float3(0,0,1),1,1,1.5f,1,1,1,-1};
  check(coherent_geometry_connect(zero_float3(),make_float3(1,0,0),make_float3(-1,0,0),mirror,2,&path),
        "same-object slab's geometric path solved");
  check(fabsf(path.optical_length-1.15f)<1e-6f,"normal slab independent optical length");
  std::printf("checks=%d failures=%d\n",checks,failures);return failures!=0;
}
