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
int main() {
 KernelCoherentPatch patches[80]{}; KernelLight light{}; light.object_id=500;
 KernelCoherentCandidate candidate{}; candidate.light=0;
 PatchTestGlobals globals{}; globals.coherent_patches=patches;
 globals.coherent_candidates=&candidate; globals.lights=&light;
 globals.data.integrator.coherent_transport_mode=1;
 globals.data.integrator.coherent_max_interface_events=1;
 globals.data.integrator.coherent_patch_count=80;
 globals.data.integrator.coherent_candidate_count=1;
 KernelGlobals kg=&globals;
 for(int i=0;i<80;i++) {
  patches[i].object=100+i; patches[i].shape=2;patches[i].mode=1;
  patches[i].primitive_type=PRIMITIVE_TRIANGLE;
  patches[i].primitive_offset=1000;patches[i].primitive_count=200;
  check(coherent_patch_for_hit(kg,100+i,1079,PRIMITIVE_TRIANGLE)==i,"80 object-qualified instance ranges");
  check(coherent_patch_for_hit(kg,100+i,999,PRIMITIVE_TRIANGLE)==-1,"before range unowned");
  check(coherent_patch_for_hit(kg,100+i,1200,PRIMITIVE_TRIANGLE)==-1,"after range unowned");
  check(coherent_patch_for_hit(kg,100+i,1079,PRIMITIVE_POINT)==-1,"point type does not alias facet");
 }
 auto h=coherent_history_begin();
 check(coherent_history_owned_candidate(kg,h,500),"direct prefix owned");
 check(!coherent_history_owned_candidate(kg,h,501),"different source unowned");
 h=coherent_history_stream_after_scatter(h,true,LABEL_REFLECT|LABEL_SINGULAR);
 check(coherent_history_owned_candidate(kg,h,500),"single reflected prefix owned beyond patch63");
 check(!coherent_history_valid(coherent_history_stream_after_scatter(h,true,LABEL_REFLECT|LABEL_SINGULAR)),"second reflection native");
 check(!coherent_history_valid(coherent_history_stream_after_scatter(coherent_history_begin(),false,LABEL_REFLECT|LABEL_SINGULAR)),"unmarked reflection native");
 check(!coherent_history_valid(coherent_history_stream_after_scatter(coherent_history_begin(),true,LABEL_REFLECT|LABEL_DIFFUSE)),"diffuse prefix native");
 check(!coherent_history_valid(coherent_history_stream_after_scatter(coherent_history_begin(),true,LABEL_TRANSMIT|LABEL_SINGULAR)),"transmission prefix native");
 check(coherent_history_candidate_within_budget(&candidate,0,0,0,3,2,1,false,2,2,1),"single R budget admitted");
 check(!coherent_history_candidate_within_budget(&candidate,2,0,0,3,2,1,false,2,3,1),"remaining camera suffix budget enforced");
 check(coherent_history_candidate_within_budget(&candidate,2,0,0,3,2,1,false,2,3,0),"local direct budget admitted");
 globals.data.integrator.coherent_max_interface_events=2;
 h=coherent_history_stream_after_scatter(h,true,LABEL_REFLECT|LABEL_SINGULAR,2);
 check(coherent_history_valid(h)&&coherent_history_count(h)==2,"two R history retained");
 check(coherent_history_owned_candidate(kg,h,500),"two R prefix owned at detector");
 check(!coherent_history_valid(coherent_history_stream_after_scatter(h,true,LABEL_REFLECT|LABEL_SINGULAR,2)),"third R remains native");
 check(coherent_history_candidate_within_budget(&candidate,0,0,0,4,3,1,true,4,3,2),"two R exact boundary admitted");
 check(!coherent_history_candidate_within_budget(&candidate,1,0,0,3,3,1,true,4,3,2),"two R camera suffix total budget");
 check(!coherent_history_candidate_within_budget(&candidate,0,1,0,4,3,1,true,4,3,2),"two R glossy suffix budget");
 check(!coherent_history_candidate_within_budget(&candidate,0,0,0,4,3,1,true,2,3,2),"two R BDPT depth budget");
 check(!coherent_history_candidate_within_budget(&candidate,0,0,0,4,3,1,true,4,1,2),"two R source depth budget");
 printf("{\"checks\":%d,\"failures\":%d}\n",checks,failures);return failures?1:0;
}
