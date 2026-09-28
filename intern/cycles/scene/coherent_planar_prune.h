/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "scene/coherent_planar_cluster.h"
CCL_NAMESPACE_BEGIN

/* Certified conservative single-reflection reachability for actual flat
 * facets. A detector triangle wholly in the source halfspace projects from the
 * image source to a triangle on the mirror plane. Its overlap with the facet
 * is necessary for a physical specular path. Straddling, grazing and numerical
 * ambiguity always retain the candidate. No smooth-curvature approximation. */
inline bool coherent_planar_reflection_may_reach(
    const CoherentPlanarPoint &source, const CoherentPlanarTriangle &facet,
    const CoherentPlanarTriangle &detector)
{
  using namespace coherent_planar_detail;
  Plane plane_frame;
  if (!plane(facet, plane_frame)) return true;
  const auto n = plane_frame.cross_raw;
  const double nn = dot(n,n);
  auto signed_distance = [&](const CoherentPlanarPoint &point) {
    return dot(sub(point,plane_frame.origin),n);
  };
  auto distance_error = [&](const CoherentPlanarPoint &point) {
    const auto delta=sub(point,plane_frame.origin);
    double scale=0;
    for(int k=0;k<3;k++) scale+=std::abs(delta[k]*n[k]);
    return 128*std::numeric_limits<double>::epsilon()*std::max(scale, std::numeric_limits<double>::min());
  };
  const double ds=signed_distance(source);
  if(!std::isfinite(ds)||std::abs(ds)<=distance_error(source)) return true;
  int same=0,opposite=0;
  for(const auto &r:detector.point) {
    const double d=signed_distance(r),error=distance_error(r);
    if(!std::isfinite(d)||std::abs(d)<=error) return true;
    if((d>0)==(ds>0)) same++;else opposite++;
  }
  if(opposite==3) return false;
  if(same!=3) return true; /* Conservative across the plane/horizon. */
  CoherentPlanarPoint image;
  for(int k=0;k<3;k++) image[k]=source[k]-2*ds/nn*n[k];
  std::array<std::array<double,2>,3> a{},b{};
  for(int i=0;i<3;i++) {
    const auto delta=sub(detector.point[i],image);
    const double denominator=dot(delta,n);
    if(!std::isfinite(denominator)||std::abs(denominator)<=distance_error(detector.point[i])+distance_error(source)) return true;
    const double t=dot(sub(plane_frame.origin,image),n)/denominator;
    if(!(t>0&&t<1)||!std::isfinite(t))return true;
    CoherentPlanarPoint hit;
    for(int k=0;k<3;k++) hit[k]=image[k]+t*delta[k];
    const auto f=sub(facet.point[i],plane_frame.origin),p=sub(hit,plane_frame.origin);
    a[i]={dot(f,plane_frame.u),dot(f,plane_frame.v)};
    b[i]={dot(p,plane_frame.u),dot(p,plane_frame.v)};
    if(!std::isfinite(b[i][0]+b[i][1]))return true;
  }
  /* Triangle SAT. Equality and uncertainty preserve edge/vertex histories;
   * BVH membership and the physical connector decide them at shading time. */
  for(const auto *polygon:{&a,&b}) for(int edge=0;edge<3;edge++) {
    const auto &p=(*polygon)[edge],&q=(*polygon)[(edge+1)%3];
    const std::array<double,2> axis={p[1]-q[1],q[0]-p[0]};
    double amin=std::numeric_limits<double>::infinity(),amax=-amin,bmin=amin,bmax=-amin,scale=0;
    for(int i=0;i<3;i++) {
      const double av=a[i][0]*axis[0]+a[i][1]*axis[1],bv=b[i][0]*axis[0]+b[i][1]*axis[1];
      amin=std::min(amin,av);amax=std::max(amax,av);bmin=std::min(bmin,bv);bmax=std::max(bmax,bv);
      scale=std::max(scale,std::abs(a[i][0]*axis[0])+std::abs(a[i][1]*axis[1]));
      scale=std::max(scale,std::abs(b[i][0]*axis[0])+std::abs(b[i][1]*axis[1]));
    }
    const double error=512*std::numeric_limits<double>::epsilon()*std::max(scale,std::numeric_limits<double>::min());
    if(amax+error<bmin||bmax+error<amin)return false;
  }
  return true;
}
CCL_NAMESPACE_END
