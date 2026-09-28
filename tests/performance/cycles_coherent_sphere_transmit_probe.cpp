/* SPDX-License-Identifier: Apache-2.0 */
#include "kernel/light/coherent_sphere_transmit_geometry.h"
#include <cstdio>
using namespace ccl;
int main()
{
  float sx, sy, sz, rx, ry, rz, cx, cy, cz, radius, ior, nx, ny, nz;
  while (scanf("%f%f%f%f%f%f%f%f%f%f%f%f%f%f",
               &sx,
               &sy,
               &sz,
               &rx,
               &ry,
               &rz,
               &cx,
               &cy,
               &cz,
               &radius,
               &ior,
               &nx,
               &ny,
               &nz) == 14)
  {
    CoherentSphereTTInventory out{};
    const auto status = coherent_sphere_transmit_inventory(make_float3(sx, sy, sz),
                                                           make_float3(rx, ry, rz),
                                                           make_float3(nx, ny, nz),
                                                           make_float3(cx, cy, cz),
                                                           radius,
                                                           ior,
                                                           &out);
    if (status == COHERENT_SPHERE_TT_OK) {
      for (int k = 0; k < out.count; k++) {
        CoherentSphereTTInventory selected{};
        const auto selected_status = coherent_sphere_transmit_inventory(
            make_float3(sx, sy, sz), make_float3(rx, ry, rz), make_float3(nx, ny, nz),
            make_float3(cx, cy, cz), radius, ior, &selected, k);
        const auto &a = out.path[k], &b = selected.path[k];
        bool same = selected_status == status && selected.count == out.count &&
                    selected.angular_momentum[k] == out.angular_momentum[k] &&
                    selected.morse_index[k] == out.morse_index[k] &&
                    selected.maslov_phase_cycles[k] == out.maslov_phase_cycles[k] &&
                    fabsf(a.spreading - b.spreading) <= 2e-6f * max(a.spreading, b.spreading) &&
                    std::abs((double(a.optical_length_split.x) + a.optical_length_split.y) -
                             (double(b.optical_length_split.x) + b.optical_length_split.y)) < 5e-11 * radius;
        for (int j = 0; j < 2; j++) same &= len(a.point[j] - b.point[j]) < 2e-6f * radius;
        if (!same) { std::fprintf(stderr,"selected/full TT mismatch branch %d spread %.9g %.9g OPL %.17g %.17g point %.9g %.9g\n",k,a.spreading,b.spreading,double(a.optical_length_split.x)+a.optical_length_split.y,double(b.optical_length_split.x)+b.optical_length_split.y,a.point[0].x,b.point[0].x); return 1; }
      }
    }
    printf("%d %d", int(status), out.count);
    for (int i = 0; i < out.count; i++) {
      const auto &p = out.path[i];
      printf(" %.9g %.9g %.17g %d",
             out.angular_momentum[i],
             p.spreading,
             double(p.optical_length_split.x) + p.optical_length_split.y,
             out.morse_index[i]);
      for (int j = 0; j < 2; j++)
        printf(" %.9g %.9g %.9g", p.point[j].x, p.point[j].y, p.point[j].z);
    }
    printf("\n");
  }
}
