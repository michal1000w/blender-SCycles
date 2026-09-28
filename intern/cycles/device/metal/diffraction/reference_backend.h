/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "response.h"
#include "scene/diffraction.h"
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <iomanip>

/* Serialized backend with shared ownership by reference-solver callbacks.
 * Shader source and the renderer-selected device are explicit inputs.
 * Used by the Metal device cache builder and standalone validation tools. */
class DiffractionMetalReferenceBackend {
 public:
  DiffractionResidentMatrix engine;
  DiffractionMetalReferenceBackend(NSString *source, id<MTLDevice> device)
      : engine(source, device) {}

  bool solve(const ccl::DiffractionGratingProfile &profile, double wavelength,
             double kx, double ky, int n, int retained,
             ccl::DiffractionGratingBlock &out, std::string &message)
  {
    std::lock_guard lock(mutex_);
        @autoreleasepool {
          out={};message.clear();
          try {
            if(n<1||unsigned(n)>diffraction_resident_max_orders||retained<0||retained>n)throw std::runtime_error("Unsupported GPU order window");
            std::complex<double> ridge,groove,substrate;
            if(!ccl::diffraction_grating_sample_indices(profile,wavelength,ridge,groove,substrate,message))return false;
            DiffractionResidentProfile p{unsigned(n),unsigned(retained),float(profile.pitch),float(wavelength),float(profile.depth),float(profile.duty),float(profile.incident_ior),float(ridge.real()),float(ridge.imag()),float(groove.real()),float(groove.imag()),float(substrate.real()),float(substrate.imag()),float(kx),float(ky)};
            unsigned steps=0;auto matrix=diffraction_resident_response(engine,p,steps);
            std::vector<Pair> data(matrix.rows*matrix.cols);engine.download(matrix,data.data());
            ccl::DiffractionGratingBlock result;
            for(int side=0;side<(substrate.imag()==0?2:1);++side)
              for(int order=-retained;order<=retained;++order)result.ports.push_back({order,bool(side)});
            if(matrix.rows!=2*result.ports.size()||matrix.cols!=matrix.rows)throw std::runtime_error("GPU port dimensions disagree");
            for(Pair v:data) {
              if(!std::isfinite(v.x)||!std::isfinite(v.y))throw std::runtime_error("Nonfinite GPU cache response");
              result.matrix.emplace_back(v.x,v.y);
            }
            // Reference-port matrices are not physical flux blocks. These
            // diagnostics are unavailable, rather than fabricated as zero.
            result.boundary_residual=result.minimum_power_gain=result.maximum_power_gain=std::numeric_limits<double>::quiet_NaN();
            if(ridge.imag()==0 && groove.imag()==0 && substrate.imag()==0 &&
               !ccl::diffraction_grating_restore_lossless_reference(result,message)) {
              std::ostringstream context;
              context << std::setprecision(17) << message << "; wavelength=" << wavelength
                      << "; kx=" << kx << "; ky=" << ky << "; half_orders=" << n
                      << "; retained=" << retained;
              message=context.str();return false;
            }
            out=std::move(result);return true;
          }catch(const std::exception &e){
            message=e.what();
            if(!engine.discard_pending())message+="; Metal command failed while draining rejected query";
            return false;
          }
        }
  }
 private:
  struct Pair { float x,y; };
  std::mutex mutex_;
};

inline ccl::DiffractionReferenceSolver diffraction_metal_reference_solver(
    const std::shared_ptr<DiffractionMetalReferenceBackend> &backend)
{
  if (!backend) throw std::invalid_argument("Missing Metal diffraction backend");
  return [backend](const ccl::DiffractionGratingProfile &profile, double wavelength,
                   double kx, double ky, int n, int retained,
                   ccl::DiffractionGratingBlock &out, std::string &message) {
    return backend->solve(profile, wavelength, kx, ky, n, retained, out, message);
  };
}
