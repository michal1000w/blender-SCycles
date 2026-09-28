/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "reference_backend.h"
#include <array>
#include <chrono>
#include <condition_variable>

/* A per-build pool: two small queries or one exclusive large query. Waiting
 * large queries take priority so a stream of small queries cannot starve them.
 * Source, device and cancellation remain alive for the lifetime of the pool. */
class DiffractionMetalReferencePool {
 public:
  DiffractionMetalReferencePool(NSString *source, id<MTLDevice> device,
                               std::function<bool()> cancelled = {},
                               std::function<void()> initialized = {})
      : source_(source), device_(device), cancelled_(std::move(cancelled)),
        initialized_(std::move(initialized))
  {
    if (!device_) throw std::invalid_argument("Missing Metal diffraction device");
  }

  struct SchedulingState {
    unsigned active, waiting;
    bool exclusive;
  };
  SchedulingState scheduling_state()
  {
    std::lock_guard lock(mutex_);
    return {unsigned(busy_[0]) + unsigned(busy_[1]), waiting_, exclusive_};
  }

  bool solve(const ccl::DiffractionGratingProfile &profile, double wavelength,
             double kx, double ky, int n, int retained,
             ccl::DiffractionGratingBlock &out, std::string &error)
  {
    out = {};
    const bool large = n > 32;
    size_t slot = 0;
    {
      std::unique_lock lock(mutex_);
      ++waiting_;
      if (large) ++waiting_large_;
      while (true) {
        if (cancelled_ && cancelled_()) {
          --waiting_;
          if (large) --waiting_large_;
          condition_.notify_all();
          error = "Metal diffraction pool wait cancelled";
          return false;
        }
        const bool available = large ? !busy_[0] && !busy_[1] :
            !exclusive_ && !waiting_large_ && (!busy_[0] || !busy_[1]);
        if (available) break;
        condition_.wait_for(lock, std::chrono::milliseconds(25));
      }
      if (large) { --waiting_large_; exclusive_ = true; }
      --waiting_;
      slot = busy_[0] ? 1 : 0;
      busy_[slot] = true;
    }
    struct Release {
      DiffractionMetalReferencePool &pool;
      size_t slot;
      bool large;
      ~Release() {
        std::lock_guard lock(pool.mutex_);
        pool.busy_[slot] = false;
        if (large) pool.exclusive_ = false;
        pool.condition_.notify_all();
      }
    } release{*this, slot, large};
    @autoreleasepool {
      try {
        // Only the checked-out caller can access this slot.
        if (!initialization_errors_[slot].empty()) {
          error = initialization_errors_[slot];
          return false;
        }
        if (!backends_[slot]) {
          try {
            backends_[slot] = std::make_unique<DiffractionMetalReferenceBackend>(source_, device_);
            backends_[slot]->engine.cancelled = cancelled_;
            if (initialized_) initialized_();
          }
          catch (const std::exception &exception) {
            initialization_errors_[slot] = exception.what();
            error = initialization_errors_[slot];
            return false;
          }
        }
        return backends_[slot]->solve(profile, wavelength, kx, ky, n, retained, out, error);
      }
      catch (const std::exception &exception) {
        error = exception.what();
        return false;
      }
    }
  }

 private:
  NSString *source_;
  id<MTLDevice> device_;
  std::function<bool()> cancelled_;
  std::function<void()> initialized_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::array<bool, 2> busy_{};
  bool exclusive_ = false;
  unsigned waiting_large_ = 0;
  unsigned waiting_ = 0;
  std::array<std::unique_ptr<DiffractionMetalReferenceBackend>, 2> backends_;
  std::array<std::string, 2> initialization_errors_;
};
