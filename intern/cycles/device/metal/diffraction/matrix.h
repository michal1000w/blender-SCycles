/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#ifdef DIFFRACTION_MPS_SOLVE
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif
#include <algorithm>
#include <cstring>
#include <cmath>
#include <map>
#include <string>
#include <stdexcept>
#include <vector>
#include <functional>

#ifndef DIFFRACTION_PENDING_COMMAND_LIMIT
#define DIFFRACTION_PENDING_COMMAND_LIMIT 16
#endif
static_assert(DIFFRACTION_PENDING_COMMAND_LIMIT > 0);

/* Experimental resident matrix operations. All payload readback is explicit.
 * Completion checks validate every submitted command and solve status before
 * allowing results to leave the engine. No CPU arithmetic fallback. */
class DiffractionResidentMatrix {
 public:
  struct Matrix { id<MTLBuffer> buffer; unsigned rows,cols; };
  id<MTLDevice> device;
  NSUInteger uploads=0,downloads=0,scalar_downloads=0;
  NSUInteger sampled_peak_device_bytes=0;
  bool profiling=false;
  unsigned pending_command_limit=DIFFRACTION_PENDING_COMMAND_LIMIT;
  /* Set only while the engine is idle. Queries run on its serialized caller. */
  std::function<bool()> cancelled;
  size_t allocation_bytes=0,allocation_count=0;
  std::map<std::string,std::pair<size_t,double>> metrics;
  explicit DiffractionResidentMatrix(NSString *source)
      : DiffractionResidentMatrix(source, MTLCreateSystemDefaultDevice()) {}
  /* Integration callers must pass the renderer's selected device. A missing
   * selection is an error, never a request to silently switch GPUs. */
  DiffractionResidentMatrix(NSString *source, id<MTLDevice> selected_device) {
    device=selected_device;
    if (!device) throw std::runtime_error("Metal device unavailable");
    queue_=[device newCommandQueue];
    if (!queue_) throw std::runtime_error("Metal command queue unavailable");
    NSError *error=nil;
    MTLCompileOptions *options=[MTLCompileOptions new];
    if (@available(macOS 15.0, *)) {
      options.mathMode=MTLMathModeSafe;
    }
    else {
      throw std::runtime_error("Metal diffraction cache solver requires macOS 15 or newer");
    }
    id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
    if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
    pipelines_=[NSMutableDictionary dictionary];
    for (NSString *name in @[@"rectangular_product",@"rectangular_residual",@"complex_solve",@"matrix_join",@"matrix_solution",@"matrix_add",@"matrix_scale",@"matrix_identity",@"matrix_slice",@"matrix_block",@"matrix_norm",@"matrix_material",@"matrix_retained"]) {
      id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];
      if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
      pipelines_[name]=pipeline;
    }
    pending_=[NSMutableArray array];statuses_=[NSMutableArray array];
#ifdef DIFFRACTION_MPS_SOLVE
    mps_statuses_=[NSMutableArray array];
    for (NSString *name in @[@"matrix_real_embedding",@"matrix_real_rhs",@"matrix_complex_solution",@"matrix_finite"]) {
      pipelines_[name]=[device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];
      if(!pipelines_[name])throw std::runtime_error("MPS conversion pipeline failed");
    }
#endif
  }
  Matrix upload(const void *values,unsigned rows,unsigned cols) {
    Matrix m=allocate(rows,cols);
    memcpy(m.buffer.contents,values,rows*cols*2*sizeof(float));++uploads;
    return m;
  }
  Matrix multiply(Matrix a,Matrix b) { return product(a,b,nullptr); }
  Matrix residual(Matrix a,Matrix x,Matrix b) { return product(a,x,&b); }
  Matrix add(Matrix a,Matrix b) {
    if (a.rows!=b.rows||a.cols!=b.cols) throw std::runtime_error("Matrix add shape mismatch");
    Matrix out=allocate(a.rows,a.cols);
    encode(@"matrix_add",@[a.buffer,b.buffer,out.buffer],{a.rows,a.cols,0,0},a.rows*a.cols,false);
    return out;
  }
  Matrix material(unsigned rows,unsigned cols,unsigned kind,const void *config,size_t bytes) {
    Matrix out=allocate(rows,cols);
    id<MTLCommandBuffer> command=make_command(@"matrix_material");
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    id<MTLComputePipelineState> pipeline=pipelines_[@"matrix_material"];
    [encoder setComputePipelineState:pipeline];[encoder setBuffer:out.buffer offset:0 atIndex:0];
    [encoder setBytes:config length:bytes atIndex:1];[encoder setBytes:&kind length:sizeof(kind) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(rows*cols,1,1) threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth,1,1)];
    [encoder endEncoding];submit(command);return out;
  }
  Matrix retained(Matrix full,unsigned channels,const void *config,size_t bytes) {
    Matrix out=allocate(channels,channels);
    id<MTLCommandBuffer> command=make_command(@"matrix_retained");
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    id<MTLComputePipelineState> pipeline=pipelines_[@"matrix_retained"];
    [encoder setComputePipelineState:pipeline];[encoder setBuffer:full.buffer offset:0 atIndex:0];
    [encoder setBuffer:out.buffer offset:0 atIndex:1];[encoder setBytes:config length:bytes atIndex:2];
    [encoder setBytes:&channels length:sizeof(channels) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(channels*channels,1,1) threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth,1,1)];
    [encoder endEncoding];submit(command);return out;
  }
  Matrix join(Matrix a,Matrix b) {
    if(a.rows!=a.cols||a.rows!=b.rows)throw std::runtime_error("Invalid horizontal matrix join");
    Matrix out=allocate(a.rows,a.cols+b.cols);
    encode(@"matrix_join",@[a.buffer,b.buffer,out.buffer],{a.rows,b.cols,0,0},out.rows*out.cols,false);
    return out;
  }
  Matrix identity(unsigned n) {
    Matrix out=allocate(n,n);
    encode(@"matrix_identity",@[out.buffer],{n,0,0,0},n*n,false);return out;
  }
  Matrix scale(Matrix a,float real,float imag=0) {
    Matrix out=allocate(a.rows,a.cols);Shape shape{a.rows,a.cols,0,0};
    memcpy(&shape.z,&real,sizeof(float));memcpy(&shape.w,&imag,sizeof(float));
    encode(@"matrix_scale",@[a.buffer,out.buffer],shape,a.rows*a.cols,false);return out;
  }
  Matrix slice(Matrix a,unsigned row,unsigned col,unsigned rows,unsigned cols) {
    if(row+rows>a.rows||col+cols>a.cols)throw std::runtime_error("Invalid matrix slice");
    Matrix out=allocate(rows,cols);
    encode(@"matrix_slice",@[a.buffer,out.buffer],{a.cols,row,col,cols},rows*cols,false);return out;
  }
  Matrix block(Matrix a,Matrix b,Matrix c,Matrix d) {
    if(a.rows!=b.rows||c.rows!=d.rows||a.cols!=c.cols||b.cols!=d.cols)
      throw std::runtime_error("Invalid matrix block dimensions");
    Matrix out=allocate(a.rows+c.rows,a.cols+b.cols);
    encode(@"matrix_block",@[a.buffer,b.buffer,c.buffer,d.buffer,out.buffer],
           {a.rows,a.cols,b.cols,c.rows},out.rows*out.cols,false);return out;
  }
  float norm1(Matrix a) {
    id<MTLBuffer> result=[device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
    encode(@"matrix_norm",@[a.buffer,result],{a.rows,a.cols,0,0},1,false);
    synchronize();++scalar_downloads;
    float norm=*static_cast<float *>(result.contents);
    if(!std::isfinite(norm))throw std::runtime_error("Nonfinite matrix norm");
    return norm;
  }
  Matrix solve(Matrix a,Matrix b) {
    if (a.rows!=a.cols||a.rows!=b.rows) throw std::runtime_error("Matrix solve shape mismatch");
#ifdef DIFFRACTION_MPS_SOLVE
    return mps_solve(a,b);
#endif
    Matrix workspace=allocate(a.rows,a.cols+b.cols),out=allocate(b.rows,b.cols);
    encode(@"matrix_join",@[a.buffer,b.buffer,workspace.buffer],{a.rows,b.cols,0,0},a.rows*(a.cols+b.cols),false);
    id<MTLBuffer> status=[device newBufferWithLength:sizeof(unsigned) options:MTLResourceStorageModeShared];
    if (!status) throw std::runtime_error("Status allocation failed");
    *static_cast<unsigned *>(status.contents)=0;
    [statuses_ addObject:status];
    id<MTLCommandBuffer> command=make_command(@"complex_solve");
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    id<MTLComputePipelineState> pipeline=pipelines_[@"complex_solve"];
    [encoder setComputePipelineState:pipeline];[encoder setBuffer:workspace.buffer offset:0 atIndex:0];
    [encoder setBuffer:status offset:0 atIndex:1];[encoder setBytes:&a.rows length:sizeof(unsigned) atIndex:2];
    [encoder setBytes:&b.cols length:sizeof(unsigned) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(128,pipeline.maxTotalThreadsPerThreadgroup),1,1)];
    [encoder endEncoding];submit(command);
    encode(@"matrix_solution",@[workspace.buffer,out.buffer],{a.rows,b.cols,0,0},out.rows*out.cols,false);
    return out;
  }
  Matrix refined_solve(Matrix a,Matrix b,unsigned iterations) {
#ifdef DIFFRACTION_MPS_SOLVE
    if(a.rows!=a.cols||a.rows!=b.rows)throw std::runtime_error("Matrix solve shape mismatch");
    auto factor=mps_factor(a);
#ifdef DIFFRACTION_INVERSE_REFINEMENT
    mps_check_finite(b,@"inverse-refinement right-hand side");
    Matrix inverse=mps_apply(factor,identity(a.rows));
    Matrix x=multiply(inverse,b);
    for(unsigned i=0;i<iterations;++i)x=add(x,multiply(inverse,residual(a,x,b)));
    mps_check_finite(x,@"inverse-refinement solution");
#else
    Matrix x=mps_apply(factor,b);
    for(unsigned i=0;i<iterations;++i)x=add(x,mps_apply(factor,residual(a,x,b)));
#endif
#else
    Matrix x=solve(a,b);
    for (unsigned i=0;i<iterations;++i) x=add(x,solve(a,residual(a,x,b)));
#endif
    return x;
  }
  /* Discard a failed query without exposing its output. Submitted work must
   * finish before a pool can admit another exclusive query or reuse this slot. */
  bool discard_pending() {
#ifdef DIFFRACTION_COMMAND_BATCH
    current_=nil;
#endif
    bool completed=true;
    for(id<MTLCommandBuffer> command in pending_) {
      [command waitUntilCompleted];
      completed &= command.status==MTLCommandBufferStatusCompleted;
    }
    [pending_ removeAllObjects];[statuses_ removeAllObjects];
#ifdef DIFFRACTION_MPS_SOLVE
    [mps_statuses_ removeAllObjects];
#endif
    return completed;
  }
  void synchronize() {
    check_cancelled();
#ifdef DIFFRACTION_COMMAND_BATCH
    if(current_) {
      [current_ commit];[pending_ addObject:current_];current_=nil;
    }
#endif
    for (id<MTLCommandBuffer> command in pending_) {
      [command waitUntilCompleted];
      if (command.status!=MTLCommandBufferStatusCompleted) throw std::runtime_error("Metal command failed");
      if(profiling) {
        auto &metric=metrics[command.label.UTF8String];++metric.first;
        metric.second+=1000*(command.GPUEndTime-command.GPUStartTime);
      }
    }
    for (id<MTLBuffer> status in statuses_)
      if (*static_cast<unsigned *>(status.contents)!=1) throw std::runtime_error(std::string("Metal solve failed: ")+(status.label?status.label.UTF8String:"unlabeled"));
#ifdef DIFFRACTION_MPS_SOLVE
    for(id<MTLBuffer> status in mps_statuses_)
      if(*static_cast<int *>(status.contents)!=MPSMatrixDecompositionStatusSuccess)
        throw std::runtime_error("MPS LU decomposition failed");
    [mps_statuses_ removeAllObjects];
#endif
    [pending_ removeAllObjects];[statuses_ removeAllObjects];
  }
  void download(Matrix m,void *values) {
    synchronize();memcpy(values,m.buffer.contents,m.rows*m.cols*2*sizeof(float));++downloads;
  }
 private:
#ifdef DIFFRACTION_MPS_SOLVE
  NSMutableArray<id<MTLBuffer>> *mps_statuses_;
  struct MPSFactor { MPSMatrix *matrix; MPSMatrix *pivots; unsigned n; };
  MPSMatrix *mps_wrap(id<MTLBuffer> buffer,unsigned rows,unsigned cols,MPSDataType type) {
    return [[MPSMatrix alloc] initWithBuffer:buffer descriptor:
      [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols rowBytes:cols*sizeof(float) dataType:type]];
  }
  void mps_check_finite(Matrix a,NSString *role) {
    id<MTLBuffer> status=[device newBufferWithLength:sizeof(unsigned) options:MTLResourceStorageModeShared];
    if(!status)throw std::runtime_error("Finite status allocation failed");
    status.label=[NSString stringWithFormat:@"%@ %ux%u submission %lu",role,a.rows,a.cols,(unsigned long)pending_.count];
    *static_cast<unsigned *>(status.contents)=1;[statuses_ addObject:status];
    encode(@"matrix_finite",@[a.buffer,status],{a.rows,a.cols,0,0},a.rows*a.cols,false);
  }
  MPSFactor mps_factor(Matrix a) {
    mps_check_finite(a,@"coefficient");
    const unsigned n=a.rows;
    // Matrix storage is complex-sized; the real embedding uses the same bytes.
    Matrix ar=allocate(2*n,n);
    encode(@"matrix_real_embedding",@[a.buffer,ar.buffer],{n,0,0,0},4*n*n,false);
    MPSMatrix *ma=mps_wrap(ar.buffer,2*n,2*n,MPSDataTypeFloat32);
    id<MTLBuffer> pivots=[device newBufferWithLength:2*n*sizeof(unsigned) options:MTLResourceStorageModeShared];
    MPSMatrix *mp=mps_wrap(pivots,1,2*n,MPSDataTypeUInt32);
    id<MTLBuffer> status=[device newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
    if(!pivots||!status)throw std::runtime_error("MPS factor allocation failed");
    *static_cast<int *>(status.contents)=-99;[mps_statuses_ addObject:status];
    MPSMatrixDecompositionLU *lu=[[MPSMatrixDecompositionLU alloc] initWithDevice:device rows:2*n columns:2*n];
    id<MTLCommandBuffer> command=make_command(@"mps_lu_factor");
    [lu encodeToCommandBuffer:command sourceMatrix:ma resultMatrix:ma pivotIndices:mp status:status];
    submit(command);
    return {ma,mp,n};
  }
  Matrix mps_apply(const MPSFactor &factor,Matrix b) {
    mps_check_finite(b,@"right-hand side");
    const unsigned n=factor.n,r=b.cols;
    Matrix br=allocate(n,r),xr=allocate(n,r),out=allocate(n,r);
    encode(@"matrix_real_rhs",@[b.buffer,br.buffer],{n,r,0,0},2*n*r,false);
    MPSMatrix *mb=mps_wrap(br.buffer,2*n,r,MPSDataTypeFloat32);
    MPSMatrix *mx=mps_wrap(xr.buffer,2*n,r,MPSDataTypeFloat32);
    MPSMatrixSolveLU *solver=[[MPSMatrixSolveLU alloc] initWithDevice:device transpose:NO order:2*n numberOfRightHandSides:r];
    id<MTLCommandBuffer> command=make_command(@"mps_lu_apply");
    [solver encodeToCommandBuffer:command sourceMatrix:factor.matrix rightHandSideMatrix:mb pivotIndices:factor.pivots solutionMatrix:mx];
    submit(command);
    encode(@"matrix_complex_solution",@[xr.buffer,out.buffer],{n,r,0,0},n*r,false);
    mps_check_finite(out,@"solution");
    return out;
  }
  Matrix mps_solve(Matrix a,Matrix b) { return mps_apply(mps_factor(a),b); }

#endif
  struct Shape { unsigned x,y,z,w; };
  id<MTLCommandQueue> queue_;
  NSMutableDictionary<NSString *,id<MTLComputePipelineState>> *pipelines_;
  NSMutableArray<id<MTLCommandBuffer>> *pending_;
  NSMutableArray<id<MTLBuffer>> *statuses_;
#ifdef DIFFRACTION_COMMAND_BATCH
  id<MTLCommandBuffer> current_=nil;
#endif
  id<MTLCommandBuffer> make_command(NSString *name) {
    check_cancelled();
#ifdef DIFFRACTION_COMMAND_BATCH
    if(!current_) {current_=[queue_ commandBuffer];current_.label=@"matrix_batch";}
    return current_;
#else
    id<MTLCommandBuffer> command=[queue_ commandBuffer];command.label=name;return command;
#endif
  }
  void check_cancelled() {
    if (!cancelled || !cancelled()) return;
    if(!discard_pending())throw std::runtime_error("Metal command failed while cancelling");
    throw std::runtime_error("Metal diffraction solve cancelled");
  }
  void submit(id<MTLCommandBuffer> command) {
#ifndef DIFFRACTION_COMMAND_BATCH
    [command commit];[pending_ addObject:command];
#ifdef DIFFRACTION_LARGE_ORDERS
    // Bound retained completed-command resources during large layer series.
    if(pending_.count>=pending_command_limit)synchronize();
#endif
#endif
  }
  Matrix allocate(unsigned rows,unsigned cols) {
    if (!rows||!cols||rows>8192||cols>8192) throw std::runtime_error("Invalid matrix dimensions");
    const size_t bytes=size_t(rows)*cols*2*sizeof(float);
    if(bytes>device.maxBufferLength)throw std::runtime_error("Matrix exceeds Metal buffer limit");
#ifdef DIFFRACTION_LARGE_ORDERS
    const uint64_t budget=std::min<uint64_t>(uint64_t(4)*1024*1024*1024,device.recommendedMaxWorkingSetSize*3/4);
    if(device.currentAllocatedSize+bytes>budget)throw std::runtime_error("GPU response exceeds working memory budget");
#endif
    id<MTLBuffer> buffer=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    sampled_peak_device_bytes=std::max(sampled_peak_device_bytes,device.currentAllocatedSize);
    if (!buffer) throw std::runtime_error("Matrix allocation failed");
    if(profiling){++allocation_count;allocation_bytes+=size_t(rows)*cols*2*sizeof(float);}
    return {buffer,rows,cols};
  }
  void encode(NSString *name,NSArray<id<MTLBuffer>> *buffers,Shape shape,unsigned count,bool /*product*/) {
    id<MTLCommandBuffer> command=make_command(name);
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    id<MTLComputePipelineState> pipeline=pipelines_[name];
    [encoder setComputePipelineState:pipeline];
    for (NSUInteger i=0;i<buffers.count;++i)
      [encoder setBuffer:buffers[i] offset:0 atIndex:(i>=3?i+1:i)];
    [encoder setBytes:&shape length:sizeof(shape) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth,1,1)];
    [encoder endEncoding];submit(command);
  }
  Matrix product(Matrix a,Matrix b,Matrix *initial) {
    if(a.cols!=b.rows || (initial&&(initial->rows!=a.rows||initial->cols!=b.cols)))
      throw std::runtime_error("Matrix product shape mismatch");
    Matrix out=allocate(a.rows,b.cols);
    NSArray *buffers=initial?@[a.buffer,b.buffer,out.buffer,initial->buffer]:@[a.buffer,b.buffer,out.buffer];
    encode(initial?@"rectangular_residual":@"rectangular_product",buffers,{a.rows,a.cols,b.cols,0},out.rows*out.cols,true);
    return out;
  }
};
