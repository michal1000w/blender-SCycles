/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
// Appended to the existing matrix-operation shader for the diagnostic.
kernel void matrix_join(device const float2 *a [[buffer(0)]], device const float2 *b [[buffer(1)]],
                        device float2 *out [[buffer(2)]], constant uint4 &s [[buffer(3)]],
                        uint i [[thread_position_in_grid]])
{
  uint row=i/(s.x+s.y),col=i%(s.x+s.y);
  out[i]=col<s.x?a[row*s.x+col]:b[row*s.y+col-s.x];
}
kernel void matrix_solution(device const float2 *workspace [[buffer(0)]],device float2 *out [[buffer(1)]],
                            constant uint4 &s [[buffer(3)]],uint i [[thread_position_in_grid]])
{
  out[i]=workspace[(i/s.y)*(s.x+s.y)+s.x+i%s.y];
}
kernel void matrix_add(device const float2 *a [[buffer(0)]],device const float2 *b [[buffer(1)]],
                       device float2 *out [[buffer(2)]],constant uint4 &s [[buffer(3)]],
                       uint i [[thread_position_in_grid]])
{
  out[i]=a[i]+b[i];
}
kernel void matrix_identity(device float2 *out [[buffer(0)]],constant uint4 &s [[buffer(3)]],uint i [[thread_position_in_grid]])
{ out[i]=float2(i/s.x==i%s.x?1.0f:0.0f,0); }
kernel void matrix_scale(device const float2 *a [[buffer(0)]],device float2 *out [[buffer(1)]],constant uint4 &s [[buffer(3)]],uint i [[thread_position_in_grid]])
{ float2 z=float2(as_type<float>(s.z),as_type<float>(s.w)),v=a[i];out[i]=float2(z.x*v.x-z.y*v.y,z.x*v.y+z.y*v.x); }
kernel void matrix_slice(device const float2 *a [[buffer(0)]],device float2 *out [[buffer(1)]],constant uint4 &s [[buffer(3)]],uint i [[thread_position_in_grid]])
{ out[i]=a[(s.y+i/s.w)*s.x+s.z+i%s.w]; }
kernel void matrix_block(device const float2 *a [[buffer(0)]],device const float2 *b [[buffer(1)]],device const float2 *c [[buffer(2)]],constant uint4 &s [[buffer(3)]],device const float2 *d [[buffer(4)]],device float2 *out [[buffer(5)]],uint i [[thread_position_in_grid]])
{
  uint row=i/(s.y+s.z),col=i%(s.y+s.z);
  out[i]=row<s.x?(col<s.y?a[row*s.y+col]:b[row*s.z+col-s.y]):
                       (col<s.y?c[(row-s.x)*s.y+col]:d[(row-s.x)*s.z+col-s.y]);
}
kernel void matrix_norm(device const float2 *a [[buffer(0)]],device float *out [[buffer(1)]],constant uint4 &s [[buffer(3)]],uint i [[thread_position_in_grid]])
{
  float maximum=0;
  for(uint col=0;col<s.y;++col) {
    float sum=0;
    for(uint row=0;row<s.x;++row)sum+=length(a[row*s.y+col]);
    maximum=max(maximum,sum);
  }
  out[0]=maximum;
}
struct ResidentProfile {
  uint half_orders,retained;
  float pitch,wavelength,depth,duty,incident,ridge_n,ridge_k,groove_n,groove_k,substrate_n,substrate_k,kx,ky;
};
float2 material_square(float2 a){return float2(a.x*a.x-a.y*a.y,2*a.x*a.y);}
float2 material_root(float2 a)
{
  float radius=length(a);
  float real=sqrt(max(0.0f,.5f*(radius+a.x)));
  float imag=sqrt(max(0.0f,.5f*(radius-a.x)));
  return float2(a.y<0?-real:real,imag);
}
kernel void matrix_material(device float2 *out [[buffer(0)]],constant ResidentProfile &p [[buffer(1)]],
                            constant uint &kind [[buffer(2)]],uint i [[thread_position_in_grid]])
{
  uint count=2*p.half_orders+1,size=kind<3?count:2*count,row=i/size,col=i%size;
  float2 er=material_square(float2(p.ridge_n,p.ridge_k)),eg=material_square(float2(p.groove_n,p.groove_k));
  if(kind<2) {
    int order=int(row)-int(col);
    float f=order==0?p.duty:sin(M_PI_F*float(order)*p.duty)/(M_PI_F*float(order));
    if(kind==1){er=complex_divide(float2(1,0),er);eg=complex_divide(float2(1,0),eg);}
    out[i]=(er-eg)*f+(row==col?eg:float2(0));return;
  }
  if(kind==2){out[i]=float2(row==col?p.kx+(int(row)-int(p.half_orders))*p.wavelength/p.pitch:0,0);return;}
  uint r=row%count,c=col%count;
  if(r!=c){out[i]=float2(0);return;}
  bool retained=abs(int(r)-int(p.half_orders))<=int(p.retained);
  if(retained&&(kind==3||(kind==4&&p.substrate_k==0))) {
    out[i]=float2(row<count?(col<count?0:-1):(col<count?1:0),0);return;
  }
  float kx=p.kx+(int(r)-int(p.half_orders))*p.wavelength/p.pitch;
  float2 epsilon=kind==3?float2(p.incident*p.incident,0):
                 kind==4?material_square(float2(p.substrate_n,p.substrate_k)):
                         p.duty*er+(1-p.duty)*eg;
  // The internal admittance is a change of basis, not a physical medium.
  // Keep its lossless reference away from modal cutoffs; physical epsilon
  // assembly (kinds 0/1) and exterior admittances (3/4) remain unchanged.
  if(kind>=5 && epsilon.y==0) epsilon.y=0.25f;
  float2 z=material_root(epsilon-float2(kx*kx+p.ky*p.ky,0));
  float2 numerator=row<count?(col<count?float2(-kx*p.ky,0):float2(kx*kx,0)-epsilon):
                              (col<count?epsilon-float2(p.ky*p.ky,0):float2(kx*p.ky,0));
  float2 admittance=complex_divide(numerator,z);
  if(kind==6) {
    // Invert the actually rounded 2x2 block, not its ideal determinant epsilon.
    // Trace is exactly zero; accumulate det(Y)=-a*a-b*c with FMA residuals.
    float2 a=complex_divide(float2(-kx*p.ky,0),z);
    float2 b=complex_divide(float2(kx*kx,0)-epsilon,z);
    float2 c=complex_divide(epsilon-float2(p.ky*p.ky,0),z);
    float2 re(0),im(0);
    re=accumulate_product(re,-a.x,a.x);re=accumulate_product(re,a.y,a.y);
    re=accumulate_product(re,-b.x,c.x);re=accumulate_product(re,b.y,c.y);
    im=accumulate_product(im,-a.x,a.y);im=accumulate_product(im,-a.y,a.x);
    im=accumulate_product(im,-b.x,c.y);im=accumulate_product(im,-b.y,c.x);
    out[i]=complex_divide(-admittance,float2(re.x+re.y,im.x+im.y));
  }
  else out[i]=admittance;
}
kernel void matrix_retained(device const float2 *full [[buffer(0)]],device float2 *out [[buffer(1)]],
                            constant ResidentProfile &p [[buffer(2)]],constant uint &channels [[buffer(3)]],uint i [[thread_position_in_grid]])
{
  uint count=2*p.half_orders+1,m=2*count,side_channels=2*(2*p.retained+1);
  uint row=i/channels,col=i%channels;
  uint r=(row/side_channels)*m+((row%side_channels)%2)*count+(row%side_channels)/2+p.half_orders-p.retained;
  uint c=(col/side_channels)*m+((col%side_channels)%2)*count+(col%side_channels)/2+p.half_orders-p.retained;
  out[i]=full[r*(2*m)+c];
}
