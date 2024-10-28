/*      
    *** cudaSlice.h ***   

------------------------------------------------------------------------
Copyright 2018-2024 Earl J. Kirkland


This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

---------------------- NO WARRANTY ------------------
THIS PROGRAM IS PROVIDED AS-IS WITH ABSOLUTELY NO WARRANTY
OR GUARANTEE OF ANY KIND, EITHER EXPRESSED OR IMPLIED,
INCLUDING BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
IN NO EVENT SHALL THE AUTHOR BE LIABLE
FOR DAMAGES RESULTING FROM THE USE OR INABILITY TO USE THIS
PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA
BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR
THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH
ANY OTHER PROGRAM). 

-----------------------------------------------------------------------------

  cuda subroutines:

	cmplPixMul()     : complex pix mul with shift
	cmplVecMul()     : complex vector mul
	cuAtompot()      : calculate atomic potential of one slice
	cuBWlimit()      : bandwidth limit
	cuFreq()         : calculate FFT frequencies
	cuPhasegrating() : phase grating
	integCBED()      : integrate ADF detector
	magSqPix()       : form sq. magnitude of pix
	probeShift()     : shift probe in FFT space
	zeroDbleArray()  : set double array to zero

  started from autostem_cuda.cu 18-aug-2018 ejk
  add probeShift(), zeroDbleArray(), integCBED() 18-aug-2018 ejk
  add cuAtompot(), cuBWlimit(), cuPhasegrating()  27-sep-2018 ejk 
  add full cache (for all NZMAX) to cuAtompot() 15-sep-2019 ejk
  start reorganize for nvcc/msvs2019 4-sep-2021 ejk
  add COMx,y mode 23-jul-2022 ejk
  update comments 11-jul-2023 ejk
  change sign in cuAtomplot  30-mar-2024 ejk

    this file is formatted for a TAB size of 4 characters 
*/


#ifndef CUDASUBS_INCLUDED
#define CUDASUBS_INCLUDED

#ifdef foofoo
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cufft.h>
#endif

#include "cuda.h"
#include "cuda_device_runtime_api.h"
#include "cuda_runtime.h"
#include "cufft.h"


//==================  extra CUDA stuff ===================


//---------------  cmplPixMul() --------------
//
// CUDA kernel definition for 2D pix mul with trans larger than probe
//     probe is mul. by a subset of trans
//  perform operation probe = probe * trans with offset
//  probe = nxprobe x nyprobe (no bigger than trans)
//  trans = nx x ny  (may be bigger than probe)
//  ixoff, iyoff = offset of probe inside trans ; edges will wrap around
//
__global__ void cmplPixMul(cufftComplex* trans, cufftComplex* probe, int nx, int ny,
    int nxprobe, int nyprobe, int ixoff, int iyoff);

//---------------  cmplVecMul() --------------
//
// cuda kernel definition for complex vector mul
//   c = a * b (element by element)
//
__global__ void cmplVecMul(cufftComplex* a, cufftComplex* b, cufftComplex* c, int nmax);

/*---------------  cuAtompot() --------------

  CUDA kernel definition to calculate single layer projected atomic potential

  this is actually no faster than doing the potential in real space on the host
	but save anyway just in case!

  calculate the summation over atoms at one point (kx,ky) in reciprocal space
  
  It is better to sum in reciprocal space to use fine-grain parallelism
  as on a GPU. Every point can then run in parallel without trying to
  access the same point. This is very different than the openMP version
  of trlayer() in autostem.cpp which summs the atomic potential in real space.

  potn[] = nx x ny  output array = half of complex plane for C2R FFT

  x[],y[] = real array of atomic coordinates
  occ[]   = real array of occupancies
  Znum[]  = array of atomic numbers

  spec[ k + 4*iatom] = packed array of x,y,occ,Znum (min. GPU transfers)
						(k=0,1,2,3 for x,y,occ,Znum)
  istart  = starting index of atom coord.
  natom   = number of atoms
  ax, by  = size of transmission function in Angstroms
  kev     = beam energy in keV
  trans   = 2D array to get complex specimen
        transmission function
  nx, ny  = dimensions of transmission functions
  *phirms = average phase shift of projected atomic potential
  *nbeams = will get number of Fourier coefficients
  k2max   = square of max k = bandwidth limit
  fparams[] = scattering factor parameters

  scale = mm0 * wavelength (put here for comparison to original trlayer()

	repeat scaling from mulslice.cpp
		mm0 = 1.0F + v0/511.0F;
		wavlen = (float) wavelength( v0 );
		scale = wavlen * mm0;

*/

__global__ void cuAtompot(cufftComplex* potn,
    const float spec[], int natom, int istart,
    const float ax, const float by, const float kev,
    const int nx, const int ny,
    float kx[], float ky[], float kx2[], float ky2[],
    const float k2max, double fparams[], const float scale);


/*---------------  cuBWlimit() --------------

  bandwidth limit tran[] - assumed to be in reciprocal space
  and add FFT scale
  
  kx2[],ky2[] = spatial freq. sq.
  k2max = max spatial freq.

*/

__global__ void cuBWlimit(cufftComplex* trans,
    float* kx2, float* ky2, float k2max, const int nx, const int ny);


/*---------------  cuFreq() --------------
//
// cuda kernel definition to calculate spatial freq.
//   
    ko[n]  = real array to get spatial frequencies
    ko2[n] = real array to get k[i]*k[i] 
    nk     = integer number of pixels
    ak     = real full scale size of image in pixels
*/
__global__ void cuFreq(float ko[], float ko2[], int nk, float ak);


/*---------------  cuPhasegrating() --------------

  Start with the atomic potential from cuAtompot() after inv. FFT
  and convert to the transmission function as in a phase grating calculation
  - assume its scaled to a phase already
*/

__global__ void cuPhasegrating(float* potnR, cufftComplex* trans,
    const int nx, const int ny);


//---------------  integCBED() --------------
//
// CUDA kernel definition to integrate STEM detector active regions
//
//  remember:
//      [1] many threads cannot access the same sumation variable at
//           one time so sum along only one direction at a time (into a 1D array)
//				- complete the last sum in 1D on the host
//      [2] many points will not be on the active portion of the detector
//             so there is less competition among threads than it might seem
//
//  cbed = input nx x ny float CBED pix = |cpix|^2
//  sums = oout double[nx]  to get sum |cpix|^2 along iy
//  nx, ny = size of cbed
//  collectorMode = detector type
//  kxp[],kyp[] = spatial freq.
//  kxp2[],kyp2[] = spatial freq. sq.
//  k2min, k2max = detector range in polar direction
//  phimin, phimax = detector range in azimuthal direction
//
__global__ void integCBED(double* sums, float* cbed, int nx, int ny,
    int collectorMode, float* kxp, float* kyp, float* kxp2, float* kyp2,
    float k2min, float k2max, float phiMin, float phiMax);

//---------------  magSqPix() --------------
//
// CUDA kernel definition for 2D pix complex to magnitude
//     take square magnitude on GPU to
// 
//  cpix = nx x ny complex
//  fpix = nx x ny  float = |cpix|^2
//  nx, ny = size of both pix
//
__global__ void magSqPix(float* fpix, cufftComplex* cpix, int nx, int ny);


//---------------  probeShift() --------------
//
//  CUDA kernel definition for 2D probe shift in FT space
//  perform operation probe *= exp( 2*pi*i * x * k) with offset
//
//  prb0 = input nx x ny complex
//  prbs = output gets prb0 shifted by (xs,ys)
//  xs,ys = amount to shift
//  nx, ny = size of both pix
//  kx[], ky[] = arrays of spatial frequencies
//
__global__ void probeShift(cufftComplex* prbs, cufftComplex* prb0, int nx, int ny,
    float xs, float ys, float* kx, float* ky);



//---------------  zeroDbleArray() --------------
//
// CUDA kernel definition to zero a double array
// 
//  a[nmax] = double array
//  nmax = size of array
//
__global__ void zeroDbleArray(double* a, int nmax);


#endif
