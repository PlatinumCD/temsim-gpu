/*              *** rfpix.hpp ***

 ------------------------------------------------------------------------
Copyright 2024 Earl J. Kirkland

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

------------------------------------------------------------------------

   C++ class to manage complex floating point images (in ANSI-C++)

   collect (isolate) the FFT details here so it can be easily changed in future

   perform complex to real and its inverse

The source code is formatted for a tab size of 4.

----------------------------------------------------------
The public member functions are:

nx()       : return x size of current image (in pixels)
ny()       : return y size of current image (in pixels)

re(ix,iy)  : return reference to real part of pixel at (ix,iy)
im(ix,iy)  : return reference to imag part of pixel at (ix,iy)

rre(ix,iy) : return reference to pixel at (ix,iy) of the
             real image for complex to real transform

fft()      :  perform FFT
ifft()     : inverse FFT

init()     : perform initialization
copyInit() : copy the initialization

resize()        : resize current in-memory image (current data may be lost)
findRange()     : find range of values in complex image

invert2D( )     : rearrange pix with corners moved to center (for FFT's)

operator*=() 
operator+=()
operator=()

----------------------------------------------------------

   started from cfpix 7-may-2024 E. Kirkland
   working 18-may-2024 ejk
*/

#ifndef RFPIX_HPP   // only include this file if its not already

#define RFPIX_HPP   // remember that this has been included

// define the following symbol to enable bound checking
//     define for debugging, and undefine for final run
//#define RFPIX_BOUNDS_CHECK

#include "fftw3.h"      // FFT routines from FFTW 3
#include <cstdio>
#include <cstdlib>
#include <cmath>        //  prob not needed, but...

#include <string>	// STD string class

//------------------------------------------------------------------
class rfpix{

public:
    
    // constructor functions
    rfpix( int nx=0, int ny=0 );         // blank image
    
    ~rfpix();        //  destructor function

    inline int nx() const { return( nxl ); }
    inline int ny() const { return( nyl ); }

    //   index a single pixel of the real part of the complex pix
    inline float& re( const int ix, const int iy )
    {
        #ifdef RFPIX_BOUNDS_CHECK
            if( (ix<0) || (ix>=nxl) ||
                (iy<0) || (iy>= nycl ) ) {
                sbuff= "out of bounds index in cfpix::re(); size = "
                        +toString(nxl) + " x " + toString(nyl)+ " access = ("
                        +toString(ix)+", " + toString(iy)+")";
                messageRF( sbuff, 2 );
                exit( EXIT_FAILURE );
            }
        #endif
        return data[iy + ix*nycl][0];
    }

    //   index a single pixel of the imaginary part of the complex pix
    inline float& im( const int ix, const int iy )
    {
        #ifdef RFPIX_BOUNDS_CHECK
            if( (ix<0) || (ix>=nxl) ||
                (iy < 0) || (iy >= nycl) ) {
                sbuff= "out of bounds index in rfpix::im(); size = "
                        +toString(nxl) + " x " + toString(nycl)+ " access = ("
                        +toString(ix)+", " + toString(iy)+")";
                messageRF( sbuff, 2 );
                exit( EXIT_FAILURE );
            }
        #endif
        return data[iy + ix*nycl][1];
    }

    //   index a single pixel of the real pix - for complex to real transform only
    inline float& rre( const int ix, const int iy )
    {
        #ifdef RFPIX_BOUNDS_CHECK
            if( (ix<0) || (ix>=nxl) ||
                (iy<0) || (iy>=nyl) ){
                sbuff= "out of bounds index in cfpix::rre(); size = "
                        +toString(nxl) + " x " + toString(nyl)+ " access = ("
                        +toString(ix)+", " + toString(iy)+")";
                messageRF( sbuff, 2 );
                exit( EXIT_FAILURE );
            }
        #endif
        return rpix[iy + ix*nyl];
    }

    int resize( const int nx, const int ny );
    void findRange( float &rmin, float &rmax );

    void init( int mode=0, int nthreads=1 );        // for FFTW plan generation
    void copyInit( rfpix &xx );

    void fft();     //  perform forward FFT
    void ifft();    //  perform inverse FFT

    // unary operators
    //  remember that simple implementations of binary operators 
    //   can be VERY slow so don't use
    rfpix& operator=( const rfpix &m );
    rfpix& operator=( const float xf );

//----------------- private functions ---------------------

private:
    //   local variables have an "l" suffix for "local
    //   image size :  real = nxl * nyl ,  complex = nxl * nycl
    int nxl, nyl, nycl;             // current stored image size
    int nrxyl;                      // totalsize of real image
    int initLevel;                  // save init level
    float *rpix;                    // real image data buffer
    fftwf_complex *data;            // complex image data buffer
    fftwf_plan  planTf, planTi;     //  FFTW plans

    void messageRF( std::string &smsg, int level = 0 );      // common message handler

    std::string sbuff;

    //  misc subroutine to convert numbers to strings for messages
    //  MSVS 2010 does NOT have to_string( int ) so make equivalent
    std::string toString( int i );
};

#endif  // RFPIX_HPP
