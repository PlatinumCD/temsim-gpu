/*              *** rfpix.cpp ***

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

   C++ class to manage complex/real floating point images (in ANSI-C++)

   collect (isolate) the FFT details here so it can be easily changed in future

   perform complex to real and its inverse

   FFTW real to complex FFT's use two separate arrays one real and one complex
    real nx x ny
    complex nx x (ny/2+1)
    forward xform = real to complex and inverse xform = complex to real

  the FFT is NOT perfomred in place

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

operator=()

----------------------------------------------------------

   started from cfpix 7-may-2024 E. Kirkland
   working 18-may-2024 ejk
*/

#include "rfpix.hpp"    // class definition + inline functions here

#include <sstream>	// string streams

#include "slicelib.hpp"    // misc. routines for multislice

//------------------ constructor --------------------------------

rfpix::rfpix( int nx, int ny )
{
    nxl = nyl = nycl = nrxyl = 0;

    initLevel = -1;   // negative to indicate no initialization

    if( (nx > 0 ) && ( ny > 0 ) ) resize( nx, ny );

}  // end rfpix::rfpix()

//------------------ destructor ---------------------------------
rfpix::~rfpix()
{
    if ((nxl > 0) && (nyl > 0)) {
        fftwf_free(data);
        fftwf_free(rpix);
    }

    nxl = nyl = nycl = nrxyl = 0;
    initLevel = -1;

}  // end rfpix::~cfpix()

//------------------ copyInit ---------------------------------
void rfpix::copyInit( rfpix &xx)
{
    if( (nxl != xx.nxl) && (nyl != xx.nyl) ) return;

    planTf = xx.planTf;
    planTi = xx.planTi;

    initLevel = xx.initLevel;

}  // end rfpix::copyInit()


//------------------ forward transfrom ---------------------------------
void rfpix::fft()
{
    if( (0 == initLevel) || (1 == initLevel) ) {
        fftwf_execute_dft_r2c( planTf, rpix, data );
    } else {
        sbuff= "error: cfpix::fft() called before init()";
        messageRF( sbuff, 2 );
        exit( EXIT_FAILURE );
    }

}  // end rfpix::fft()

//------------------ inverse transfrom ---------------------------------
void rfpix::ifft()
{
    int ix, iy, j;
    float scale;

    if( (0 == initLevel) || (1 == initLevel) ) {
        fftwf_execute_dft_c2r( planTi, data, rpix );

        /*  multiplied by the scale factor */
        scale = 1.0F/( (float)(nxl * nyl) );

        for( ix=0; ix<nxl; ix++) {
            j = ix*nyl;
            for( iy=0; iy<nyl; iy++) {
                rpix[j++] *= scale;  //  rpix[iy + ix*ny]
            }
        } /* end for(ix..) */

    } else {
        sbuff= "error: rfpix::ifft() called before init()";
        messageRF( sbuff, 2 );
        exit( EXIT_FAILURE );
    }

}  // end rfpix::ifft()

//------------------ initializer ---------------------------------
//
//   mode = 0 for full measure (slow setup and fast execution)
//   mode = 1 for estimate (fast setup and slow execution)
//
//   nthreads = number of FFTW threads to use
//
// remember: FFTW has inverse sign convention so forward/inverse reversed
//
void rfpix::init( int mode, int nthreads )
{
    int  ns;


        if( 0 == mode ) {

            initLevel = mode;
            if (nthreads > 1) {    //  initialize FFTW multithreading
                ns = fftwf_init_threads();
                fftwf_plan_with_nthreads(nthreads);
            }

            planTi = fftwf_plan_dft_c2r_2d( nxl, nyl, data, rpix, FFTW_MEASURE);  // inverse FFT
            planTf = fftwf_plan_dft_r2c_2d( nxl, nyl, rpix, data, FFTW_MEASURE);  // forward FFT

        } else if (1 == mode) {

            initLevel = mode;

            planTi = fftwf_plan_dft_c2r_2d(nxl, nyl, data, rpix, FFTW_ESTIMATE);  // inverse FFT
            planTf = fftwf_plan_dft_r2c_2d(nxl, nyl, rpix, data, FFTW_ESTIMATE);  // forward FFT
        }

}  // end rfpix::init()



/*------------------------- messageRF() ----------------------*/
/*
    common message output
    redirect all print message to here so this can be redirected
        to a dialog box in a GUI or cmd line 

   stemp[] = character string with message to disply
   level = level of seriousness
            0 = simple status message
        1 = significant warning
        2 = possibly fatal error
*/
void rfpix::messageRF( std::string &smsg,  int level )
{
    messageSL( smsg.c_str(), level );  //  just call slicelib version for now
}

//--------------------- operator=() ----------------------------------
//  element by element copy (only data not plan)
//   real data not implemented yet
rfpix& rfpix::operator=( const rfpix& m  )
{
    int nxyt= nxl*nyl, nxyct= nxl*nycl;

    if( (m.nxl != nxl) || (m.nyl != nyl)  ){
        sbuff= "rfpix operator= invoked with unequal sizes:\n"
                +toString(nxl)+" x "+ toString(nxl) +" and "
                + toString(m.nxl)+" x "+ toString( m.nxl );
        messageRF( sbuff, 2 );
        exit( EXIT_FAILURE );
    } else if( (nxl>0) && (nyl>0) ) {

        for( int i=0; i<nxyct; i++) {
           data[i][0] = m.data[i][0];   //  real part 
           data[i][1] = m.data[i][1];   //  imag part
        }
        for (int i = 0; i < nxyt; i++) {
            rpix[i] = m.rpix[i];   //  real part 
        }
    } else {
        sbuff= "bad operator=() in rfpix"; // gcc requires this step
        messageRF( sbuff, 2 );
        exit( EXIT_FAILURE );
    }

    return *this;

}     //  end  rfpix::operator=()

//--------------------- operator=() ----------------------------------
//  initial to a real value
//   
rfpix& rfpix::operator=( const float xf )
{
    int nxyt = nxl * nyl, nxyct = nxl * nycl;

    if( (nxl > 0) && (nyl>0)  ) {
        int i;
        for( i=0; i<nxyct; i++) {  // can't do FFT part yet
           data[i][0] = 0.0F;      //  real part 
           data[i][1] = 0.0F;      //  imag part
        }
        for (int i = 0; i < nxyt; i++) {
            rpix[i] = xf;   //  real part 
        }
    } else {
        resize( 1, 1 );    //  just do what we can...
        rpix[0] = xf;
        data[0][0] = 0.0F;
        data[0][1] = 0.0F;
    }

    return *this;
}     //  end  rfpix::operator=()

//--------------------- resize() ----------------------------------
//  resize data buffers 
//   nx, ny = new size of real image
//  note: existing data (if any) may be destroyed
//
int rfpix::resize( const int nx, const int ny )
{
    if( (nx != nxl) || (ny != nyl) ){
        if ((nxl != 0) && (nyl != 0)) {
            fftwf_free(data);
            fftwf_free(rpix);
        }
        nxl = nx;
        nyl = ny;
        nycl = ny/2 + 1;
        data = (fftwf_complex*) fftwf_malloc( nxl*nycl * sizeof(fftwf_complex) );
        rpix = (float*)fftwf_malloc(nxl * nyl * sizeof(float));
        if( (NULL == data) || (NULL == rpix) ) {
            sbuff= "Cannot allocate complex array memory in rfpix::resize()";
            messageRF( sbuff, 2 );
            return( -1 );
        }
    }

    return( +1 );

};  // end rfpix::resize()

//--------------------- findRange() ----------------------------------
//  find range of real pix
//
void rfpix::findRange( float &rmin, float &rmax )
{
    int nxyt= nxl*nyl;
    float x;

    if( (nxl > 0) && (nyl > 0)  ){
        int i;
        rmin = rmax = rpix[0];
        for( i=0; i<nxyt; i++) {
           x = rpix[i];
           if( x < rmin ) rmin = x;
           if( x > rmax ) rmax = x;
        }  /* end for(i...) */

        return;
    }

};  // end rfpix::findRange()

/*------------------------- toString( int ) ----------------------*/
/*
    convert a number into a string
*/
std::string rfpix::toString( int i )
{
        std::stringstream ss;
        ss << i;
        return ss.str();

};  // end rfpix::toString( int )

