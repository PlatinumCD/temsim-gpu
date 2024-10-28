/*      *** ransubs.h ***

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

        Some simple random number generators

    ranflat()     : return a random number with uniform distribution
    rangauss()    : return a random number with Gaussian distribution
    ranPoisson()  : return a random number with Poisson distribution

    getInitSeed() : diagnostic
    resetSeed()   : diagnostic - should not normally use this

    move RNG from slicelib to here
    to protect the seed instead of awkwardly passing it around


    started 23-dec-2023 E. Kirkland
    update resetSeed() to return void  11-jun-2024 ejk
    update to better low level RNG xorshift* 14-jul-2024 ejk
    last modified 14-jul-2024 ejk
*/


#ifndef RANSUBS_HPP   // only include this file if its not already

#define RANSUBS_HPP   // remember that this has been included

#include <cmath>
#include <cstdint>

//------------------------------------------------------------------
class ransubs {

public:

    ransubs(uint64_t iseed0=0 );         // constructor functions

    ~ransubs();        //  destructor function

    uint64_t getInitSeed() { return initseed;  }

    // <0 indicates a bad init
    int getStatus() { return initOK;  }

    /*---------------------------- ranflat -------------------------------*/
    /*  from slicelib 23-dec-2023 ejk
        update to  xorshift* 14-jul-2024 ejk

        return a random number in the range 0.0->1.0
        with uniform distribution

        remember: seed == 0 is not allowed

    for xorshift* method see:
    SEBASTIANO VIGNA, ACM Transactions on Mathematical Software,
          Vol. 42, No. 4, Article 30, Publication date: June 2016

     also see:

     plain xorshift in: George Marsaglia, J. Stat. Software 8, 14 (2003)
         p. 1-6.

     review in: www.pcg-random.org
    */
    inline double ranflat()
    {
        iseed ^= iseed >> 12;
        iseed ^= iseed << 25;
        iseed ^= iseed >> 27;
        return(5.42101086242752217E-20 * double(UINT64_C(2685821657736338717) * iseed));
    }  // end ransubs::ranflat()

    double rangauss();

    int ranPoisson(double mean);

    void resetSeed( uint64_t newSeed ) { iseed = newSeed; }

private:

    uint64_t iseed;
    uint64_t initseed;
    int initOK;
 
}; // end ransubs::


#endif