/*      *** ransubs.cpp ***
 
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
 
   NOTE: stl::random has several random number generators with different distributions
   similar to what is here. They probably have good statistical properties but are very
   slow and cannot be inlined. Probably because it cannot be inlined.
   This RNG is fast and fairly good for most things.
 
    started 23-dec-2023 E. Kirkland
    update to better low level RNG xorshift* 14-jul-2024 ejk
    last modified 14-jul-2024 ejk

*/

#include "ransubs.hpp"   //  header for this class

#include <ctime>  // to init iseed
#include <cstdint>

//=============================================================
//---------------  creator and destructor --------------

//  usually iseed0=0 to use system time (approx. random every run)
//  but can force repeatable RN sequence every time by speficyig iseed0
//
//  an actual value of iseed=0 will not work

ransubs::ransubs( uint64_t iseed0)
{
    long  ltime;

    initseed = iseed0;   //  save for later diagnostics if needed
    initOK = +1;         //  <0 inidcates a bad init

    if (0 == iseed0) {
        ltime = (long)time(NULL);
        initseed = ltime;
        if (ltime == -1) {   //  if time() failed (?) 
            ltime = (long) 234;  // use something (will get same sequence everytime; BAD)
            initOK = -1;
        }
        iseed = (uint64_t) ltime;
    }
    else iseed = (uint64_t) iseed0;

    return;
};

ransubs::~ransubs()
{
}

//=============================================================

/*-------------------- rangauss() -------------------------- */
/*  from slicelib 23-dec-2023 ejk
 
    Return a normally distributed random number with
    zero mean and unit variance using Box-Muller method

    ranflat() is the source of uniform deviates

    ref.  Numerical Recipes, 2nd edit. page 289

    added log(0) test 10-jan-1998 E. Kirkland
*/
double ransubs::rangauss()
{
    double x1, x2, y;
    static double tpi = 2.0 * 3.141592654;

    // be careful to avoid taking log(0) 
    do {
        x1 = ranflat();
        x2 = ranflat();
    } while ((x1 < 1.0e-30) || (x2 < 1.0e-30));

    y = sqrt(-2.0 * log(x1)) * cos(tpi * x2);

    return(y);

}  // end ransubs::rangauss()


/*---------------------------- ranPoisson -------------------------------*/
/*  from slicelib 23-dec-2023 ejk

    return a random number with Poisson distribution
    and mean value mean

    There is a nice poisson RNG in c++11 but some compilers still do not
    have it, so make one here. The function lgamma()= log of gamma function
    is in the C99 standard and seems to have more widespread support
    (in mac osx 10.8 but NOT MSVS2010!).  Leave lgamma() code for future
    and approximate large means with Stirling's formula.

    A. C, Atkinson, "The Computer Generation of Poisson Random Variable"
         J. Royal Statistical Society, Series C (Applied Statistics),
         Vol. 28, No. 1, (1979) p. 29-35.

    D. E. Knuth, "The Art of Computer Programming, vol.2, Seminumerical
        Algorithms"  Addison-Wesley 1981, 1969, p. 132.

    calls ranflat()

    input:
        mean = desired mean (can be fractional)
        iseed = random number seed

    started 23-sep-2015 ejk
    add large mean portion 26-sep-2015 ejk
*/
int ransubs::ranPoisson(double mean)
{
    int n;
    static int lut = -1;
    static double oldMean = -100, oldEmean = 0;
    static double alpha, beta, c, k, PI, lnf[256];

    //  negative mean is not allowed
    if (mean <= 0) return 0;

    if (lut < 0) {   // init log( n! ) look up table
        lnf[0] = lnf[1] = 0.0;
        for (int i = 2; i < 256; i++) lnf[i] = lnf[i - 1] + log((double)i);
        lut = +1;
    }

    //---- use Atkinson method PM for small means
    //     (also Knuth vol.2)
    if (mean < 30) {

        double s;
        if (oldMean != mean) {  //  save if same mean is repeated
            oldMean = mean;
            oldEmean = exp(-mean);
        }
        n = -1;
        s = 1.0;
        do {
            n = n + 1;
            s = s * ranflat();
        } while (s >= oldEmean);
        return(n);

        //---  use Atkinson method PA for large means 
    }
    else {

        double u1, u2, x, y, lhs, rhs, temp, lnf0;

        if (oldMean != mean) {  //  save if same mean is repeated
            oldMean = mean;
            PI = 4.0 * atan(1.0);
            beta = PI / sqrt(3.0 * mean);
            alpha = beta * mean;
            c = 0.767 - 3.36 / mean;
            k = log(c) - mean - log(beta);
        }

        while (1) {      //  hopefully this will end !!

            do {
                u1 = ranflat();
                x = (alpha - log((1.0 - u1) / u1)) / beta;
            } while (x < -0.5);
            n = (int)(x + 0.5);
            u2 = ranflat();
            y = alpha - beta * x;
            temp = 1.0 + exp(y);
            lhs = y + log(u2 / (temp * temp));
            //  lgamma() is missing in some compilers so.....
            //rhs = k + n*log(mean) - lgamma(n+1); // gamma(n+1) = n!
            if (n < 0) continue;
            if (n < 255) lnf0 = lnf[n]; // log( n! ) from lookup table
            else {
                x = n;   //  Stirling Formula
                lnf0 = 0.5 * log(2 * PI) + (x + 0.5) * log(x) - x + 1.0 / (12.0 * x);
            }
            rhs = k + n * log(mean) - lnf0;
            if (lhs <= rhs) return n;
        }
    }

}  // end ransubs::ranPoisson()
