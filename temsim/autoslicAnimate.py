#
#   autoslicAnimate.py
#
#  read magnitude of slices from autoslic and make a movie
#  tested with python 3.8
#  python uses Imagemagick (includes ffmpeg) 
#    which must be installed separately
#
#------------------------------------------------------------------------
#Copyright 2021 Earl J. Kirkland
#
#This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#---------------------- NO WARRANTY ------------------
#THIS PROGRAM IS PROVIDED AS-IS WITH ABSOLUTELY NO WARRANTY
#OR GUARANTEE OF ANY KIND, EITHER EXPRESSED OR IMPLIED,
#INCLUDING BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
#MERCHANABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#IN NO EVENT SHALL THE AUTHOR BE LIABLE
#FOR DAMAGES RESULTING FROM THE USE OR INABILITY TO USE THIS
#PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA
#BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR
#THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH
#ANY OTHER PROGRAM). 
#-----------------------------------------------------------------------------
#
#  started 3-jul-2021 ejk
#  small updates 26-dec-2021 ejk
#  last modified 26-dec-2021 ejk
#
# uses numpy, scipy, matplotlib packages
#from pylab import *

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import animation
import glob
import sys 

#  a few parameter offsets from slicelib.hpp
pNPIX    =   0   # number of pix 1 for real and 2 for complex
pDX      =   14  # dimension of pixel in x direction in Angstroms
pDY      =   15  # dimension of pixel in y direction in Angstroms
pAX      =  101  #  supercell size x, y
pBY      =  102
pNX      =  104 # (int) main image size (transmission function)
pNY      =  105


#---------- read_fTIFF() -------------------------------
#  read a floatTIFF 32 bit floating point image
#  remember that python reads in matrix order (row,col) not image (x,y)
def read_fTIFF( filename ):
    img = Image.open( filename, mode='r' )
    img.seek( 1 )    #  32 bit floating point image
    pix = np.array(img)
    img.seek( 2 )    #  32 bit floating point parameters
    param = np.transpose(np.array(img))  #  convert to 1D array
    if( 2 == param[pNPIX] ):   # a complex image
        #ny = pix.shape[0]
        nx = pix.shape[1]/2
        repix = pix[:,0:nx] # re, im parts are side by side
        impix = pix[:,nx:2*nx]
        cpix = repix + 1j * impix
        return cpix, param
    else:
        return pix, param
    #  end read_fTIFF

#--------- start main program here -----------------------
print("start autoslicAnimate.py...")

#  try current directory first
infiles = "slice*.tif"
files = glob.glob(infiles)
if( len(files) <= 0):   # if not here then ask for a path
    slicePath = input("type directory for slice*.tif files: ")
    infiles = slicePath + "slice*.tif"
    files = glob.glob(infiles)
    
if( len(files) <= 0):
    print("cannot find input files like ", infiles)
    sys.exit()

#------ first pass to find total range of all slices ----------
#  assume files listed is correct order - may not be guaranteed
    
nx0 = -1
ny0 = -1
ax = 0.0  # make globals
by = 0.0
nslices = 0
for infile in files:
    #print("read file ",infile)   # diagnostic
    pix, param = read_fTIFF( infile )
    if( nx0 < 0 ):
        nx0 = pix.shape[1]    #  = param[pNX] = image size in pixels
        ny0 = pix.shape[0]    #  = param[pNY]
        rmin = pix.min()
        rmax = pix.max()
        ax = float( param[pAX] )   #  supercell size (in Ang.)
        by = float( param[pBY] )
        
    else:
        z = pix.min()
        if( z < rmin):
            rmin = z
        z = pix.max()
        if(z > rmax ):
            rmax = z
        nx = pix.shape[1]    #  = param[pNX] = image size in pixels
        ny = pix.shape[0]    #  = param[pNY]
        
        if( (nx != nx0) or (ny != ny0)):
            print( "slice ",infile," bad size= ",nx,", ",ny)
        
    nslices += 1
    
print( "found ", nslices, " slices, with total range = ",rmin," to ",rmax)
print( "  and size= ",nx0,", ",ny0," pixels, a,b= ",ax,", ",by)


#-------- start animation stuff --------------------------
print( 'start animation...' )

images = []
fig, axis = plt.subplots()

islice = 0
for infile in files:
    #print("add file ",infile)   # diagnostic
    pix, param = read_fTIFF( infile )
    img =  axis.imshow( pix, extent=(0.0,ax,0.0,by), vmin=rmin,
            vmax= rmax, animated=True)
            #vmax= rmax, cmap=plt.get_cmap('hot'), animated=True)
    img.set_cmap('hot')  # works OK
    images.append([img])
    islice += 1

fig.colorbar(img, ax=axis)  #  show a colorbar
plt.title("%d slices" % islice)
plt.xlabel("x position (Ang)")
plt.ylabel("y position (Ang)")

anim = animation.ArtistAnimation(fig, images, interval=50, blit=True,
                        repeat=True, repeat_delay=1000)

#  choose one format
print("save animation as animate_slices.mp4...")
anim.save('animate_slices.mp4', dpi=150, fps=15, \
          extra_args=['-vcodec', 'libx264'])

# print("save animation as animate_slices.gif...")
# anim.save('animate_slices.gif', dpi=80 ) #, writer='imagemagick' )

plt.show()

print( "animation done" )
