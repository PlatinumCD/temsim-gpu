NUM_THREADS=8
FFTW_DOWNLOAD_URL="https://www.fftw.org/fftw-3.3.10.tar.gz"
FFTW_INSTALL_DIR="$(pwd)/fftw"

# Check if FFTW is already installed; if not, install it
if [ ! -d "$FFTW_INSTALL_DIR" ]; then
    echo "FFTW not found. Installing to $FFTW_INSTALL_DIR..."
    
    # Create installation directory
    mkdir -p "$FFTW_INSTALL_DIR"
    
    # Download and extract FFTW
    wget -O fftw.tar.gz "$FFTW_DOWNLOAD_URL"
    tar -xzf fftw.tar.gz
    cd fftw-3.3.10 || exit 1  # Navigate to extracted directory or exit if it fails

    # Configure, build, and install FFTW with single precision and threading support
    ./configure --prefix="$FFTW_INSTALL_DIR" --enable-single --enable-threads
    make -j "$NUM_THREADS"
    make install

    # Clean up downloaded files
    cd ..
    rm -rf fftw.tar.gz fftw-3.3.10

    echo "FFTW with multi-threading support installed in $FFTW_INSTALL_DIR"
else
    echo "FFTW is already installed in $FFTW_INSTALL_DIR. Skipping installation."
fi


# Attempt to load CUDA module
module load cuda/12.3
if [ $? -ne 0 ]; then
    echo -e "\033[31mFailed to load CUDA module version 12.3.\033[0m"
    exit 1
else
    echo -e "CUDA module \033[32mloaded successfully\033[0m."
fi

# Navigate to temsim-gpu/temsim-cuda directory
pushd temsim-cuda > /dev/null
    if [ $? -eq 0 ]; then
        echo -e "Navigated to \033[32mtemsim-gpu/temsim-cuda\033[0m directory."
    else
        echo -e "\033[31mFailed to navigate to temsim-gpu/temsim-cuda.\033[0m"
        exit 1
    fi


    mkdir build
    pushd build > /dev/null
        
        cmake -DFFTW_DIR=$FFTW_INSTALL_DIR ..
        make -j $NUM_THREADS
    popd
popd
