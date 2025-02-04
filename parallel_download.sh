#!/bin/bash

declare -a bricks=( "0900m897" "0112m895" "0337m895" "0562m895" "0787m895" "1012m895")

downloadimage() {
    url_base="https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/coadd/"
    section=$1
    wget -q "$url_base${section:0:3}/${1}/legacysurvey-${1}-image-g.fits.fz"
}
export -f downloadimage
parallel -j4 downloadimage ::: ${bricks[@]}

# rm legacysurvey-*.fits.fz