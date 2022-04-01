# This script will download all datasets.

# Check if gdown is available
if ! command -v gdown &> /dev/null
then
  echo "gdown is not installed. Please install it via 'pip install gdown'"
  exit
fi

download_dataset() {
  if [ -d ../dataset/mpc-$1 ]; then
    echo $1 'dataset already exists. Skipping.'
    return
  fi
  mkdir -p ../dataset
  if gdown $2 -O ../dataset/mpc-$1.zip ; then
    unzip ../dataset/mpc-$1.zip -d ../dataset/
  else
    echo 'Download failed. Either check your internet connection or upgrade gdown.'
  fi
}

download_dataset cape 1der12IAm_1o_M92nj71r0HpfxmBCaQmc
download_dataset dt4d 1r9VFHIZcatSej6guY_hGoGjrNqbgazAz
download_dataset dd 1ykFSe9TI9kZ-RozZw874YHDiO1cLRCgc
download_dataset sapien 13yMOoFmUV2Ca9j0tm_CD0nd1BGx1T8Jx
