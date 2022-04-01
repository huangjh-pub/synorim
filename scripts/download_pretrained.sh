# This script will download all pretrained models.

# Check if gdown is available
if ! command -v gdown &> /dev/null
then
  echo "gdown is not installed. Please install it via 'pip install gdown'"
  exit
fi

download_model() {
  if [ -d out_pretrained/$1 ]; then
    echo $1 'model already exists. Skipping.'
    return
  fi
  mkdir -p out_pretrained
  if gdown $2 -O out_pretrained/$1-model.zip ; then
    unzip out_pretrained/$1-model.zip -d out_pretrained/ && rm out_pretrained/$1-model.zip
  else
    echo 'Download failed. Either check your internet connection or upgrade gdown.'
  fi
}

download_model cape 1FMgXeM8zX448j8zQulfm0Zy0aHfblRPJ
download_model dt4d 1vs9rOfGeXOXk6Q4gGfkDR3ziCV_XvKYT
download_model dd 1ezvt-MoW0AIMHiJj8j9RZFu2PQWPqaLw
download_model sapien 1mmJDrVsDbUd1wjazDKGpeF0tUrpYtUDu
