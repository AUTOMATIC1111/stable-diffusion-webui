
rm download.tar

read -n 1 -p "Build tar?(y/N)" BUILD_TAR
if [ $BUILD_TAR == "y" ]
then
  tar -cvf download.tar ./log/ ./outputs/
fi

rm ./log/images/*
rm ./outputs/extras-images/*
rm ./outputs/img2img-grids/*
rm ./outputs/img2img-images/*
rm ./outputs/txt2img-grids/*
rm ./outputs/txt2img-images/*
