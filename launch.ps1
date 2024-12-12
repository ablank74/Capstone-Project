docker build . -t capstone
$unixPath = $(Get-Location).Path.Replace('\', '/')
docker run -it --rm --gpus=all -p 8888:8888 -p 5000:5000 -v "$unixPath/notebooks:/root/notebooks" capstone