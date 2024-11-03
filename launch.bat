docker build . -t capstone
docker run -it --rm --gpus=all -p 8888:8888 -p 5000:5000 -v "C:\GitRepos\Capstone_Project\notebooks":/root/notebooks capstone