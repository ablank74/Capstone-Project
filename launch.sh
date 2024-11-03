#!/bin/bash



if [[ $(id -nG) =~ "docker" ]] ; then
    echo "$USER in Docker Group"
else
  # Add the current user to the Docker group
  echo "Adding $USER to the Docker group"
  sudo usermod -aG docker $USER
  echo "User added to the Docker group and restarting Docker"
  sudo systemctl restart docker
fi

#prompt if we should rebuild the container
read -p "Do you want to rebuild the container? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Rebuilding the container"
    docker build . -t capstone --no-cache
fi
docker run -it --rm --gpus=all -p 8888:8888 -p 5000:5000 -v GitRepos/ITAI-2376/aws-notebooks:/root/notebooks capstone