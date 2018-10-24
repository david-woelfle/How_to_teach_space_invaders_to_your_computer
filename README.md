# Introduction
This repository contains the presentation and code for my talk ["How to teach space invaders to your computer"](https://de.pycon.org/schedule/talks/how-to-teach-space-invaders-to-your-computer/) at PyCon.DE 2018 held on 25.10.2018 in Karlsruhe.

# Presentation
[presentation/How_to_teach_space_invaders_to_your_computer.pdf](presentation/How_to_teach_space_invaders_to_your_computer.pdf) is a pdf version of the slides I used during my presentation. The original slides have been interactive, while the pdf is not. The videos of the Space Invaders gameplay are also located in the [presentation](presentation) directory. Other interactive content can be found online, links to the sources are given in the slides.

# Structure of the repository.
The repo contains two interactive notebooks.  
[analyse_training.ipynb](analyse_training.ipynb) contains the code that generated parts of the presentation content, i.e. the videos and plots about the training progress.  
  
[run_training.ipynb](run_training.ipynb) contains the code for running the training of autoencoder and agent.  
  
Most of the actual training logic is located in [training.py](training.py) while additional supporting code can be found in [helpers.py](helpers.py) and [extractor.py](extractor.py). The results of the training will be placed in a subdirectory under [results](results) and actually the results used in the presentation are stored there too. 

# Running the juypter notebooks
The notebooks have been executed inside a docker container which contains all required packages and dependencies.  
See https://docs.docker.com/get-started/ for an introduction to docker.  
See also https://docs.docker.com/compose/overview/ for an introduction to docker-compose.  

To handle the containers you may want to execute the following commands inside the root directory of the repository:  
Use `docker-compose up -d` to start the container. Afterwards you can visit [http://localhost:8888](http://localhost:8888) in your browser to access the notebook server.  
Use `docker-compose logs | grep token= | cut -d = -f 2'` to retrieve the token required for logging in at juypter.  
Use `docker-compose down` to stop and remove the container.  
  
The training can also be executed on a GPU, to do so overwrite the [docker-compose.yml](docker-compose.yml) file with the version in [gpu_mode/docker-compose.yml](gpu_mode/docker-compose.yml). You will need to have [nvidia docker](https://github.com/NVIDIA/nvidia-docker) installed.  
  
This has been tested and design for linux. If you wish to execute the code under windows you probably need to adapt the [docker-compose.yml](docker-compose.yml) file especially the volume mount.

