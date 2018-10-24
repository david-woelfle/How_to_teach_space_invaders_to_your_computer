The training scripts should be run on a GPU, at least if you want to play around with the settings.

On Google Cloud you may use the following steps:
* Create a VM
  * Image: NVIDIA GPU Cloud Image for Deep Learning and HPC
  * 8 Cores
  * 40Gb Ram
  * 32Gb Disk
* Create a firewall rule that let's you access the Jupyter Notebook and eventually Tensorboard.
* Log in and execute the following commands:
  * Allow the Nvidia Programs to finish installing.
  * Add Nvidia API key or ignore it, it doens't seem to make a difference
  * sudo curl -L "https://github.com/docker/compose/releases/download/1.22.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
  * sudo chmod +x /usr/local/bin/docker-compose
  * git clone git@github.com:david-woelfle/How_to_teach_space_invaders_to_your_computer.git
  * cd How_to_teach_space_invaders_to_your_computer
  * mv gpu_mode/docker-compose.yml .
  * sudo docker-compose up -d
  * sudo docker-compose logs | grep token   # will print you the token to login in jupyter

That's it. You can now open the access the Jupyter notebooks on the remote machine.
 

