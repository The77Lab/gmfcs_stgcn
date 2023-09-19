# Summary

Here's a Docker container setup for Visual Studio Code users who want to reproduce our results, test and experiment with our code in a more consistent environment.

# Step-by-Step Instructions

## Prerequisite

To access your NVidia GPU within the Docker container, you need to

1. Install the NVidia Driver
2. [Setup the Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## 1. Follow this [tutorial](https://code.visualstudio.com/docs/devcontainers/tutorial) to install VSCode, Docker, Dev Containers Extension (VSCode).

The Dev Containers Extension looks for the files in `.devcontainer` folder to build a Docker image (if it's not built yet) and launch the Docker container for you with specific arguments and configurations. Specifically, it consumes a `devcontainer.json` file. I already provided one here.

```JSON
{
  "name": "PyTorch",
  "build": {
    "dockerfile": "Dockerfile"
  },
  "runArgs": [
    "--privileged",
    "--network=host",
    "--gpus=all"
  ],
  "containerEnv": {
    "DISPLAY": "${localEnv:DISPLAY}"
  },
  "workspaceMount": "source=${localWorkspaceFolder},target=/${localWorkspaceFolderBasename},type=bind",
  "workspaceFolder": "/${localWorkspaceFolderBasename}",
  "mounts": [
    "source=${localEnv:HOME}${localEnv:USERPROFILE}/.bash_history,target=/home/vscode/.bash_history,type=bind",
    "source=/tmp/.X11-unix,target=/tmp/.X11-unix,type=bind"
  ],
  "features": {},
  "forwardPorts": [
    8888
  ],
  "customizations": {
    "vscode": {
      "settings": {
        "terminal.integrated.defaultProfile.linux": "bash",
        "terminal.integrated.profiles.linux": {
          "bash": {
            "path": "/bin/bash"
          }
        }
      },
      "extensions": [
        "ms-python.python",
        "ms-python.black-formatter",
        "ms-python.isort",
        "ms-python.mypy-type-checker",
        "ms-azuretools.vscode-docker",
        "eamodio.gitlens",
        "janisdd.vscode-edit-csv"
      ]
    }
  }
}
```

### Remarks:
* `build` field specifies the Dockerfile to use. Again, I provided one [Dockerfile](Dockerfile) here already.
  * Note that the base image used is: [pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime](https://hub.docker.com/layers/pytorch/pytorch/2.0.1-cuda11.7-cudnn8-runtime/images/sha256-82e0d379a5dedd6303c89eda57bcc434c40be11f249ddfadfd5673b84351e806?context=explore), and it is still using `Ubuntu 18.04`.
  * I created a non-root user with user ID 1000, which should be the same as that of the user on your host machine. This is to make permission management easy because any mounted files/folders will have the exact same permissions as outside the container. See a more detailed explanation [here](https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user#_creating-a-nonroot-user)
  * Finally, I installed `jupyter`.
* `runArgs` field specifies the `docker run` arguments, you can adjust it accordingly to your needs.
* `containerEnv` field specifies the env vars to populate for the container environment, `DISPLAY` and the `source=/tmp/.X11-unix,target=/tmp/.X11-unix,type=bind` in `mounts` field enables GUI for the docker container if you ever need it. You can check out more details [here](https://janert.me/guides/running-gui-applications-in-a-docker-container/).
* `forwardPorts` field sets up port forwarding for the container and the host machine, I added port 8888, so the jupyter notebook should work out of the box.
* `customizations` field allows configuration of preferred shell program, VSCode extensions to install etc.
* `workspaceMount` and `workspaceFolder` allow you to specify where to mount your code respository folder to in the Docker container, and the name of the folder.

## 2. First time setup: From the `Command Palette (F1)`, run `Dev Containers: Open Folder in Container...` to connect to the container (this will build the Docker image, start a Docker container, and mount your workspace for you).

## 3. (Optional) To run the jupyter notebook server:

```
jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root &
```
