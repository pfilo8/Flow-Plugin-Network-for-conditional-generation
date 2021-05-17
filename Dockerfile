FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel
USER root

RUN apt update
RUN apt install -y build-essential gcc g++ git openssh-server vim htop autoconf libboost-all-dev libtiff-dev curl \
    unzip libz-dev libpng-dev libjpeg-dev libopenexr-dev wget cmake sudo

RUN curl -fsSL https://deb.nodesource.com/setup_15.x | bash -
RUN apt install -y nodejs

# Install SSH
RUN mkdir /var/run/sshd
# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# Change default port for SSH
RUN sed -i 's/#Port 22/Port 22/' /etc/ssh/sshd_config
RUN sed -i 's/Port 22/Port 4444/' /etc/ssh/sshd_config
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
EXPOSE 4444

# Install POV-Ray for point cloud rendering
RUN wget https://github.com/POV-Ray/povray/archive/v3.7.0.8.tar.gz -q --show-progress --progress=bar:force 2>&1
RUN tar -xzf v3.7.0.8.tar.gz
WORKDIR povray-3.7.0.8/unix
RUN bash prebuild.sh
WORKDIR ..
RUN bash configure COMPILED_BY="A B <a@b.com>" LIBS="-lboost_system -lboost_thread"
RUN make --jobs
RUN make install
WORKDIR ..

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt


# Setup Jupyterlab and extensions
RUN mkdir --parents /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension
RUN echo "{\"theme\": \"JupyterLab Dark\", \"theme-scrollbars\": true}" > /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings
RUN echo "{\"disclaimed\": true, \"enabled\": true}" > /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/plugin.jupyterlab-settings
RUN jupyter nbextension enable --py widgetsnbextension
RUN jupyter lab build

# Add necessary paths to $PATH
RUN echo "export PYTHONPATH=/src:$PYTHONPATH" >> /root/.bashrc
RUN echo "export PATH=$PATH:/usr/local/cuda/bin" >> /root/.bashrc

WORKDIR /src
USER root
CMD ["/usr/sbin/sshd", "-D"]
