FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel
USER root

RUN apt update
#RUN apt full-upgrade -y
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

# Initialize conda environment
RUN conda init
# RUN conda update --name base --channel defaults conda

# COPY environment.yml /tmp/environment.yml
# RUN conda env update --name base --file /tmp/environment.yml
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# RUN git clone https://github.com/rtqichen/torchdiffeq.git
# WORKDIR torchdiffeq
# RUN git checkout cbbec05b37acc6dd1912abd81613c558cb0df9aa
# RUN pip install -e .
# WORKDIR ..

# Setup Jupyterlab and extensions
#RUN mkdir --parents /home/$username/.jupyter/lab/user-settings/@jupyterlab/apputils-extension
#RUN echo "{\"theme\": \"JupyterLab Dark\", \"theme-scrollbars\": true}" > /home/$username/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings
#RUN echo "{\"disclaimed\": true, \"enabled\": true}" > /home/$username/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/plugin.jupyterlab-settings
RUN mkdir --parents /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension
RUN echo "{\"theme\": \"JupyterLab Dark\", \"theme-scrollbars\": true}" > /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings
RUN echo "{\"disclaimed\": true, \"enabled\": true}" > /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/plugin.jupyterlab-settings
RUN jupyter nbextension enable --py widgetsnbextension
#RUN jupyter labextension install --no-build @jupyter-widgets/jupyterlab-manager
#RUN jupyter labextension install --no-build @jupyterlab/toc
#RUN jupyter labextension install --no-build @jupyterlab/git
#RUN jupyter labextension install --no-build @jupyterlab/debugger
#RUN jupyter labextension install --no-build @aquirdturtle/collapsible_headings
#RUN jupyter labextension install --no-build @krassowski/jupyterlab_go_to_definition
#RUN jupyter labextension install --no-build @telamonian/theme-darcula
#RUN jupyter labextension install --no-build jupyterlab-execute-time
#RUN jupyter labextension install --no-build jupyterlab-plotly@4.9.0
#RUN jupyter labextension install --no-build plotlywidget@4.9.0
RUN jupyter lab build

# Add necessary paths to $PATH
#RUN echo "export PYTHONPATH=/src:$PYTHONPATH" >> /home/$username/.bashrc
#RUN echo "export PATH=$PATH:/usr/local/cuda/bin" >> /home/$username/.bashrc
RUN echo "export PYTHONPATH=/src:$PYTHONPATH" >> /root/.bashrc
RUN echo "export PATH=$PATH:/usr/local/cuda/bin" >> /root/.bashrc

WORKDIR /src
USER root
CMD ["/usr/sbin/sshd", "-D"]
