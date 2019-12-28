FROM gitpod/workspace-full
                    
USER gitpod

# Install Python and the TensorFlow package dependencies
RUN sudo apt install python3-dev python3-pip -y

# Install the TensorFlow pip package dependencies
RUN pip3 install -U --user pip six numpy wheel setuptools mock 'future>=0.17.1' \
 && pip3 install -U --user keras_applications --no-deps \
 && pip3 install -U --user keras_preprocessing --no-deps

# Install Bazel
# install v0.24.1 
# Version	        Python version  Compiler    Build tools
# tensorflow-1.14.0 2.7, 3.3-3.7    GCC 4.8	    Bazel 0.24.1

# change compiler to GCC 4.8
#RUN sudo update-alternatives --remove-all gcc \
# && sudo update-alternatives --remove-all g++

RUN sudo apt-get install python-software-properties
RUN sudo add-apt-repository ppa:ubuntu-toolchain-r/test
RUN sudo apt-get update
RUN sudo apt-get install gcc-4.8
RUN sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 50

# bazel related
RUN sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python

RUN wget https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel-0.24.1-installer-linux-x86_64.sh

RUN chmod +x bazel-0.24.1-installer-linux-x86_64.sh

RUN ./bazel-0.24.1-installer-linux-x86_64.sh --user

# Download the TensorFlow source code
RUN git clone https://github.com/tensorflow/tensorflow.git \
 && cd tensorflow \
 && git checkout v1.14.0

# install pip3 dependencies
RUN pip3 install keras==2.2.4 \
 && pip3 uninstall six -y \
 && pip3 install six \
 && pip3 uninstall wrapt -y \
 && pip3 install wrapt \
 && pip3 uninstall python-dateutil -y \
 && pip3 install python-dateutil \
 && pip3 install matplotlib \
 && pip3 install numpy \
 && pip3 install tensorflow==1.14.0