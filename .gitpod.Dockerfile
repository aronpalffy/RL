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

RUN sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python

RUN wget https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel-0.24.1-installer-linux-x86_64.sh

RUN chmod +x bazel-0.24.1-installer-linux-x86_64.sh

RUN ./bazel-<version>-installer-linux-x86_64.sh --user

# Download the TensorFlow source code
RUN git clone https://github.com/tensorflow/tensorflow.git \
 && cd tensorflow \
 && git checkout r1.14.0
