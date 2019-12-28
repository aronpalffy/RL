FROM gitpod/workspace-full
                    
USER gitpod

# Install Python and the TensorFlow package dependencies
RUN sudo install python3-dev python3-pip -y

# Install the TensorFlow pip package dependencies
RUN pip3 install -U --user pip six numpy wheel setuptools mock 'future>=0.17.1' \
 && pip3 install -U --user keras_applications --no-deps \
 && pip3 install -U --user keras_preprocessing --no-deps

# Install Bazel
RUN sudo apt install curl \
 && curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add - echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list

# install v0.24.1 
# Version	        Python version  Compiler    Build tools
# tensorflow-1.14.0 2.7, 3.3-3.7    GCC 4.8	    Bazel 0.24.1
RUN sudo apt update && sudo apt install bazel-0.24.1

# Download the TensorFlow source code
RUN git clone https://github.com/tensorflow/tensorflow.git \
 && cd tensorflow \
 && git checkout r1.14.0
