FROM gcr.io/tensorflow/tensorflow:0.12.0-rc1-gpu

MAINTAINER Sanghoon Yoon <shygiants@gmail.com>

# Install PyMongo
RUN pip --no-cache-dir install pymongo

# Copy source files
COPY . /cdbn

WORKDIR "/cdbn"