ARG NAMESPACE
ARG TARGET_REF
FROM ${NAMESPACE}/base_python3.6:${TARGET_REF}

LABEL maintainer="matthieu.simonin@inria.fr"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates


## Install miniconda
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN mkdir -p /opt/prog
WORKDIR /opt/prog


COPY . .

RUN conda env create -f requirements.yml

ENTRYPOINT ["/entrypoint.sh", "./main"]