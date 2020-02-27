ARG NAMESPACE
ARG TARGET_REF
FROM ${NAMESPACE}/multitaskais_base:${TARGET_REF}

LABEL maintainer="matthieu.simonin@inria.fr"

SHELL ["/bin/bash", "-c"]

RUN apt update && apt install -y git g++ openjdk-8-jre

COPY . .
COPY docker/tracks/. .
RUN chmod 755 main

RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate SESAME_PY3CPU && \
    conda install -c conda-forge pyspark && \
    pip install -r docker/tracks/requirements.txt

ENTRYPOINT ["./main"]