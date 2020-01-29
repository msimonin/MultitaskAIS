ARG NAMESPACE
ARG TARGET_REF
FROM ${NAMESPACE}/multitaskais_base:${TARGET_REF}

LABEL maintainer="matthieu.simonin@inria.fr"

SHELL ["/bin/bash", "-c"]

RUN apt install -y git g++

COPY . .
COPY docker/alertstats/. .
RUN chmod 755 main

RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate SESAME_PY3CPU && \
    pip install -r docker/tracks/requirements.txt

ENTRYPOINT ["./main"]