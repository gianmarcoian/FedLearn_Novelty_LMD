FROM nvcr.io/nvidia/pytorch:22.11-py3

WORKDIR /app

COPY app/requirements.txt /app
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

COPY app/ /app

ENTRYPOINT ["python3"]
CMD ["main.py"]


