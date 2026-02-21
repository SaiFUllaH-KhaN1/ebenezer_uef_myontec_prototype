FROM python:3.13

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 9090

# host_port : container_port
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9090"]