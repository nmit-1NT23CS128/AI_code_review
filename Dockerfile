FROM python:3.10

WORKDIR /app
ENV ENABLE_WEB_INTERFACE=true
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "-m", "server.app"]
