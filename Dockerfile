FROM python:3.10.8-slim-bullseye

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"


WORKDIR /app/
COPY train/ ./train
COPY main.py requirements.txt ./
RUN pip install -r requirements.txt
RUN useradd app

USER app
EXPOSE 8000
ENTRYPOINT ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0"]
