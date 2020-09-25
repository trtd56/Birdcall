FROM python:3.7

WORKDIR /app

# RUN virtualenv /env
# ENV VIRTUAL_ENV /env
# ENV PATH /env/bin:$PATH
# ADD requirements.txt /app/requirements.txt
# RUN pip install -r /app/requirements.txt

ADD . /app

RUN apt-get update
RUN apt-get install -y libsndfile1
RUN pip install --trusted-host pypi.python.org -r requirements.txt
# EXPOSE 80

ENV NAME World

CMD ["python", "app.py"]
