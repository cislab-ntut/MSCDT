FROM python:3.8

WORKDIR /app
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /app

#EXPOSE 8888:8888

CMD [ "python", "CSP0.py" ]