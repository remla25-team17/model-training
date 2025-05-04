FROM python:3.12.9-slim
WORKDIR /root
RUN apt-get update
RUN apt-get install -y git
COPY requirements.txt /root/
RUN pip install -r requirements.txt
COPY run.py /root/
COPY a1_RestaurantReviews_HistoricDump.tsv /root/
COPY a2_RestaurantReviews_FreshDump.tsv /root/
ENTRYPOINT ["python"]
CMD ["run.py"]