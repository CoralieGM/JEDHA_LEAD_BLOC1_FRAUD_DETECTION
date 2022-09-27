# Code inspired from Confluent Cloud official examples library
# https://github.com/confluentinc/examples/blob/7.1.1-post/clients/cloud/python/producer.py

from confluent_kafka import Producer
import json
import ccloud_lib # Library not installed with pip but imported from ccloud_lib.py
import numpy as np
import time
import requests
import datetime

# Initialize configurations from "python.config" file
CONF = ccloud_lib.read_ccloud_config("python.config") # lecture du fichier de conf
TOPIC = "fraud_detection" # quel topic va être utilisé


# Create Producer instance
producer_conf = ccloud_lib.pop_schema_registry_params_from_config(CONF) # transmission de la configuration
producer = Producer(producer_conf) 

# Create topic if it doesn't already exist
ccloud_lib.create_topic(CONF, TOPIC)

delivered_records = 0

# Callback called acked (triggered by poll() or flush())
# when a message has been successfully delivered or
# permanently failed delivery (after retries).
def acked(err, msg):
    global delivered_records
    # Delivery report handler called on successful or failed delivery of message
    if err is not None:
        print("Failed to deliver message: {}".format(err))
    else:
        delivered_records += 1
        print("Produced record to topic {} partition [{}] @ offset {}"
                .format(msg.topic(), msg.partition(), msg.offset()))

try: 
    # Starts an infinite while loop that produces random current temperatures
    while True:
        url = "https://real-time-payments-api.herokuapp.com/current-transactions"
        response = requests.get(url)
        transaction = json.loads(response.json())["data"]
        #record_key = "price"
        record_value = json.dumps({"data": transaction})
        record_key  = str(datetime.datetime.now())
        print(record_key, record_value)

        # This will actually send data to your topic
        producer.produce(
            TOPIC,
            key=record_key, # key = weather
            value=record_value, # value = valeur random
            on_delivery=acked
        )
        # p.poll() serves delivery reports (on_delivery)
        # from previous produce() calls thanks to acked callback
        producer.poll(0)
        time.sleep(15)

 # Interrupt infinite loop when hitting CTRL+C
except KeyboardInterrupt:
    pass
finally:
    producer.flush() # Finish producing the latest event before stopping the whole script