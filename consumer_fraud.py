# Example written based on the official 
# Confluent Kakfa Get started guide https://github.com/confluentinc/examples/blob/7.1.1-post/clients/cloud/python/consumer.py

from confluent_kafka import Consumer, Producer
import json
import ccloud_lib
import datetime
import time
import joblib
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, StandardScaler, LabelEncoder)
from pickle import load





# Initialize configurations from "python.config" file
CONF = ccloud_lib.read_ccloud_config("python.config")
TOPIC = "fraud_detection" 

# Create Consumer instance
# 'auto.offset.reset=earliest' to start reading from the beginning of the
# topic if no committed offsets exist
consumer_conf = ccloud_lib.pop_schema_registry_params_from_config(CONF)
consumer_conf['group.id'] = 'transaction_data'
consumer_conf['auto.offset.reset'] = 'earliest' # This means that you will consume latest messages that your script haven't consumed yet!
consumer = Consumer(consumer_conf)

# Subscribe to topic
consumer.subscribe([TOPIC])

#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#         definitions          #
#______________________________#

def distance(row):
    try: 
        return (geodesic(row['geometry'], row['merch_geometry']).km) 
    except:
        return np.nan

def from_Unix_to_dateTime(unix_time):
    manipDate = datetime.datetime.utcfromtimestamp(int(str(unix_time)[0:10])).strftime('%Y-%m-%d %H:%M:%S') 
    manipDate = datetime.datetime.strptime(manipDate, '%Y-%m-%d %H:%M:%S')
    return manipDate

def processusDataset(df_fraud_detection):
    df_fraud_detection['dob']= pd.to_datetime(df_fraud_detection['dob'])
    df_fraud_detection['age'] = (df_fraud_detection['trans_date_trans_time'] - df_fraud_detection['dob']).astype('<m8[Y]')
    df_fraud_detection['age'] = df_fraud_detection['age'].astype(int)
    df_fraud_detection["hour"] = df_fraud_detection['trans_date_trans_time'].dt.hour.astype(int)
    df_fraud_detection['dayofweek'] = df_fraud_detection['trans_date_trans_time'].dt.dayofweek.astype(int)
    df_fraud_detection['month'] = df_fraud_detection['trans_date_trans_time'].dt.month.astype(int)
    df_fraud_detection['dayofyear'] = df_fraud_detection['trans_date_trans_time'].dt.dayofyear.astype(int)
    df_fraud_detection['dayofmonth'] = df_fraud_detection['trans_date_trans_time'].dt.day.astype(int)
    df_fraud_detection['weekofyear'] = df_fraud_detection['trans_date_trans_time'].dt.isocalendar().week.astype(int)
    df_fraud_detection['card_issuer_MMI'] = [ f'mmi{str(x)[0:1]}' for x in df_fraud_detection['cc_num']]
    df_fraud_detection['card_issuer_Bank'] = [ int(str(x)[1:6]) for x in df_fraud_detection['cc_num']]
    df_fraud_detection['distance'] = df_fraud_detection.apply(lambda row: distance(row), axis = 1 )
    df_fraud_detection['distance'] = (df_fraud_detection['distance']).astype(float,2)
    df_fraud_detection['amt'] = df_fraud_detection['amt'].astype(float,2)
    df_fraud_detection = df_fraud_detection.drop(columns=['trans_date_trans_time','is_fraud','merchant','first','last','street','city','state','zip','job','dob','trans_num','unix_time','cc_num'])
    return df_fraud_detection


def acked(err, msg):
    global delivered_records
    # Delivery report handler called on successful or failed delivery of message
    if err is not None:
        print("Failed to deliver message: {}".format(err))
    else:
        delivered_records += 1
        print("Produced record to topic {} partition [{}] @ offset {}"
                .format(msg.topic(), msg.partition(), msg.offset()))


# Process messages
try:
    while True:
        msg = consumer.poll(1.0) # Search for all non-consumed events. It times out after 1 second
        if msg is None:
            # No message available within timeout.
            # Initial message consumption may take up to
            # `session.timeout.ms` for the consumer group to
            # rebalance and start consuming
            print("Waiting for message or event/error in poll()")
            continue
        elif msg.error():
            print('error: {}'.format(msg.error()))
        else:
            # Check for Kafka message
            #record_key = msg.key()
            #record_key2 = record_key.decode('utf8')
            #record_key2 = json.loads(record_key2)
            #record_key2 = json.dumps(record_key2, indent=4, sort_keys=True)
            #record_value = msg.value()
            #record_value2 = record_value.decode('utf8')
            #record_value2 = json.loads(record_value2)
            #record_value2 = json.dumps(record_value, indent=4, sort_keys=True)
            record_key = msg.key()
            record_value = msg.value()
            data = json.loads(record_value)
            print(f"New transaction recorded : {record_key} with {record_value} ")

            # Convert JSON data to Pandas DF for Model prediction
                     
            df_cols = ['cc_num','merchant','category','amt','first','last','gender','street','city','state','zip','lat','long','city_pop','job','dob','trans_num','merch_lat','merch_long','is_fraud','unix_time']

            dataset_from_api = pd.DataFrame(columns = df_cols, index=["0"])
            data_list =  data["data"]
            for transaction in data_list:
                for i in range(len(transaction)):
                    dataset_from_api.iloc[0,i] = transaction[i]
                print("\n Dataset updated with: \n")
                
            
            #¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
            #        Execution Code        #
            #______________________________#
            print(dataset_from_api)

            # Correstions difference API and Test Set. 
            dataset_from_api['trans_date_trans_time'] = [ from_Unix_to_dateTime(x) for x in dataset_from_api['unix_time']]
            dataset_from_api['unix_time'] = [ int(str(x)[0:10]) for x in dataset_from_api['unix_time']]

            # importations 
            path = os.path.dirname(__file__)
            #preprocessor = joblib.load(path+"/src/custom_transformer.pkl")
            model = joblib.load(path+"/src/model.joblib")

            # Apply columns transformations
            dataset_from_api = processusDataset(dataset_from_api) # We apply the same transformations than the first dataset 
            preprocessor_df = load(open(path+'/src/scaler.pkl', 'rb'))
            X_numpy = preprocessor_df.transform(dataset_from_api)

            # Prediction 
            Y = model.predict(X_numpy) # Prédictions on test set 

            # Final dataset to return 
            finalDataset = dataset_from_api
            finalDataset['is_fraud'] = Y

            print(finalDataset)
            record_value2 = finalDataset.to_json(orient='values')


            #¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
            #     End creation dataset     #
            #______________________________#


            #for i in range(len(record_value)):
            #    print(record_value[i])

            #test = record_value[1]
            #print(f"\nTest is {test}\n")
            ##Starts producing new record update and push to PostgreSQL##

            # Initialize configurations from "python.config" file
            CONF = ccloud_lib.read_ccloud_config("python.config") # lecture du fichier de conf
            TOPIC2 = "fraud_detection_predict" # quel topic va être utilisé


            # Create Producer instance
            producer_conf = ccloud_lib.pop_schema_registry_params_from_config(CONF) # transmission de la configuration
            producer = Producer(producer_conf) 

            # Create topic if it doesn't already exist
            ccloud_lib.create_topic(CONF, TOPIC)

            delivered_records = 0

            # Callback called acked (triggered by poll() or flush())
            # when a message has been successfully delivered or
            # permanently failed delivery (after retries).
            

           
        producer.produce(
            TOPIC2,
            key=record_key, 
            value=record_value2, 
            on_delivery=acked
        )
                # p.poll() serves delivery reports (on_delivery)
        # from previous produce() calls thanks to acked callback
        producer.poll(0)
        producer.flush() # Finish producing the latest event before stopping the whole script
        print("\n*** Data consumed *** \n \n {} {} \n".format(record_key, record_value2) )
        time.sleep(15) # Wait half a second


except KeyboardInterrupt:
    pass
finally:
    # Leave group and commit final offsets
        consumer.close()