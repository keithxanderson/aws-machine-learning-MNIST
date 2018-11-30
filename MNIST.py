import boto3
import os
import json
import base64
import datetime
import time

def create_training_datasource(client, train_S3_URL, schema, percentage_split, name):
    
    #Boto3 'machinelearning' client function to create a training datasource from a CSV file.  Function will
    #pull a CSV file from an S3 bucket, pull a schema file from an S3 bucket, and split the training data  
    #into a training datasource and an evaluation datasource using the percentage passed to the function.  The
    #name argument is just a friendly display name. 
    
    training_datasource_id = 'trainingID.' + base64.b32encode(os.urandom(5)).decode()
    eval_datasource_id = 'evaID.' + base64.b32encode(os.urandom(5)).decode()

    client.create_data_source_from_s3(
    DataSourceId=training_datasource_id,
    DataSourceName= name + ' - training split',
    DataSpec={
        'DataLocationS3': train_S3_URL,
        'DataRearrangement': json.dumps({
            "splitting":{
                "percentBegin": 0,
                "percentEnd": percentage_split,
                "strategy": "sequential"
            }
        }),
        'DataSchemaLocationS3': schema
    },
    ComputeStatistics=True
)
    
    client.create_data_source_from_s3(
    DataSourceId=eval_datasource_id,
    DataSourceName= name + ' - eval split',
    DataSpec={
        'DataLocationS3': train_S3_URL,
        'DataRearrangement': json.dumps({
            "splitting":{
                "percentBegin": percentage_split,
                "percentEnd": 100,
                "strategy": "sequential"
            }
        }),
        'DataSchemaLocationS3': schema
    },
    ComputeStatistics=True
)
    
    print("Created 70 percent training data set %s" % training_datasource_id)
    print("Created 30 percent eval data set %s" % eval_datasource_id)
    return(training_datasource_id, eval_datasource_id)    

def create_model (client, datasource_id, recipe, name):
    
    #Boto3 'machinelearning' client function to create a machine learning model from a training dataset.  Function
    #will use an existing traning dataset passed to the function as the AWS identifier number.  The recipe
    #file is pulled from an S3 bucket and must adhear to the AWS ML recipe format.  The name is just a friendly 
    #display name.    

    model_id = 'modelID.' + base64.b32encode(os.urandom(5)).decode()
    
    client.create_ml_model(
    MLModelId= model_id,
    MLModelName= name + ' - model',
    MLModelType= "MULTICLASS",
    Parameters={
        "sgd.maxPasses": "10",
            "sgd.maxMLModelSizeInBytes": "104857600",  
            "sgd.l2RegularizationAmount": "1e-5",
            "sgd.shuffleType": "auto"
    },
    TrainingDataSourceId=datasource_id,
    RecipeUri=recipe
    )
    print("Created model %s" % model_id)
    return model_id

def create_evaluation(client, model_ID, eval_dataset_ID, name):
    
    #Boto3 'machinelearning' client function to create an evaluation of an existing AWS ML model using an 
    #existing evaluation datasource.  Function will use an existing ML model passed to the function as an AWS 
    #identifier number and use an existing evaluation datasource passed to the function as an AWS identifier
    #number.  The name is just a friendly display name. 
    
    eval_id = 'evalID.' + base64.b32encode(os.urandom(5)).decode()
    
    client.create_evaluation(
    EvaluationId=eval_id,
    EvaluationName= name + ' - eval',
    MLModelId=model_ID,
    EvaluationDataSourceId=eval_dataset_ID
    )
    
    print("Created eval %s" % eval_id)
    return eval_id

def create_batch_prediction_dataset(client,batch_S3_URL, schema):
    
    #Boto3 'machinelearning' client function to create a prediction datasource from a CSV file.  Function will
    #pull a CSV file from an S3 bucket, pull a schema file from an S3 bucket, and create a batch datasource to be
    #used in for a batch prediction later. The batch dataset will have a unique schema since it does not contain
    #any target prediction values.  
    
    batch_id = 'batchID.' + base64.b32encode(os.urandom(5)).decode()
    
    client.create_data_source_from_s3(
    DataSourceId=batch_id,
    DataSourceName="batch prediction datasource %s" % batch_S3_URL,
    DataSpec={
        'DataLocationS3': batch_S3_URL,
        'DataSchemaLocationS3': schema
    },
    ComputeStatistics=True
    )
    
    print("Created batch %s" % batch_id)
    return batch_id

def poll_evaluation_status(client, evaluation_id):
    
    #The batch prediction should not run until our evaluation of our model is complete.  This is a polling function
    #to wait for the evaluation job to complete.  See link for original author of this polling function.
    #https://github.com/aws-samples/machine-learning-samples/blob/master/targeted-marketing-python/use_model.py
    
    delay = 60
    while True:
        eval = client.get_evaluation(EvaluationId=evaluation_id)
        status = eval['Status']
        message = eval.get('Message','')
        now = str(datetime.datetime.now().time())
        print("Evaluation %s is %s (%s) at %s" % (evaluation_id, status, message, now))
        if status in ['COMPLETED', 'FAILED', 'INVALID']:
            break
            
        time.sleep(delay)
        
def batch_prediction(eval_id, model_id, output_s3, batch_datasource_ID):
    
        #Boto3 'machinelearning' client function to create a batch prediction from a ML model and a batch 
        #datasource.  Function will use an existing model passed as an AWS identifier number and an existing batch
        #datasource passed as an AWS identifier number.  The output of the batch analysis will be written to an 
        #output S3 bucket using the output_s3 argument as the bucket (make sure permissions are set).  The 
        #evaluation ID is only used for the polling function, we don't want to start the batch prediction until 
        #the evaluation is complete (although it would still work)

        poll_evaluation_status(client, eval_id)
        prediction_id = 'predictionID.' + base64.b32encode(os.urandom(5)).decode()
        
        client.create_batch_prediction(
        BatchPredictionId=prediction_id,
        BatchPredictionName="Batch Prediction for MNIST",
        MLModelId=model_id,
        BatchPredictionDataSourceId=batch_datasource_ID,
        OutputUri=output_s3
        )
    
        print("Created batch prediction %s" % prediction_id)
        return prediction_id

    
#Define the Boto3 'machinelearning' client
#Define our training data CSV file that will be split into training and evaluation datasources
#Define our batch data CSV file that will be used to make a batch prediction against our completed model
#Define our training data schema that includes our target value (label)
#Define our recipe JSON file that instructs AWS ML how to treat our input/output data
#Define our batch data schema that does not include the target value (label)
#Define our S3 bucket where the output prediction file will be written (make sure permissions are set on the bucket)
#Define our training dataset split percentage for training the model and evaluating the model
#Define our friendly name for viewing in the AWS ML console
#Obviously edit the S3 bucket location names with your own bucket and make sure permissions are set for AWS ML
    
client = boto3.client('machinelearning')
TRAINING_DATA_S3_URL = "s3://<EDIT.YOUR.BUCKET.GOES.HERE>/MNIST.TRAIN.csv"
BATCH_DATA_S3_URL = "s3://<EDIT.YOUR.BUCKET.GOES.HERE>/MNIST.batch.prediction.csv"
SCHEMA_S3_URL = "s3://<EDIT.YOUR.BUCKET.GOES.HERE>/SCHEMA"
RECIPE_S3_URL = "s3://<EDIT.YOUR.BUCKET.GOES.HERE>/RECIPE"
BATCH_SCHEMA_S3_URL = "s3://<EDIT.YOUR.BUCKET.GOES.HERE>/BATCH_SCHEMA"
S3_RESULTS_OUTPUT_URL = "s3://<EDIT.YOUR.BUCKET.GOES.HERE>/out"
PERCENT = 70
NAME = "MNIST"

#Call each function and assign identification numbers that will be used for calling other functions

(trainID, evalID) = create_training_datasource(client,TRAINING_DATA_S3_URL,SCHEMA_S3_URL,70,NAME)
(modelID) = create_model(client,trainID,RECIPE_S3_URL,NAME)
(evalID) = create_evaluation(client, modelID, evalID, NAME)
(batchID) = create_batch_prediction_dataset(client,BATCH_DATA_S3_URL,BATCH_SCHEMA_S3_URL)
(predictionID) = batch_prediction(evalID, modelID, S3_RESULTS_OUTPUT_URL, batchID)
