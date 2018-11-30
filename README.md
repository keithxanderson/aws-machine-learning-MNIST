# aws-machine-learning-MNIST

Using Amazon Machine Learning to explore the Kaggle MNIST dataset.  This entire process will cost you a few dollars!!  See Amazon's Machine Learning pricing for more information - https://aws.amazon.com/aml/pricing/.

## How to use this code

1) Download the Kaggle versions of the training data and the test data.  We are using the Kaggle version because it is already split for testing, feel free to download MNIST elsewhere but it may take a few edits to the schema files

    Kaggle MNIST training data -> https://www.kaggle.com/c/digit-recognizer/download/train.csv
    
    Kaggle MNIST test data -> https://www.kaggle.com/c/digit-recognizer/download/test.csv
  
2) Upload both the train.csv and test.csv to an S3 bucket in your AWS account.  The train.csv will be used to create a training and evaluation dataset for a multiclass classificaiton model.  The test.csv will be used to create a batch prediction dataset for making a batch prediciton using our completed model.

3) Upload the SCHEMA file and the BATCH_SCHEMA file to an S3 bucket in your AWS account (can be the same as your CSV files).  The schema files will work on the Kaggle MNIST train/test CSV files, if you source the MNIST dataset elsewhere you may need to edit the schemas.  

4) Upload the RECIPE file to an S3 bucket in your AWS account (can be the same as above).  The recipe is not doing anything fancy, all the pixels in MNIST are treated as CATEGORICAL in the schema.  The recipe simply treats the outputs of the model using the default group "ALL_CATEGORICAL".  

5) Edit the MNIST.py file S3 paths for your file location declarations.  The file marks the S3 paths as <EDIT.YOUR.BUCKET.GOES.HERE>.  Also fix the CSV file names to match the files you uploaded to S3.  

6) Edit the MNIST.py file batch prediction output results file path in the declarataions.  This is where the batch prediction output will be stored (the file shows this as <EDIT.YOUR.BUCKET.GOES.HERE>).  

7) The MNIST.py PERCENT is set to 70 for a 70/30% split between training and evaluation datasources.  The code will shuffle and randomly use 70% of the train.cvs for training the model and 30% for evaluating the model.  Change the percent if you want to split differently.  

8) Make sure the Amazon Machine Learning service has IAM access to the S3 bucket that stores the input files (CSV, schema, recipe) and the output bucket for the batch prediction results.  Setting permissions is out of scope for this README.

9) Make sure you have your AWS account credentials defined, Python installed, Boto3 installed, etc.  See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html for getting started, 

10) Run MNIST.py with Python, there are no arguments or user inputs at the moment, everything is hard-coded in the MNIST.py file.  The code will create a training dataset, evaluation dataset, batch prediction dataset, ML model, model evaluation, and batch prediction (in that order).  The code waits for the evaluation to complete with a polling fuction prior to running the batch prediction - this is optional and just keeps things simple.  The evaluation score is not necessary to run a batch prediction!  Also, you will need to log into the console to explore the model performance that results from the evaluation.

11) The batch prediction will output a CSV in the output S3 bucket.  Interpreting the results is documented in AWS, see "Interpreting the Contents of Batch Prediction Files for a Multiclass Classification ML Model"  https://docs.aws.amazon.com/machine-learning/latest/dg/reading-the-batchprediction-output-files.html.  
