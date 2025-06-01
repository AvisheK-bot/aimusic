import boto3
import os
import zipfile
import shutil

def deploy_to_lambda():
    # Initialize AWS clients
    lambda_client = boto3.client('lambda')
    s3_client = boto3.client('s3')
    
    # Create a temporary directory for the deployment package
    os.makedirs('deployment', exist_ok=True)
    
    # Copy required files
    shutil.copy('lambda_function.py', 'deployment/')
    shutil.copy('requirements-sagemaker.txt', 'deployment/')
    
    # Create a zip file for the deployment package
    with zipfile.ZipFile('deployment.zip', 'w') as zipf:
        for root, dirs, files in os.walk('deployment'):
            for file in files:
                zipf.write(os.path.join(root, file), file)
    
    # Upload model files to S3
    bucket_name = 'your-bucket-name'  # Replace with your bucket name
    s3_client.upload_file('model/model.joblib', bucket_name, 'model/model.joblib')
    s3_client.upload_file('model/scaler.joblib', bucket_name, 'model/scaler.joblib')
    s3_client.upload_file('model/song_metadata.csv', bucket_name, 'model/song_metadata.csv')
    
    # Create or update Lambda function
    try:
        with open('deployment.zip', 'rb') as f:
            zip_bytes = f.read()
        
        response = lambda_client.create_function(
            FunctionName='music-recommender',
            Runtime='python3.9',
            Role='arn:aws:iam::YOUR_ACCOUNT_ID:role/lambda-role',  # Replace with your role ARN
            Handler='lambda_function.lambda_handler',
            Code={'ZipFile': zip_bytes},
            Timeout=30,
            MemorySize=512,
            Environment={
                'Variables': {
                    'BUCKET_NAME': bucket_name
                }
            }
        )
        print(f"Lambda function created: {response['FunctionArn']}")
        
        # Create API Gateway
        api_client = boto3.client('apigateway')
        api_response = api_client.create_rest_api(
            name='MusicRecommenderAPI',
            description='API for Music Recommender'
        )
        
        # Create resource and method
        resource_response = api_client.create_resource(
            restApiId=api_response['id'],
            parentId=api_response['rootResourceId'],
            pathPart='recommend'
        )
        
        api_client.put_method(
            restApiId=api_response['id'],
            resourceId=resource_response['id'],
            httpMethod='POST',
            authorizationType='NONE'
        )
        
        # Integrate with Lambda
        api_client.put_integration(
            restApiId=api_response['id'],
            resourceId=resource_response['id'],
            httpMethod='POST',
            type='AWS_PROXY',
            integrationHttpMethod='POST',
            uri=f"arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/{response['FunctionArn']}/invocations"
        )
        
        # Deploy the API
        deployment_response = api_client.create_deployment(
            restApiId=api_response['id'],
            stageName='prod'
        )
        
        print(f"API Gateway URL: https://{api_response['id']}.execute-api.us-east-1.amazonaws.com/prod/recommend")
        
    except Exception as e:
        print(f"Error deploying Lambda function: {str(e)}")
    
    # Clean up
    shutil.rmtree('deployment')
    os.remove('deployment.zip')

if __name__ == '__main__':
    deploy_to_lambda() 