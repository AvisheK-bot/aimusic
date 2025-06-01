import sagemaker
from sagemaker.sklearn import SKLearn
import boto3
import os

def deploy_to_sagemaker():
    # Initialize SageMaker session
    sess = sagemaker.Session()
    role = sagemaker.get_execution_role()
    
    # Create S3 bucket and upload model files
    bucket = sess.default_bucket()
    prefix = 'music-recommender'
    
    # Upload model files to S3
    model_data = sess.upload_data(
        path='model',
        bucket=bucket,
        key_prefix=f'{prefix}/model'
    )
    
    # Create SKLearn estimator
    sklearn = SKLearn(
        entry_point='sagemaker_endpoint.py',
        role=role,
        instance_type='ml.t2.medium',
        framework_version='1.0-1',
        py_version='py3',
        source_dir='.',
        dependencies=['requirements-sagemaker.txt']
    )
    
    # Deploy the model
    predictor = sklearn.deploy(
        initial_instance_count=1,
        instance_type='ml.t2.medium',
        endpoint_name='music-recommender-endpoint'
    )
    
    print(f"Endpoint deployed successfully: {predictor.endpoint_name}")
    return predictor

if __name__ == '__main__':
    deploy_to_sagemaker() 