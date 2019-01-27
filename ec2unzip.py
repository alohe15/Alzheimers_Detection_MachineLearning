#ec2unzip

import boto3
from boto3.session import Session
import os
import gzip

bucketname = 'adnimcinc'
s3_client = boto3.client('s3')

def do_work(s3obj):
    #download the file as 's3obj'_dw
    s3_client.download_file(bucketname, s3obj.key, 'tempzip')

    #unzip
    rawFile = gzip.GzipFile('tempzip', 'rb')
    s = rawFile.read()
    rawFile.close()
    with open('tempxtr', 'w') as f:
        f.write(s)

    #uploading
    s3_client.upload_file('tempxtr', bucketname, s3obj.key + '_extracted')
    #delete local file
    os.remove('tempxtr')

#connecting to S3
"""conn = boto3.s3.connection.S3Connection('AKIAJ5IB6H4DWEG7GPJQ','11X0GXDoqys7ZdEZkBCa58MpA0Ir/BcQ61CrhhyK')
bucket = conn.get_bucket(bucketname)"""

session = Session(aws_access_key_id = 'AKIAIZ6K3YPMUWJO7YJQ', aws_secret_access_key = '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')
s3 = session.resource('s3')
bucket = s3.Bucket(bucketname)
for key in bucket.objects.all():
    do_work(key)
