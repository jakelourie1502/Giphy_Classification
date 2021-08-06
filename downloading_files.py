import boto3
import argparse
parser = argparse.ArgumentParser(description='get some keys.')
parser.add_argument('aws key', metavar='key', type=str)
parser.add_argument('aws secret key', metavar='secret_key', type=str)

aws_access_key_id, aws_secret_access_key = vars(parser.parse_args()).values()

BUCKET_NAME = 'giphy-classification-bucket'
idx = 1
aws_acces
r3 = boto3.resource('s3',
    aws_access_key_id = aws_access_key_id,
    aws_secret_access_key = aws_secret_access_key,
    region_name = 'eu-west-2')

my_bucket = r3.Bucket(BUCKET_NAME)
file_list = []
for file in my_bucket.objects.filter(Prefix='padded_ims'):
    if file.key[-4:] == '.mp4':
        file_list.append(file.key)
print(len(file_list))
print(file_list[0])

for file_name in file_list:
    r3.Bucket(BUCKET_NAME).download_file(file_list[0], f'/home/ubuntu/{file_name}')

# import av
# import torchvision
# reader = torchvision.io.read_video(f'/home/ubuntu/{file_list[0]}')
# print(reader[0].shape)
