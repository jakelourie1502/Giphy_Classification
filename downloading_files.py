import boto3

BUCKET_NAME = 'giphy-classification-bucket'
idx = 1
r3 = boto3.resource('s3',
    aws_access_key_id = 'AKIAQEASK5DNYV56GQUK',
    aws_secret_access_key = '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',
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
