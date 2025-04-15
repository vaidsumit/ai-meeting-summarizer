# AI Meeting Summarizer

This is a Python and AI / ML utility that:
- Extracts audio from MP4 videos
- Transcribes using AWS Transcribe
- Summarizes the transcript using Amazon Bedrock's Titan model

## Usage
run app.py to start a server on localhost:3000
browse http://localhost:3000/ on local browser.. this will bring up a tool to browse your mp4 file and once clicked will generate a summary..

# note:
you need to have access to AWS S3, AWS Transcribe and AWS Bedrock Titan lite..
You may need to change these settings:
 line 12: if you are using a different region, as I am using the one here ( REGION = 'us-east-1')
 line 13 for S3 bucket name ( as of now, I am using mine as  'my-meeting-audio-bucket-svaid01')
 line 93: any specific model you are using - if other than Titan lite -- modelId="amazon.titan-text-lite-v1",

# setup:
pip install moviepy boto3 requests

# Requirements
Python 3.8+
AWS credentials with access to:
S3
Transcribe
Bedrock


# Author:
Sumit Vaid
