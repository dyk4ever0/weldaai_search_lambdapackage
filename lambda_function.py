import json
import boto3
from openai import OpenAI
from pinecone import Pinecone
import os

def get_api_key(service, key):
    ssm = boto3.client('ssm')
    parameter = ssm.get_parameter(Name=f"/{service}/{key}/api_key", WithDecryption=True)
    return parameter['Parameter']['Value']

def load_pinecone_index(index_name):
    pinecone_api_key = os.environ['pinecone_api_key']
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    return index
    # existing_indexes = [index['name'] for index in pc.list_indexes()]
    # if index_name in existing_indexes:
    #     return pc.Index(index_name)
    # else:
    #     return None

def request_query(index, query):
    openai_api_key = os.environ['openai_api_key']
    client = OpenAI(api_key=openai_api_key)
    embedded_query = client.embeddings.create(
        input=[query],
        model="text-embedding-3-large",
        dimensions=1024
    )
    results = index.query(
        vector=embedded_query.data[0].embedding,
        top_k=30,
        include_metadata=False
    )
    matches = results['matches']
    #match_idx = [int(match['metadata']['guid_idx']) for match in matches]
    match_idx = [int(match['id']) for match in matches]
    return match_idx

def lambda_handler(event, context):
    query_text = event.get('queryStringParameters', {}).get('query')
    if query_text:
        index = load_pinecone_index('wellda-sample')
        if not index:
            return {
                'statusCode': 500,
                'body': json.dumps('Index not loaded or query handling failed')
            }
        match_idx = request_query(index, query_text)
        return {
            'statusCode': 200,
            'body': json.dumps({'match_idx': match_idx})
        }
    else:
        return {
            'statusCode': 400,
            'body': json.dumps('Query parameter is missing')
        }