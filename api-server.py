from fastapi import FastAPI
from pydantic import BaseModel
import json
import os
import requests
import tokenization

app = FastAPI()

class TextContainer(BaseModel):
    text: str

VOCAB_FILE_PATH = os.getenv('VOCAB_FILE') or "./weights_base/uncased_L-12_H-768_A-12/vocab.txt"
SERVE_API = os.getenv('SERVE_API_HOST') or "localhost"
SERVE_API_PORT = os.getenv('SERVE_API_PORT') or "8501"

@app.post('/predict')
def predict(body: TextContainer):
    serve_endpoint = f"http://{SERVE_API}:{SERVE_API_PORT}/v1/models/bert:predict"
    headers = {"content-type":"application-json"}
    tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE_PATH, do_lower_case=True)
    token_a = tokenizer.tokenize(body.text)
    tokens = []
    segments_ids = []
    tokens.append("[CLS]")
    segment_ids = []
    segment_ids.append(0)

    for token in token_a:
        tokens.append(token)
        segment_ids.append(0)

    tokens.append('[SEP]')
    segment_ids.append(0)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    max_seq_length = 128

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    label_id = 0
    instances = [{"input_ids":input_ids, "input_mask":input_mask, "segment_ids":segment_ids, "label_ids":label_id}]
    data = json.dumps({"signature_name":"serving_default", "instances":instances})
    response = requests.post(serve_endpoint, data=data, headers=headers)
    prediction = json.loads(response.text)['predictions']

    return {
        'predictions': prediction
    }
