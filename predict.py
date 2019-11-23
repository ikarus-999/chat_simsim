import tensorflow as tf
import data
import os
import sys
import model as ml

from configs import DEFINES
from flask import Flask, request
import json, requests
app = Flask(__name__)

# 챗봇 클래스
class ChatBot(object):

    # 초기화
    def __init__(self):
        self.FACEBOOK_TOKEN = 'FACEBOOK_TOKEN'
        self.VERIFY_TOKEN = 'VERIFY_TOKEN'
        self.FBM_API = "https://graph.facebook.com/v2.6/me/messages"
        self.fbm_processed = []

    # 메시지 처리
    def process_fbm(self, payload):
        for sender, msg in self.fbm_events(payload):
            resp = self.response(msg)
            self.fbm_api({"recipient": {"id": sender}, "message": {"text": resp}})

    # 메시지 생성
    def response(self, msg):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        if __name__ == '__main__':
            tf.logging.set_verbosity(tf.logging.ERROR)

            # 데이터를 통한 사전 구성 한다.
            char2idx, idx2char, vocabulary_length = data.load_vocabulary()

            # 테스트용 데이터 만드는 부분이다.
            # 인코딩 부분 만든다.
            input_txt = msg #input()
            print(input_txt)
            predic_input_enc, predic_input_enc_length = data.enc_processing([input_txt], char2idx)
            # 학습 과정이 아니므로 디코딩 입력은
            # 존재하지 않는다.(구조를 맞추기 위해 넣는다.)
            predic_output_dec, predic_output_dec_length = data.dec_output_processing([""], char2idx)
            # 학습 과정이 아니므로 디코딩 출력 부분도
            # 존재하지 않는다.(구조를 맞추기 위해 넣는다.)
            predic_target_dec = data.dec_target_processing([""], char2idx)

            # 에스티메이터 구성한다.
            classifier = tf.estimator.Estimator(
                model_fn=ml.Model,  # 모델 등록한다.
                model_dir=DEFINES.check_point_path,  # 체크포인트 위치 등록한다.
                params={  # 모델 쪽으로 파라메터 전달한다.
                    'model_hidden_size': DEFINES.model_hidden_size,  # 가중치 크기 설정한다.
                    'ffn_hidden_size': DEFINES.ffn_hidden_size,
                    'attention_head_size': DEFINES.attention_head_size,
                    'learning_rate': DEFINES.learning_rate,  # 학습율 설정한다.
                    'vocabulary_length': vocabulary_length,  # 딕셔너리 크기를 설정한다.
                    'embedding_size': DEFINES.embedding_size,  # 임베딩 크기를 설정한다.
                    'layer_size': DEFINES.layer_size,
                    'max_sequence_length': DEFINES.max_sequence_length,
                    'xavier_initializer': DEFINES.xavier_initializer
                })
            # 예측을 하는 부분이다.
            predictions = classifier.predict(input_fn=lambda: data.eval_input_fn(predic_input_enc, predic_output_dec, predic_target_dec, 1))

            answer, finished = data.pred_next_string(predictions, idx2char)

            # 예측한 값을 인지 할 수 있도록
            # 텍스트로 변경하는 부분이다.
            print("answer: ", answer)
            return answer

    # 메시지 이벤트 처리
    def fbm_events(self, payload):
        data = json.loads(payload.decode('utf8'))

        for event in data["entry"][0]["messaging"]:
            if "message" in event and "text" in event["message"]:
                q = (event["sender"]["id"], event["message"]["mid"])

                if q in self.fbm_processed:
                    continue
                else:
                    self.fbm_processed.append(q)
                    yield event["sender"]["id"], event["message"]["text"]

    # 페이스북 API로 메시지 전송
    def fbm_api(self, data):
        r = requests.post(self.FBM_API,
                          params={"access_token": self.FACEBOOK_TOKEN},
                          data=json.dumps(data),
                          headers={'Content-type': 'application/json'})

        if r.status_code != requests.codes.ok:
            print("fb error:", r.text)

# 검증 함수
@app.route('/', methods=['GET'])
def Verify():
    if request.args.get('hub.verify_token', '') == bot.VERIFY_TOKEN:
        return request.args.get('hub.challenge', '')
    else:
        return 'Error, wrong validation token'


# Webhook 함수
@app.route('/', methods=['POST'])
def Webhook():
    payload = request.get_data()
    bot.process_fbm(payload)
    return "ok"

# 메인 함수
if __name__ == "__main__":
    bot = ChatBot()
    # contextSSL = ('server.crt', 'server.key')
    app.run(host='0.0.0.0', port=5000)  # ssl_context=contextSSL)
