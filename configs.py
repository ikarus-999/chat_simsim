# -*- coding: utf-8 -*-
import tensorflow as tf

# Define FLAGS
DEFINES = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('f', '', 'kernel') # UnrecognizedFlagError: Unknown command line flag 'f'
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')  # 배치 크기
tf.app.flags.DEFINE_integer('train_steps', 20000, 'train steps')  # 학습 에포크
tf.app.flags.DEFINE_float('dropout_width', 0.3, 'dropout width')  # 드롭아웃 크기
tf.app.flags.DEFINE_integer('embedding_size', 256, 'embedding size')  # 가중치 크기 # 논문 512 사용
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')  # 학습률
tf.app.flags.DEFINE_integer('shuffle_seek', 1000, 'shuffle random seek')  # 셔플 시드값
tf.app.flags.DEFINE_integer('max_sequence_length', 25, 'max sequence length')  # 시퀀스 길이
tf.app.flags.DEFINE_integer('model_hidden_size', 256, 'model weights size')  # 모델 가중치 크기 128 -> 256
tf.app.flags.DEFINE_integer('ffn_hidden_size', 512, 'ffn weights size')  # ffn 가중치 크기 ->512
tf.app.flags.DEFINE_integer('attention_head_size', 4, 'attn head size')  # 멀티 헤드 크기
tf.app.flags.DEFINE_integer('layer_size', 3, 'layer size')  # 논문은 6개 레이어이나 3개가 시퀀스 생성에 적절함, 심심이로 쓰기에 적절: 적절한
tf.app.flags.DEFINE_string('data_path', 'data_in/ChatBotData.csv', 'data path')  # 데이터 위치
tf.app.flags.DEFINE_string('vocabulary_path', 'data_out/vocabularyData.voc', 'vocabulary path')  # 사전 위치
tf.app.flags.DEFINE_string('check_point_path', 'data_out/check_point', 'check point path')  # 체크 포인트 위치
tf.app.flags.DEFINE_boolean('tokenize_as_morph', True, 'set morph tokenize')  # 형태소에 따른 토크나이징 사용 유무
tf.app.flags.DEFINE_boolean('xavier_initializer', True, 'set xavier initializer')  # Xavier Init
