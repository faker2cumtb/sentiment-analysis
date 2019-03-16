#!/usr/bin/env bash

#python sentiment_classify.py --model_path models/save/test.h5 --mode train --sql_file_path experiment_record/sql/db_sentiment_corpus_3.sql  --epochs 20 --model_type cnn_lstm -g train_in_layer

#python sentiment_classify.py  --model_path models/save/cnn_lstm_all_corpus_2_v1.h5 --mode train --sql_file_path experiment_record/sql/db_sentiment_corpus_2.sql  --epochs 20 --model_type cnn_lstm -g train_in_layer --sql_file_path_eval_when_train experiment_record/sql/db_sentiment_corpus_3.sql

models=('cnn_1_layer' 'cnn_2_layers' 'simple_nn' 'simple_lstm' 'bi_lstm' 'cnn_lstm' 'stacked_lstm' 'fasttext')
for model in ${models[@]};do
model_name="$model"_all_corpus_2_v3
echo $model_name


#python sentiment_classify.py  --model_path models/save/$model_name.h5 --mode train --sql_file_path experiment_record/sql/db_sentiment_corpus_2.sql  --epochs 20 --model_type cnn_lstm -g train_in_layer --sql_file_path_eval_when_train experiment_record/sql/db_sentiment_corpus_3.sql

python sentiment_classify.py  --model_path models/save/$model_name.h5 --mode train --data_file_path data/db_sentiment_corpus_2.txt  --epochs 20 --model_type bilstm -g train_in_layer --data_file_path_eval_when_train data/test/db_sentiment_corpus_3.txt

python funcs_test.py eval $model_name > results/$model_name



done
