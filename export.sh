python run_classifier.py \
    --task_name=cola \
    --do_export=true \
    --data_dir=./yelp_review_full_csv/ \
    --vocab_file=./weights_base/uncased_L-12_H-768_A-12/vocab.txt \
    --bert_config_file=./weights_base/uncased_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=./bert_output_base/model.ckpt-60328 \
    --max_seq_length=128 \
    --output_dir=./bert_output_base/ \
    --export_dir="./bert_model/"
