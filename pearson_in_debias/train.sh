python main.py --data_dir "../data/ml-1m" --data_name "movie" --vocab_file "vocab.json" --epoch_num 50 --train --batch_size 64 --learning_rate 0.001 --print_interval 50 --optimizer "Adam" --user_emb_size 128 --item_emb_size 128 --weight_decay 0.00