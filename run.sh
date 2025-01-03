#bash data_download.sh
#python create_merged_data.py
python embedding/temporal_aisles.py
#python embedding/temporal_products.py
#python embedding/temporal_reorder.py
#python embedding/nmf.py
#python ml_feat/non_self_prob.py --mode 1
#python ml_feat/non_self_prob.py --mode 0
#python ml_feat/interval_days.py --mode 1
#python ml_feat/interval_days.py --mode 0
#python ml_feat/popup_portion.py --mode 1
#python ml_feat/popup_portion.py --mode 0
#python ml_feat/streak.py --mode 1
#python ml_feat/streak.py --mode 0
#python ml_feat/one_shot_item.py --mode 1
#python ml_feat/one_shot_item.py --mode 0
#python ml_feat/N_order.py --mode 1
#python ml_feat/N_order.py --mode 0
#python ml_feat/cart_order.py --mode 1
#python ml_feat/cart_order.py --mode 0
#python ml_feat/interval_orders.py --mode 1
#python ml_feat/interval_orders.py --mode 0
#python ml_feat/time_distribution.py --mode 1
#python ml_feat/time_distribution.py --mode 0
#python ml_feat/consecutive_runs.py --mode 1
#python ml_feat/consecutive_runs.py --mode 0
python utils/submitter.py
