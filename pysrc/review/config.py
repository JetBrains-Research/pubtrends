# model structure
d_hidden = 768  # 768
d_bert = 768  # 768
n_layers = 6  # 6 or 12
n_heads = 6  # 6
dropout = 0.3  # 0.2
d_ff = 2048
pe_maxlen = 5000

# train
label_smoothing = True
amp_enabled = 1
num_workers = 0

# eval
draft_strategy = 'beam'  # 'beam' or 'greedy' or 'top-k' or 'top-p' or 'topk_beam'
temperature = 1.0  # != 1.0 to utilize temperature sampling
eliminate_trigrams = True
beam_size = 5  # 4
beam_lenpenalty = 4.5  # 5.0
topk_k = 2
topp_p = 0.7
n_worst2writer = 20
n_random2writer = 100

# additional
seed = 1234

base_path = "../summarization"
data_path = f"{base_path}/data"
weights_path = f"{base_path}/weights"
log_filepath = f"{base_path}/logs/log.log"
tb_logdir = f"{base_path}/logs/tb_runs"
predicted_path = "summarized_texts"