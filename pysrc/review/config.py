# model structure
d_hidden = 768  # 768
d_bert = 768  # 768

# train
amp_enabled = 1

# additional
seed = 1234

# Paths config
base_path = "~/review"
dataset_path = f"{base_path}/dataset"
weights_path = f"{base_path}/weights"

# Model lookup is in /model and ~/.pubtrends/model folders
model_name = 'learn_simple_berta.pth'
