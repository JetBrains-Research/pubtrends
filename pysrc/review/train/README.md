## Instructions

Here are the instructions on how to create the dataset and then how to run different models training on this dataset. Now, the easiest way to do it is to use python notebook `test_new_algo.ipynb`. There are 18 steps there. 

Soon the instructions for running everything from command line will be added.

### Create a dataset

You can create a dataset with features or without them. 

To generate a dataset look at the steps 3-8 (+10 in case you want to divide the dtaset on train-test-val on the spot). Some of them are not needed in case you create a dataset without features. The model with features is still being developed, you can try it but now it works worse.

### Train the model

Steps 1-2, 9, 10 (to get the dataset), 11-14 are needed. 

You can save the trained model at any time, using function `model.save("name_of_the_file")`

Also, try different parameters for the model (you can either freeze some layers, or not, you can try a model with features)

### See how the model works

This is what the step 15 is for. You can just try the model on different examples.

### Evaluation

These are the last steps. 16 and 17 are to look at MSE score and to understand, how good it is compared to the dataset, while step 18 is to perform a test of summarizing several scientific papaers into a review one.



<!--

- preprocess data

```bash
preprobach.sh
```

- run

```bash
python -W ignore -m main \
                   -lr_decoder 0.01 \
                   -lr_encoder 0.001 \
                   -dataset pubmed \
                   -article_len 1500 \
                   -epochs 2 \
                   -warmup 3000 \
                   -weight_decay 0.0 \
                   -clip_value 10.0 \
                   -batch_size 30 \
                   -accumulation_interval 10 \
                   -valid_interval 5 \
                   -model_type bert \
                   -distributed false \
                   -froze_strategy froze_all \
                   -bert_strategy lastlayer \
                   -mode trainval \
                   -save_filename extbert2 \
                   -tb_tag extbertgo2
```

- check train

```bash
python -W ignore -m main \
                   -lr_decoder 0.007 \
                   -lr_encoder 0.0002 \
                   -dataset pubmed \
                   -article_len 1500 \
                   -epochs 10 \
                   -warmup 10000 \
                   -weight_decay 0.0 \
                   -clip_value 10.0 \
                   -batch_size 8 \
                   -accumulation_interval 8 \
                   -valid_interval 1 \
                   -model_type bert \
                   -distributed false \
                   -froze_strategy unfroze_all \
                   -bert_strategy lastlayer \
                   -mode trainval \
                   -save_filename good_bert \
                   -tb_tag extbert
```

- check eval

```bash
python -W ignore -m main \
                   -dataset pubmed \
                   -article_len 1500 \
                   -batch_size 30 \
                   -model_type bert \
                   -distributed false \
                   -mode test \
                   -save_filename good_bert \
                   -tb_tag TEST
```



- for topic summarization:

1. Prepare test dataset by running topic_summarization.ipynb
2. Run 

```bash
python -W ignore -m main \
                   -dataset pubmed \
                   -article_len 1500 \
                   -batch_size 30 \
                   -model_type bert \
                   -distributed false \
                   -mode test_topic \
                   -save_filename good_bert \
                   -tb_tag TEST
```

-->
