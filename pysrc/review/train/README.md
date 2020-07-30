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

