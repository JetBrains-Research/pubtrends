import nltk


def split_text(text):
    return nltk.tokenize.sent_tokenize(text)


def preprocess_text(text, max_len, tokenizer):
    sents = [[tokenizer.artBOS.tkn] + tokenizer.tokenize(sent) + [tokenizer.artEOS.tkn]
             for sent in text]
    ids, segments, segment_signature = [], [], 0
    n_sents = 0
    for s in sents:
        if len(ids) + len(s) <= max_len:
            n_sents += 1
            ids.extend(tokenizer.convert_tokens_to_ids(s))
            segments.extend([segment_signature] * len(s))
            segment_signature = (segment_signature + 1) % 2
        else:
            break
    mask = [1] * len(ids)

    pad_len = max(0, max_len - len(ids))
    ids += [tokenizer.PAD.idx] * pad_len
    mask += [0] * pad_len
    segments += [segment_signature] * pad_len

    return ids, mask, segments, n_sents


def text_to_data(text, max_len, tokenizer):
    text = split_text(text)
    total_sents = 0
    data = []
    while total_sents < len(text):
        magic = max(0, total_sents - 5)
        article_ids, article_mask, article_segment, n_setns = \
            preprocess_text(text[magic:], max_len, tokenizer)
        if magic + n_setns <= total_sents:
            total_sents += 1
            continue
        data.append((article_ids, article_mask, article_segment, total_sents - magic, text[magic:magic + n_setns]))
        total_sents = magic + n_setns
    return data


def convert_token_to_id(tokenizer, tkn):
    return tokenizer.convert_tokens_to_ids([tkn])[0]
