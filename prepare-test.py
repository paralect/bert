import pandas as pd

# df_test = pd.read_csv("./yelp_review_full_csv/test.tsv", sep="\t", encoding="UTF-8")

reviews = [
    'I Really Love them much',
    'I hate them',
    'I Really hate them'
]
df_bert_train = pd.DataFrame({
    'id': list(range(len(reviews))),
    'text': reviews
})

df_bert_train.to_csv('./yelp_review_full_csv/test.tsv', sep='\t', index=False, encoding="UTF-8")