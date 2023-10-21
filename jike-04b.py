from transformers import T5Tokenizer, T5Model
import torch
from openai.embeddings_utils import cosine_similarity
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import PrecisionRecallDisplay

## 通过T5模型进行数据向量化判定


# load the T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=512)
model = T5Model.from_pretrained('t5-base')

# set the model to evaluation mode
model.eval()

# encode the input sentence
def get_t5_vector(line):
    input_ids = tokenizer.encode(line, return_tensors='pt', max_length=512, truncation=True)
    # generate the vector representation
    with torch.no_grad():
        outputs = model.encoder(input_ids=input_ids)
        vector = outputs.last_hidden_state.mean(dim=1)
    return vector[0]
  
  
positive_review_in_t5 = get_t5_vector("An Amazon review with a positive sentiment.")
negative_review_in_t5 = get_t5_vector('An Amazon review with a negative sentiment.')
positive_text = """Wanted to save some to bring to my Chicago family but my North Carolina family ate all 4 boxes before I could pack. These are excellent...could serve to anyone"""
negative_text = """First, these should be called Mac - Coconut bars, as Coconut is the #2 ingredient and Mango is #3. Second, lots of people don't like coconut. I happen to be allergic to it. Word to Amazon that if you want happy customers to make things like this more prominent. Thanks."""

datafile_path = "data/fine_food_reviews_with_embeddings_1k.csv"

df = pd.read_csv(datafile_path)
# df["embedding"] = df.embedding.apply(eval).apply(np.array)
df["embedding_t5"] = df["Text"].apply(get_t5_vector)

# convert 5-star rating to binary sentiment
df = df[df.Score != 3]
df["sentiment"] = df.Score.replace({1: "negative", 2: "negative", 4: "positive", 5: "positive"})



def evaluate_embeddings_approach(
    labels = ['negative', 'positive'], 
):
    label_embeddings = [get_t5_vector(label) for label in labels]

    def label_score(review_embedding, label_embeddings):
        return cosine_similarity(review_embedding, label_embeddings[1]) - cosine_similarity(review_embedding, label_embeddings[0])

    probas = df["embedding_t5"].apply(lambda x: label_score(x, label_embeddings))
    preds = probas.apply(lambda x: 'positive' if x>0 else 'negative')

    report = classification_report(df.sentiment, preds)
    print(report)

    display = PrecisionRecallDisplay.from_predictions(df.sentiment, probas, pos_label='positive')
    _ = display.ax_.set_title("2-class Precision-Recall curve")

evaluate_embeddings_approach(labels=['An Amazon review with a negative sentiment.', 'An Amazon review with a positive sentiment.'])

