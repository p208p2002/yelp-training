from ReviewDataset import ReviewDataset
import torch
from ReviewClassifier import ReviewClassifier
import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text

def predict_rating(review, classifier, vectorizer, decision_threshold=0.5):
    """Predict the rating of a review
    
    Args:
        review (str): the text of the review
        classifier (ReviewClassifier): the trained model
        vectorizer (ReviewVectorizer): the corresponding vectorizer
        decision_threshold (float): The numerical boundary which separates the rating classes
    """
    review = preprocess_text(review)
    
    vectorized_review = torch.tensor(vectorizer.vectorize(review))
    result = classifier(vectorized_review.view(1, -1))
    
    probability_value = torch.sigmoid(result).item()
    index = 1
    if probability_value < decision_threshold:
        index = 0

    return vectorizer.rating_vocab.lookup_index(index)

dataset = ReviewDataset.load_dataset_and_load_vectorizer('reviews_with_splits_lite.csv','vectorizer.json')
vectorizer = dataset.get_vectorizer()
model = ReviewClassifier(num_features=len(vectorizer.review_vocab))
model.load_state_dict(torch.load('model.pth'))
model.eval()

test_review = "this is a pretty awesome book"
model = model.cpu()
prediction = predict_rating(test_review, model, vectorizer, decision_threshold=0.5)
print("{} -> {}".format(test_review, prediction))