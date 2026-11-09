from typing import List, Dict
import re
import unicodedata

from rapidfuzz.distance import Levenshtein


_ws_re = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    """
    Normalizes the given text. 

    Converts text to the NFKC normalization form. 
    Replaces typography characters with the most common versions.
    Handles trailing or leading whitespace, and lowercases the text.

    Args:
        s (str): original str

    Returns:
        str: normalized str
    """
    # If string was empty 
    if s is None:
        return ""
    
    # Normalize
    s = unicodedata.normalize("NFKC", s)

    # Apply lowercase and strip
    s = s.lower().strip()

    # Replace common characters with a single one 
    s = s.replace("’", "'").replace("‘", "'").replace("´", "'").replace("`", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = _ws_re.sub(" ", s)
    return s


def calculate_normalized_levenshtein_distance(target_answer: str, pred_answer: str) -> float: 
    # Normalize text
    target_answer = normalize_text(target_answer)
    pred_answer = normalize_text(pred_answer)

    # Check if both are empty, and if they are then return 1 for perfect match
    if target_answer == "" and pred_answer == "":
        return 1.0
    
    # Check if one of them is not, then we have no match at all
    if target_answer == "" or pred_answer == "":
        return 0.0

    # Calculate levenshtein distance 
    levenshtein_distance = Levenshtein.distance(target_answer, pred_answer)

    # Calculate the max length of the strings 
    max_len = max(len(target_answer), len(pred_answer))    

    # calculate the normalized levenshtein distance 
    norm_dist = 1.0 - (levenshtein_distance / max_len)
    return norm_dist

def calculate_anls(target_answers: List[str], pred_answer: str, threshold:float = 0.5):
    max_similarity = 0.0
    for target_answer in target_answers:
        # calculate the ANLS
        similarity = calculate_normalized_levenshtein_distance(pred_answer=pred_answer, target_answer=target_answer)

        # set it to max anls 
        max_similarity = max(max_similarity, similarity)
            
    # Return the maximum similarity if the score is above 0.5
    return max_similarity if max_similarity >= threshold else 0.0


def anls_metric(target_answers: List[List[str]], pred_answers: List[str]) -> float:
    """
    ANLS metric averaged over the batch of answers.
    Evaluation metric from the original DocVQA paper.
    """
    
    # Check default case
    if len(target_answers) == 0 or len(pred_answers) == 0 or len(target_answers) != len(pred_answers):
        return 0.0

    # Calculate per example
    total = 0.0
    batch_size = len(pred_answers)
    for targets, pred in zip(target_answers, pred_answers):
        anls = calculate_anls(target_answers=targets, pred_answer=pred)
        total += anls

    # Calculate the average ANLS from the batch 
    average_anls = total / batch_size
    return average_anls


def calculate_accuracy(target_answers: List[str], pred_answer: str) -> float:
    """
    Calculates the accuracy for the example.
    Returns 1 if any of the target answer matches exactly.

    Normalizes the prediction text and compare it against normalized target text. 

    Args:
        target_answers (List[str]): List of the answers for the question
        pred_answer (str): the predicted answer 

    Returns:
        float: 1 if any match or 0 else
    """
    pa = normalize_text(pred_answer)
    for ta in target_answers:
        if normalize_text(ta) == pa:
            return 1.0
    return 0.0

def accuracy_metric(target_answers: List[List[str]], pred_answers: List[str]) -> float:
    """
    Accuracy from the original DocVQA paper.
    Per query there are multiple answers. The percentage is then averaged and returned 
    """

    # Check default case
    if len(target_answers) == 0 or len(pred_answers) == 0 or len(target_answers) != len(pred_answers):
        return 0.0

    # For each query in the batch calculate the match percentage 
    total_match_percentage = 0.0
    for targets, pred in zip(target_answers, pred_answers):
        acc = calculate_accuracy(target_answers=targets, pred_answer=pred)
        total_match_percentage += acc

    # return the average 
    return total_match_percentage / len(pred_answers)


class DocVQAEvaluator:
    """ 
    DocVQA Evaluation metrics implemented from the original DocVQA paper.
    """
    @staticmethod
    def evaluate(target_answers: List[List[str]], pred_answers: List[str]) -> Dict[str, float]:
        # Check if there is a prediction for each answer at least
        if len(target_answers) != len(pred_answers):
            raise ValueError("target_answers and pred_answers did not have the same length for DocVQA Evaluation")

        # calculate the accuracy metric 
        acc = accuracy_metric(target_answers=target_answers, pred_answers=pred_answers)

        # Calculate ANLS metric 
        anls = anls_metric(target_answers=target_answers, pred_answers=pred_answers)

        return {
            "acc": acc,
            "anls": anls
        }
