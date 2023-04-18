from typing import List, Dict
from datasets import Dataset, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel, DataCollatorWithPadding
from tqdm import tqdm
import numpy as np
from math import ceil
import string

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.translate.bleu_score import corpus_bleu
from nltk.stem import porter
from nltk import ngrams

from rouge_score import tokenize

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from metrics.bart_score import BARTScorer
from metrics.summac.summac.model_summac import SummaCZS
from metrics.infolm import InfoLM


def decode(eval_preds, tokenizer):
    predictions, labels, *inputs = eval_preds
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    if len(inputs) > 0:
        input_ids = inputs[0]
        input_ids = np.where(input_ids != -100, input_ids, tokenizer.pad_token_id)
        decoded_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        decoded_texts = [text.strip() for text in decoded_texts]
        return decoded_preds, decoded_labels, decoded_texts
        
    return decoded_preds, decoded_labels


def smoothing_function(p_n, references, hypothesis, hyp_len):
    """
    Smooth-BLEU (BLEUS) as proposed in the paper:
    Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
    evaluation metrics for machine translation. COLING 2004.
    """
    smoothed_p_n = []
    for i, p_i in enumerate(p_n, start=1):
        # Smoothing is not applied for unigrams
        if i > 1:
            # If hypothesis length is lower than the current order, its value equals (0 + 1) / (0 + 1) = 0
            if hyp_len < i:
                assert p_i.denominator == 1
                smoothed_p_n.append(1)
            # Otherwise apply smoothing
            else:
                smoothed_p_i = (p_i.numerator + 1) / (p_i.denominator + 1)
                smoothed_p_n.append(smoothed_p_i)
        else:
            smoothed_p_n.append(p_i)
    return smoothed_p_n

def pair_bleu(references, prediction):
    """
    Compute the bleu score between two given texts.
    A smoothing function is used to avoid zero scores when
    there are no common higher order n-grams between the
    texts.
    """
    tok_ref = [word_tokenize(sent) for sent in sent_tokenize(references)]
    tok_pred = [word_tokenize(sent) for sent in sent_tokenize(prediction)]
    score = 0
    for c_cent in tok_pred:
        try:
            score += corpus_bleu(
                [tok_ref], [c_cent], smoothing_function=smoothing_function
            )
        except KeyError:
            score = 0.0
    try:
        score /= len(tok_pred)
    except ZeroDivisionError:
        score = 0.0

    return score


def calculate_bart_score(preds, refs=None, texts=None, scorer=None, batch_size=4, aggregate=True):
    if scorer is None:
        scorer = BARTScorer()
    scores = {}
    if texts is not None:
        scores["BARTScore-sh"] = np.array(scorer.score(texts, preds, batch_size=batch_size))
    if refs is not None:
        scores["BARTScore-rh"] = np.array(scorer.score(refs, preds, batch_size=batch_size))
        scores["BARTScore-hr"] = np.array(scorer.score(preds, refs, batch_size=batch_size))
        scores["BARTScore-fa"] = (scores["BARTScore-rh"] + scores["BARTScore-hr"]) / 2

    if aggregate:
        scores = {key: np.mean(value) for key, value in scores.items()}
    return scores

def calculate_summac_score(
    predictions: List[str], texts: List[str], labels: List[str] = None, aggregate: bool = True
) -> Dict[str, np.ndarray]:
    scorer = SummaCZS(granularity="sentence", model_name="vitc")
    preds_score = scorer.score(texts, predictions)["scores"]
    if labels is not None:
        labels_score = scorer.score(texts, labels)["scores"]
        rel_score = np.array(preds_score) / np.array(labels_score)
    if aggregate:
        preds_score = np.mean(preds_score)
        if labels is not None:
            rel_score = np.mean(rel_score)
    if labels is not None:
        return {"SummaC-tp": preds_score, "SummaC-rel": rel_score}
    return {"SummaC-tp": preds_score}

def calculate_cola_model_predictions(
    texts,
    checkpoint="Aktsvigun/electra-large-cola",
    batch_size=64,
    device="cuda",
    return_sent_data: bool = False,
    aggregate: bool = True
):
    model = AutoModelForSequenceClassification.from_pretrained("Aktsvigun/electra-large-cola").to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    text_sentences = [nltk.sent_tokenize(text) for text in texts]
    len_maps = np.cumsum([len(x) for x in text_sentences])
    sentences = [sent for text in text_sentences for sent in text]
    
    def tokenize_fn(instance):
        return tokenizer(instance["text"], truncation=True)
    
    tokenized_data = Dataset.from_dict({"text": sentences}).map(tokenize_fn, remove_columns=["text"], batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(
        tokenized_data, batch_size=batch_size, shuffle=False, collate_fn=data_collator
    )
    
    sent_probas = torch.empty(len(sentences), dtype=torch.float32, device=device)
    probas = torch.empty(len(texts), dtype=torch.float32, device=device)
    start = 0
    end = batch_size
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            batch_pred = model(**{k: v.cuda() for k, v in batch.items()})
            batch_probas = 1 / (1 + (-batch_pred.logits[:, 1]).exp())
            sent_probas[start:end].copy_(batch_probas)
            start = end
            end += batch_size
    
    for i, end_idx in enumerate(len_maps):
        start_idx = len_maps[i - 1] if i != 0 else 0
        probas[i].copy_(sent_probas[start_idx : end_idx].mean())
        
    if aggregate:
        return probas.mean().item()
    if return_sent_data:
        return probas.cpu().detach().numpy(), sent_probas.cpu().detach().numpy(), sentences
    return probas.cpu().detach().numpy()

def calculate_infolm_score(predictions, references, batch_size=4):
    
    assert len(predictions) == len(references), "Lengths must coincide!"
    infolm_fr = InfoLM(measure_to_use='fisher_rao')
    infolm_ab = InfoLM(measure_to_use='ab', alpha=1., beta=1.)
    
    infolm_fr_scores, infolm_ab_scores = [], []
    idf_ref, idf_hyps = infolm_fr.prepare_idfs(references, predictions)
    
    num_batches = ceil(len(predictions) / batch_size)
    for i in tqdm(range(num_batches)):
        batch_preds = predictions[i * batch_size : (i + 1) * batch_size]
        batch_refs = references[i * batch_size : (i + 1) * batch_size]
    
        infolm_fr_scores += infolm_fr.evaluate_batch(
            batch_preds, batch_refs, idf_ref=idf_ref, idf_hyps=idf_hyps
        )["fisher_rao"]
        infolm_ab_scores = infolm_ab.evaluate_batch(
            batch_preds, batch_refs, idf_ref=idf_ref, idf_hyps=idf_hyps
        )["ab"]
        
    return infolm_fr_scores, infolm_ab_scores

def calculate_abstractiveness_scores(predictions, texts, references = None, aggregate: bool = True):
    stemmer = porter.PorterStemmer()
    tokenized_preds = [tokenize.tokenize(x, stemmer) for x in predictions]
    tokenized_texts = [tokenize.tokenize(x, stemmer) for x in texts]
    if references is not None:
        tokenized_refs = [tokenize.tokenize(x, stemmer) for x in references]
    else:
        tokenized_refs = tokenized_preds
    
    result = {}
    for use_modified in [False, True]:
        for n in range(1, 5):
            pred_ngram_overlaps = []
            label_ngram_overlaps = []
            for pred, label, text in zip(
                tokenized_preds, tokenized_refs, tokenized_texts
            ):
                pred_pair_ngram_overlap = calculate_ngram_overlap(
                    pred, text, n, use_modified
                )
                pred_ngram_overlaps.append(pred_pair_ngram_overlap)
                if references is not None:
                    label_pair_ngram_overlap = calculate_ngram_overlap(
                        label, text, n, use_modified
                    )
                    label_ngram_overlaps.append(label_pair_ngram_overlap)
            key = (
                f"ngram_overlap_{n}"
                if use_modified
                else f"novel_ngrams_{n}"
            )
            
            pred_ngram_overlaps = np.array(pred_ngram_overlaps)
            cond_abs = ~np.isnan(pred_ngram_overlaps)
            result[key + "_abs"] = pred_ngram_overlaps[cond_abs]
            
            if references is not None:
                label_ngram_overlaps = np.array(label_ngram_overlaps)
                cond_rel = cond_abs & ~np.isnan(label_ngram_overlaps)
                result[key + "_rel"] = pred_ngram_overlaps[cond_rel] / label_ngram_overlaps[cond_rel]
                
    if aggregate:
        for key, value in result.items():
            result[key] = np.mean(value)
    
    return result
            
            
def calculate_ngram_overlap(summary, text, n=1, use_modified=True):
    summary_ngrams = list(ngrams(summary, n))
    text_ngrams = list(ngrams(text, n))

    if len(summary_ngrams) > 0:
        ngrams_intersection = set(summary_ngrams).intersection(set(text_ngrams))
        if use_modified:
            word_is_part_of_ngram_copied = [
                any((x in ngram for ngram in ngrams_intersection)) for x in summary
            ]
            return 1 - sum(word_is_part_of_ngram_copied) / len(
                word_is_part_of_ngram_copied
            )
        else:
            return sum([x not in ngrams_intersection for x in summary_ngrams]) / len(
                summary_ngrams
            )
    return np.nan


class SentBert:
    def __init__(self, checkpoint: str = "sentence-transformers/all-mpnet-base-v2", device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint).to(device)
        self.device = device
        
    def __call__(
        self, source_texts: List[str], ref_texts: List[str], batch_size: int = 32
    ) -> np.ndarray:
        assert len(source_texts) == len(ref_texts)
        # Make batch_size an even number
        if batch_size % 2 == 0:
            batch_size -= 1
        half_batch_size = batch_size // 2
        n_texts = len(source_texts)
        scores = np.empty(n_texts, dtype=np.float32)
        start = 0
        end = 0
        
        while end < n_texts:
            end += half_batch_size
            batch_idx = slice(start, end)
            # Tokenize sentences
            encoded_input = self.tokenizer(
                source_texts[batch_idx] + ref_texts[batch_idx], padding=True, truncation=True, return_tensors='pt'
            )
            encoded_input = {key: value.to(self.device) for key, value in encoded_input.items()}
            # Calculate the probability of belonging to the positive class
            model_output = self.model(**encoded_input)
            # Perform pooling
            sent_embs = self.mean_pooling(model_output, encoded_input['attention_mask'])
            # Normalize embeddings
            sent_embs = F.normalize(sent_embs, p=2, dim=1)
            n_source_embs = len(sent_embs) // 2
            scores[batch_idx] = (sent_embs[:n_source_embs] * sent_embs[n_source_embs:]).sum(-1).cpu().detach().numpy()
            start = end
            
        return scores
                 
    @staticmethod      
    def mean_pooling(model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        """
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)