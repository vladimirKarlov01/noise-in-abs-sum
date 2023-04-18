from typing import List, Dict, Tuple, Union
from evaluate import load
import numpy as np
from tqdm import tqdm

from metrics.metrics import (
    pair_bleu, calculate_bart_score, calculate_cola_model_predictions,
    calculate_summac_score, decode, calculate_abstractiveness_scores, SentBert
)


SACREBLEU = load("sacrebleu")
ROUGE = load("rouge")
BERTSCORE = load("bertscore")
SENTBERT = SentBert()


def compute_metrics(
    eval_preds, tokenizer, add_metrics_to_use: Union[Tuple[str], List[str]] = ("bartscore", "bertscore", "sentbert", "summac")
) -> Dict[str, float]:
    
    generated_texts, reference_texts, *original_texts = decode(eval_preds, tokenizer)
    if len(original_texts) > 0:
        original_texts = original_texts[0]
        
    result = {}
    ### Metrics than only employ the generated texts
    result["word_length_gen"] = np.array([
        len(text.split()) for text in generated_texts
    ])
    result["char_length_gen"] = np.array([
        len(text) for text in generated_texts
    ])
    result["token_length_gen"] = np.array([
        np.count_nonzero(pred != tokenizer.pad_token_id) - 3
        for pred in eval_preds.predictions
    ])
    result["cola_score"] = calculate_cola_model_predictions(generated_texts, aggregate=False)

    ### Metrics that use both the generated texts and the original texts and
    ### those than are not ''obliged'' to use reference texts
    # Lengths
    src_word_lengths = [
        len(text.split()) for text in original_texts
    ]
    src_char_lengths = [
        len(text) for text in original_texts
    ]
    result["word_length_src_rel"] = result["word_length_gen"] / src_word_lengths
    result["char_length_src_rel"] = result["char_length_gen"] / src_char_lengths
    # Relative cola score
    src_cola_score = calculate_cola_model_predictions(original_texts, aggregate=False)
    result["cola_score_src_rel"] = result["cola_score"] / src_cola_score
    
    if "bertscore" in add_metrics_to_use:
        bertscore_art = BERTSCORE.compute(
            predictions=generated_texts, references=original_texts, model_type="roberta-large"
        )
        for key in bertscore_art:
            if key != "hashcode":
                result[f"bertscore_art_{key}"] = bertscore_art[key]
    result.update(
        calculate_abstractiveness_scores(generated_texts, original_texts, reference_texts)
    )
    if "summac" in add_metrics_to_use:
        result.update(
            calculate_summac_score(generated_texts, original_texts, reference_texts)
        )
    if "bartscore" in add_metrics_to_use:
        result.update(calculate_bart_score(
            preds=generated_texts,
            texts=original_texts,
            refs=reference_texts,
            batch_size=4,
        ))
    if "sentbert" in add_metrics_to_use:
        result["sentbert_src"] = SENTBERT(original_texts, generated_texts)
    ### Metrics that use both the generated texts and the reference texts
    if reference_texts is not None:
        # BLEU
        result["bleu"] = np.array([
            pair_bleu(pred, ref)
            for pred, ref in tqdm(zip(generated_texts, reference_texts))
        ])
        # ROUGE
        result.update(ROUGE.compute(
            predictions=generated_texts, references=reference_texts, use_stemmer=True
        ))
        # Sacrebleu
        sacrebleu_result = SACREBLEU.compute(
            predictions=generated_texts, references=[[ref] for ref in reference_texts]
        )
        result["sacrebleu"] = sacrebleu_result.pop("score")
        # Lengths
        ref_word_lengths = [
            len(text.split()) for text in reference_texts
        ]
        ref_char_lengths = [
            len(text) for text in reference_texts
        ]
        ref_token_lengths = [
            np.count_nonzero(lab != tokenizer.pad_token_id) - 2
            for lab in eval_preds.label_ids
        ]
        result["word_length_rel"] = result["word_length_gen"] / ref_word_lengths
        result["char_length_rel"] = result["char_length_gen"] / ref_char_lengths
        result["token_length_rel"] = result["token_length_gen"] / ref_token_lengths
        # Relative cola score
        ref_cola_score = calculate_cola_model_predictions(reference_texts, aggregate=False)
        result["cola_score_rel"] = result["cola_score"] / ref_cola_score
        # BERTScore
        if "bertscore" in add_metrics_to_use:
            bertscore = BERTSCORE.compute(
                predictions=generated_texts, references=reference_texts, model_type="roberta-large"
            )
            for key in bertscore:
                if key != "hashcode":
                    result[f"bertscore_{key}"] = bertscore[key]
                
        if "sentbert" in add_metrics_to_use:
            result["sentbert_ref"] = SENTBERT(reference_texts, generated_texts)
            sentbert_ref_src = SENTBERT(original_texts, reference_texts)
            result["sentbert_rel"] = result["sentbert_src"] / sentbert_ref_src
            
    for key, value in result.items():
        result[key] = float(np.mean(value))
    result = {key: result[key] for key in sorted(result.keys())}

    return result