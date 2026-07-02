#!/usr/bin/env python3
"""
Score model outputs with an LLM judge (GPT-5).

Reads per-sample model outputs produced by collect_llm_judge_inputs.py, sends each
(question, target, model_answer) to GPT-5 for scoring on a 0.0-1.0 scale, and
writes the results to an output file.

Usage:
    python run_llm_judge.py \
        --input_file judge_inputs/eval_sciq_llama.json \
        --output_file judge_scores/eval_sciq_llama.json

Requires OPENAI_API_KEY to be set in the environment.

Dataset type is auto-detected from the input_text content, so no --dataset flag
is needed.
"""

import argparse
import json
import os
from openai import OpenAI


def detect_dataset_type(input_text, target):
    if input_text.startswith("Here is the dialogue") or input_text.startswith("Here's the dialogue"):
        return "dialogue_summary"
    if input_text.startswith("Here is the text") or input_text.startswith("Here's the text"):
        return "text_summary"
    if input_text.startswith("The following are stories"):
        return "coqa"
    if input_text.startswith("The following are multiple choice"):
        return "mmlu"
    if input_text.startswith("The following are abstract"):
        return "pubmed"
    if input_text.startswith("The following are context"):
        return "sciq"
    if input_text.startswith("Question:"):
        return "triviaqa" if isinstance(target, list) else "medquad"
    if input_text.startswith("Q:"):
        return "truthfulqa"
    raise ValueError(f"Cannot detect dataset type for input: {input_text[:120]!r}")


def preprocess(input_text, target, dataset_type):
    """
    Extract the relevant question/context snippet from the full few-shot prompt,
    and determine the label caveat text. Returns (question, label, caveat, prompt_type).
    prompt_type is 'qa' or 'summary'.
    """
    if dataset_type in ("dialogue_summary", "text_summary"):
        summary_at_end = input_text.rfind("Summary")
        q = input_text[:summary_at_end].strip()
        text_mention = q.rfind("Text:")
        q = q[text_mention:]
        label = target if isinstance(target, str) else " / ".join(target)
        return q, label, None, "summary"

    label = target if isinstance(target, str) else " / ".join(str(t) for t in target)

    if dataset_type == "coqa":
        find_story = input_text.find("Story:")
        find_story_end = input_text.find("Question:")
        story_text = input_text[find_story:find_story_end].strip()
        answer_at_end = input_text.rfind("Answer:")
        q = input_text[:answer_at_end]
        last_q = q.rfind("Question:")
        q = story_text + "\n" + q[last_q:].strip()
        return q, label, ":", "qa"

    if dataset_type == "mmlu":
        answer_at_end = input_text.rfind("Answer:")
        q = input_text[:answer_at_end]
        last_q = q.rfind("Q:")
        q = q[last_q:].strip().replace("Q:", "Question:", 1)
        return q, label, ":", "qa"

    if dataset_type == "pubmed":
        answer_at_end = input_text.rfind("Answer:")
        q = input_text[:answer_at_end].strip()
        context_mention = q.rfind("Abstract:")
        q = q[context_mention:].replace("Abstract: \n", "")
        return q, label, ":", "qa"

    if dataset_type == "sciq":
        answer_at_end = input_text.rfind("Answer:")
        q = input_text[:answer_at_end].strip()
        context_mention = q.rfind("Context")
        q = q[context_mention:].replace("Context: \n", "")
        return q, label, ":", "qa"

    if dataset_type in ("triviaqa", "medquad"):
        answer_at_end = input_text.rfind("Answer:")
        q = input_text[:answer_at_end]
        last_q = q.rfind("Question:")
        q = q[last_q:].strip()
        caveat = " (any of the following are correct):" if dataset_type == "triviaqa" else ":"
        return q, label, caveat, "qa"

    if dataset_type == "truthfulqa":
        answer_at_end = input_text.rfind("\nA:")
        q = input_text[:answer_at_end]
        last_q = q.rfind("\nQ:")
        q = q[last_q:].strip().replace("Q:", "Question:", 1)
        return q, label, " (any of the following are correct):", "qa"

    raise ValueError(f"No preprocessing defined for dataset type: {dataset_type!r}")


def build_qa_prompt(question, label, model_answer, caveat):
    return f"""
Evaluate the following answers to questions. For each question you will be given a model answer and the correct answer.
You must rate how correct the model answer is on a scale from 0.0 to 1.0, where:
- 1.0 means the model answer is completely correct
- 0.0 means the model answer is completely incorrect or wrong
- Values in between reflect answers that are partially correct but incomplete or imprecise

Only respond with a number between 0.0 to 1.0. Do not write any explanation.

For example:

Question: who is the young guitarist who played with buddy guy?
Ground Truth: Quinn Sullivan
Model Answer: Ronnie Earl
Score: 0.0

Question: name of the first episode of stranger things
Ground Truth: Chapter One: The Vanishing of Will Byers
Model Answer: The disappearance of Will Byers
Score: 0.0

Question: What are the symptoms of diabetes?
Ground Truth: Common symptoms include increased thirst, frequent urination, fatigue, and blurred vision.
Model Answer: Symptoms of diabetes include increased thirst and frequent urination.
Score: 0.6

Question: What is the capital of Australia?
Ground Truth: Canberra
Model Answer: Canberra, which is located in the Australian Capital Territory
Score: 1.0

{question}
Ground Truth{caveat} {label}
Model Answer: {model_answer}
Score:
"""


def build_summary_prompt(question, label, model_answer):
    return f"""
Only respond with a number between 0.0 to 1.0. Do not write any explanation.

The task below is a text summarisation task. You will see a Text, a Ground Truth Summary, and a Model Summary. Score how well the Model Summary matches the Ground Truth Summary from 0.0 to 1.0, where 1.0 means it conveys the same information, 0.0 means it is completely different or irrelevant, and scores in between reflect partial overlap in the key points covered.

{question}
Ground Truth Summary: {label}
Model Summary: {model_answer}
Score:
"""


def query_judge(prompt, client, max_retries=10):
    for attempt in range(max_retries):
        response = client.chat.completions.create(
            temperature=1,
            top_p=1,
            model="gpt-5-2025-08-07",
            messages=[{"role": "user", "content": prompt}],
            logprobs=False,
        )
        text = response.choices[0].message.content.strip()
        try:
            score = float(text)
            if 0.0 <= score <= 1.0:
                return score
        except ValueError:
            pass
        print(f"  Bad response (attempt {attempt + 1}/{max_retries}): {text!r}")
    raise RuntimeError(f"Judge failed to return a valid score after {max_retries} attempts")


def main():
    parser = argparse.ArgumentParser(description="Score model outputs with an LLM judge")
    parser.add_argument("--input_file", required=True, help="JSON array from collect_llm_judge_inputs.py")
    parser.add_argument("--output_file", required=True, help="Output JSON array with judge_score added")
    args = parser.parse_args()

    with open(args.input_file) as f:
        samples = json.load(f)

    client = OpenAI()
    results = []

    for idx, sample in enumerate(samples):
        print(f"\n[{idx + 1}/{len(samples)}]")
        input_text = sample["input_text"]
        target = sample["target"]
        model_answer = sample["model_answer"]

        dataset_type = detect_dataset_type(input_text, target)
        question, label, caveat, prompt_type = preprocess(input_text, target, dataset_type)

        if prompt_type == "summary":
            prompt = build_summary_prompt(question, label, model_answer)
        else:
            prompt = build_qa_prompt(question, label, model_answer, caveat)

        score = query_judge(prompt, client)
        print(f"  score: {score}")

        results.append({
            "input_text": input_text,
            "target": target,
            "model_answer": model_answer,
            "judge_score": score,
        })

    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} scored samples to {args.output_file}")


if __name__ == "__main__":
    main()
