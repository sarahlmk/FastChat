"""Generate answers with vllm server.

Usage:
python gen_model_answer_server.py --model-id functionary
"""
import argparse
import json
import os
import random
import time

import shortuuid
import torch
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype
import openai
openai.api_base = "http://localhost:8000/v1"
openai.api_key = "functionary" # We just need to set this something other than None, so it works with openai package. No API key is required.


def convert_to_chat_message(list_conv):
    list_chat_message = []
    roles = ["user", "assistant"]
    
    for idx, message in enumerate(list_conv):
        role = roles[idx % 2]
        list_chat_message.append({"role": role, "content": message})

    return list_chat_message

def generate(history, temperature, max_tokens):
    messages = convert_to_chat_message(history)
    response = openai.ChatCompletion.create(
        model="../functionary-13b",
        messages=messages,
        max_tokens=max_tokens,
        temperature = temperature,
        functions=[]
    )
    response_message = response["choices"][0]["message"]
    return response_message

def run_eval(
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    get_answers_func = get_model_answers

    chunk_size = 1
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices
            )
        )

    # if use_ray:
    #     ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices
):
    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template(model_id)
            turns = []
            history = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                history.append(qs)
                output = generate(history, temperature, max_new_token)
                turns.append(output["content"])
                history.append(output["content"])

            choices.append({"index": i, "turns": turns, "history": history})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )

    args = parser.parse_args()

    
    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices
    )

    reorg_answer_file(answer_file)
