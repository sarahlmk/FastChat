import json

answers = []
with open("mt_bench/model_answer/functionary.jsonl", "r") as answer_file:
    for line in answer_file:
        # Parse the JSON data in each line
        data = json.loads(line)
        answers.append(data)
        
qns = []
with open("mt_bench/question.jsonl", "r") as qn_file:
    for line in qn_file:
        # Parse the JSON data in each line
        data = json.loads(line)
        qns.append(data)
        
judge = []
with open("mt_bench/model_judgment/gpt-4_single.jsonl", "r") as f:
    for line in f:
        # Parse the JSON data in each line
        data = json.loads(line)
        judge.append(data)
judges = sorted(judge, key=lambda x: x['question_id'])

combine = []
i = 0
for qn, answer in zip(sorted(qns, key=lambda x: x['question_id']), sorted(answers, key=lambda x: x['question_id'])):
    for judge in judges:
        if judge["question_id"] == qn["question_id"]:
            combine
            
# Create a dictionary to map question IDs to questions
question_dict = {q["question_id"]: {"turns": q["turns"],"category": q["category"]} for q in qns}

# Create a dictionary to map question IDs and turns to answers
answer_dict = {a["question_id"]: a["choices"][0]["turns"] for a in answers}

# Create a dictionary to map question IDs, turns, and judges to scores
judge_dict = {(j["question_id"], j["turn"]): {"judgment": j["judgment"], "score": j["score"]} for j in judges}


combine = []
for j in judges:
    question_id = j["question_id"]
    turn = j["turn"]
    combine.append({"question_id": question_id,
                   "turn": turn,
                   "question": question_dict[question_id]["turns"][turn-1],
                   "category": question_dict[question_id]["category"],
                   "answer": answer_dict[question_id][turn-1],
                   "judgment": j["judgment"],
                   "score": j["score"]})


category_scores = {}
all_score = 0
# Populate the category_scores dictionary
for item in combine:
    all_score += item["score"]
    category = item["category"]
    score = item["score"]
    if category not in category_scores:
        category_scores[category] = [score]
    else:
        category_scores[category].append(score)

# Calculate the average score for each category
average_scores_by_category = {}
for category, scores in category_scores.items():
    average_score = sum(scores) / len(scores)
    average_scores_by_category[category] = average_score

# Print the average scores for each category
print(f"avg all categories: {all_score/len(combine)}")
print()
for category, average_score in average_scores_by_category.items():
    print(f"Category: {category}, Average Score: {average_score}")
import pandas as pd
df = pd.DataFrame(combine).to_csv("mt_bench_score_1130.csv")