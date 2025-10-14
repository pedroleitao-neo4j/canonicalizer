from t5_judge.judge import T5Judge

fact = "John is a person"
evidence_ent = "John thought that her daughter was at school."
evidence_con = "Mary sarcastically named her cat John."
evidence_unk = "John clearly was ill."

print("Testing T5Judge with different evidence for the fact:", fact)
print(f"Evidence (entailed): {evidence_ent}")
print(f"Evidence (contradicted): {evidence_con}")
print(f"Evidence (unknown): {evidence_unk}")

judge = T5Judge(model_name="google/flan-t5-xl")
result_ent = judge.judge_claim(evidence_ent, fact)
print(result_ent)
result_con = judge.judge_claim(evidence_con, fact)
print(result_con)
result_unk = judge.judge_claim(evidence_unk, fact)
print(result_unk)

fact = "Paris is the capital of France"
evidence_ent = "Paris can be such a beautiful city in the spring."
evidence_con = "Paris is the granddaughter of a famous billionaire."
evidence_unk = "Did Paris win the match?"

print("\nTesting T5Judge with different evidence for the fact:", fact)
print(f"Evidence (entailed): {evidence_ent}")
print(f"Evidence (contradicted): {evidence_con}")
print(f"Evidence (unknown): {evidence_unk}")

result_ent = judge.judge_claim(evidence_ent, fact)
print(result_ent)
result_con = judge.judge_claim(evidence_con, fact)
print(result_con)
result_unk = judge.judge_claim(evidence_unk, fact)
print(result_unk)