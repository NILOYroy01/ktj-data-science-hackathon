import csv
from app import answer_query, retrieve_evidence

# -------------------------
# LOAD TEST QUESTIONS
# -------------------------
test_questions = [
    "Who is the protagonist of the story?",
    "What is the main conflict?",
    "Where does the story take place?",
    "What year is mentioned in the novel?"
]

# -------------------------
# WRITE RESULTS.CSV
# -------------------------
with open("results.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    
    # Header
    writer.writerow(["id", "question", "answer", "evidence", "status"])

    # Rows
    for idx, question in enumerate(test_questions, start=1):
        answer = answer_query(question)
        evidence = retrieve_evidence(question)

        status = "SUCCESS"
        if "Not found" in answer:
            status = "NOT_FOUND"

        writer.writerow([
            idx,
            question,
            answer,
            " | ".join(evidence),
            status
        ])

print("✅ results.csv generated successfully")

