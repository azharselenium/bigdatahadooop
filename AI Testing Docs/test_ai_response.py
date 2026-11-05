from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval, AnswerRelevancyMetric
from deepeval.models import OllamaModel
import ollama

# --------------------------------
# STEP 1: Generate answer using Llama 3
# --------------------------------

question = "What is Selenium?"

response = ollama.chat(
    model="llama3:latest",
    messages=[
        {"role": "user", "content": question}
    ]
)

actual_answer = response["message"]["content"]

print("\n========== LLAMA 3 RESPONSE ==========")
print("Question:", question)
print("AI Answer:", actual_answer)

# --------------------------------
# STEP 2: Define expected answer
# --------------------------------

expected_answer = (
    "Selenium is an open-source framework "
    "used to automate web browsers."
)

print("\n========== EXPECTED ANSWER ==========")
print(expected_answer)

# --------------------------------
# STEP 3: Create test case
# --------------------------------

test_case = LLMTestCase(
    input=question,
    actual_output=actual_answer,
    expected_output=expected_answer
)

# --------------------------------
# STEP 4: Configure evaluator
# --------------------------------

judge_model = OllamaModel(
    model="llama3:latest",
    base_url="http://localhost:11434",
    temperature=0
)

# --------------------------------
# STEP 5: Correctness evaluation
# --------------------------------

correctness = GEval(
    name="Correctness",
    criteria=(
        "Evaluate whether the AI answer is factually correct, "
        "complete and consistent with the expected answer. "
        "Accept different wording when the meaning is correct."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    model=judge_model,
    threshold=0.7,
    verbose_mode=True
)

# --------------------------------
# STEP 6: Answer relevancy evaluation
# --------------------------------

relevancy = AnswerRelevancyMetric(
    model=judge_model,
    threshold=0.7,
    verbose_mode=True
)

# --------------------------------
# STEP 7: Execute evaluation
# --------------------------------

print("\n========== DEEPEVAL EVALUATION ==========")

correctness.measure(test_case)
relevancy.measure(test_case)

# --------------------------------
# STEP 8: Display evaluation results
# --------------------------------

print("\n========== FINAL EVALUATION REPORT ==========")

print("\nCORRECTNESS")
print("Score:", correctness.score)
print("Reason:", correctness.reason)
print("Status:", "PASS" if correctness.is_successful() else "FAIL")

print("\nANSWER RELEVANCY")
print("Score:", relevancy.score)
print("Reason:", relevancy.reason)
print("Status:", "PASS" if relevancy.is_successful() else "FAIL")

print("\n==========================================")