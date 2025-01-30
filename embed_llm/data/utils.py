templates_for_qa = [
    "Question: {question}?\nAnswer:",
    "{question}?\n",
    "Answer the following question:\n\n{question}\n",
    "Answer this question:\n\n{question}?\n",
    "Please answer this question: {question}\n",
    "Answer the question...{question}?\n",
    "What is the answer to this question? {question}\n\n",
    "Can you tell me the answer to {question}?",
    "Next question: {question}\n\n",
    "Q: {question} A:",
    "{question}\nWhat is the answer?",
    "Write the answer: {question}\n",
    "{question}???\n",
]

# Used to preprocess datasets
templates_for_sum = [
    "Write a short summary for the previous text\n\nSummary:",
    "Briefly summarize this article:\nSummary:",
    "What is a shorter version of the former text:\n\nSummary:",
    "Write a brief summary in a sentence or less.",
    "What is a very short summary of the above text?",
    "Summarize the aforementioned text in a single phrase.",
    "Can you generate a short summary of the above paragraph?",
    "Summarize the above articles\n\ntl;dr:",
]

template_for_fact_checking = [
    'Verify the following claims with "True" or "False":\n{question}',
]
