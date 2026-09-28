"""Base de connaissances de l'assistant.

Chaque entrée est un court document. Pour enrichir l'assistant, il suffit
d'ajouter des textes à cette liste (notes de cours, extraits de PDF...).
"""

DOCUMENTS = [
    "AI ethics involves principles for responsible AI development, such as fairness, "
    "transparency, accountability, privacy and respect for human rights.",
    "Large Language Models (LLMs) like GPT are Transformer-based neural networks trained on "
    "huge text corpora. They can generate human-like text but may reproduce biases present "
    "in their training data.",
    "Nvidia GPUs accelerate deep learning training by running thousands of computations in "
    "parallel, which makes training large neural networks much faster than on CPUs.",
    "Retrieval-Augmented Generation (RAG) combines search with generation: relevant documents "
    "are first retrieved from a knowledge base, then given to the LLM as context, which "
    "reduces hallucinations and lets the model answer with up-to-date or private information.",
    "Guardrails in AI are safety mechanisms that detect unsafe inputs or outputs, for example "
    "blocking requests about weapons or self-harm before they reach the model.",
    "An embedding is a vector of numbers that represents the meaning of a text. Texts with "
    "similar meanings have embeddings that are close to each other in the vector space.",
    "A vector database stores embeddings and performs semantic search: it returns the "
    "documents whose embeddings are most similar to the embedding of the query, usually "
    "measured with cosine similarity.",
    "An AI agent is an LLM that can decide to call external tools, such as a calculator, a "
    "search engine or an API, and use their results to answer a question. The ReAct pattern "
    "alternates reasoning steps and tool calls.",
    "Hallucination is when a language model produces an answer that sounds plausible but is "
    "factually wrong or not supported by any source.",
    "Training large AI models consumes a lot of energy, so the environmental impact of AI is "
    "an important ethical concern alongside bias and privacy.",
]
