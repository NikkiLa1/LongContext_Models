# Lost in the Middle: How Language Models Use Long Contexts

**Paper:** Lost in the Middle: How Language Models Use Long Contexts (arXiv:2307.03172)

**Authors:** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang

**Institutions:** Stanford University, UC Berkeley, Samaya AI

**ArXiv:** https://arxiv.org/abs/2307.03172 


**Presented by:** Nikki La, Nov 18

# Overview

This paper investigates a fundamental question about modern language models: Can they actually use their long context windows effectively?

While models like GPT-3.5-Turbo (4K), GPT-4 (8K-32K), and Claude (100K) can accept long inputs, this paper reveals they struggle to use information in the middle of those contexts.

**Core Finding:** Language models exhibit a U-shaped performance curve - they excel at using information at the beginning or end of context, but performance degrades 20%+ when accessing information in the middle.

**Why This Matters:** Most real-world applications (RAG, document QA, multi-turn conversations) assume models can access information regardless of position in context. This paper shows that assumption is false, with major implications for system design.

# Background

### The Context Window Arms Race

Recent improvements in hardware and algorithms have resulted in language models with increasingly large context windows (4K, 32K, even 100K tokens).

Models evaluated in this paper:
| **Model**              | **Context Length**      | **Source / Notes**                                       |
|--------------------------|-------------------------|-----------------------------------------------------------|
| MPT-30B-Instruct         | 8,192 tokens            | Section 2.2; uses ALiBi positional embeddings             |
| LongChat-13B (16K)       | 16,384 tokens           | Section 2.2; extends LLaMA-13B with condensed rotary embeddings |
| GPT-3.5-Turbo            | 4,096 tokens            | Section 2.2; OpenAI API                                   |
| GPT-3.5-Turbo-16K        | 16,384 tokens           | Section 2.2; OpenAI API                                   |
| Claude-1.3               | 8,192 tokens            | Section 2.2; Anthropic API                                |
| Claude-1.3-100K          | 100,000 tokens          | Section 2.2; Anthropic API                                |
| Flan-T5-XXL              | 512 tokens (training)   | Section 4.1; Raffel et al., 2020                          |
| Flan-UL2                 | 2,048 tokens (training) | Section 4.1; Tay et al., 2023                             |

### Real-World Applications Affected

"These input contexts can contain thousands of tokens, especially when language models are used to process long documents (e.g., legal or scientific documents, conversation histories, etc.) or when language models are augmented with external information (e.g., relevant documents from a search engine, database query results, etc.)"

**Applications that assume uniform context acces:**

- Retrieval-Augmented Generation (RAG)
- Document question answering
- Conversational AI with long histories
- Code analysis with large codebases
- Legal/scientific document analysis

### Research Question

"If language models can robustly use information within long input contexts, then their performance should be minimally affected by the position of the relevant information in the input context."

**This paper tests:** Does position of information in context affect model performance?

**The hypothesis:** If models truly "understand" their context, performance should be roughly the same regardless of where information appears.

# Approach

### Experimental Design Overview

The paper uses **controlled experiments** to systematically test position effects:

**Key innovation:** 
Instead of random positioning, test every position (1st, 5th, 10th, 15th, 20th...) to create fine-grained performance curves.

### Task 1: Multi-Document Question Answering

**Purpose:** Test realistic retrieval scenario (mimics RAG applications)

**Data Source:** NaturalQuestions-Open (2,655 real Google search queries)

**Setup:**
Given:
- k documents (Wikipedia passages, each ≤100 tokens)
- Exactly 1 document contains the answer
- k-1 "distractor" documents (retrieved by Contriever - relevant but wrong)
- A question

Manipulation:
- Vary k ∈ {10, 20, 30} documents
- Systematically place answer document at each position

Evaluation:
- Does model's output contain the correct answer?

<img width="743" height="302" alt="image" src="https://github.com/user-attachments/assets/c5c69ad0-7fb6-4675-99dd-da25ce906671" />

**Why distractors matter:** Using Contriever-retrieved distractors makes the task realistic - these are documents that seem relevant but don't contain the answer, just like real retrieval systems.


### Task 2: Key-Value Retrieval (Synthetic)

**Purpose:** Test minimal retrieval capability (no reasoning required)

**Setup:**
Given:
- JSON object with k key-value pairs
- All keys and values are random 128-bit UUIDs
- A target key to look up

Manipulation:
- Vary k ∈ {75, 140, 300} pairs
- Systematically vary position of target pair

Evaluation:
- Does model return exact matching value?

<img width="739" height="299" alt="image" src="https://github.com/user-attachments/assets/e2030982-3dd5-4f16-b4f1-2bbc25c51aa7" />


**Why use UUIDs:** Removes all semantic/linguistic cues - tests pure positional retrieval ability.

**Controlled Variables**
- Context length: Number of documents/pairs
- Position: Where the relevant information appears
- Content: Everything else held constant

This allows isolating the effect of position on performance.






