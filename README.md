# Lost in the Middle: How Language Models Use Long Contexts (Experimental Approach)

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

# Context and Problem

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

### Overview

This paper does **not** introduce a new model or algorithm. Instead, it provides a systematic **experimental framework** for evaluating position bias in existing language models.

**Core contribution**: A controlled evaluation methodology that reveals how model performance varies based on information position.

**Key innovation**: Systematic position variation (testing every 5th position) rather than random or coarse-grained testing (just beginning/middle/end).


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

# Key Results

### Result 1: The U-Shaped Performance Curve

<img width="454" height="428" alt="image" src="https://github.com/user-attachments/assets/756fa1f7-31fb-4c37-bd6e-5a8e7729dad2" />

This is the paper's core finding. For GPT-3.5-Turbo with 20 documents:

**Quantitative data** (from Table 6, Appendix G.2):

| **Position**     | **Accuracy** | **Change from Position 1** | **vs. No Documents (56.1%)** | **Comparison** |
|------------------|--------------|-----------------------------|-------------------------------|----------------|
| 🟢 **1st (beginning)** | **75.8 %** | Baseline | + 19.7 % better | 🟢 Best |
| 🟡 **5th** | 57.2 % | − 18.6 % | + 1.1 % better | 🟡 Slightly better |
| 🔴 **10th (middle)** | 53.8 % | − 22.0 % | − 2.3 % worse | 🔴 Worse |
| 🟠 **15th** | 55.4 % | − 20.4 % | − 0.7 % worse | 🟠 Slightly worse |
| 🟢 **20th (end)** | 63.2 % | − 12.6 % | + 7.1 % better | 🟢 Better |

**The shocking finding** 
"When relevant information is placed in the middle of its input context, GPT-3.5-Turbo's performance on the multi-document question task is lower than its performance when predicting without any documents (i.e., the closed-book setting; 56.1%)."

**This means:** Giving the model the answer in the middle makes it perform worse than giving it no documents at all.

### Result 2: All Models Show This Pattern

Every model texted exhibits the U-shaped curve:
| **Model**              | **Position 1** | **Position 10** | **Position 20** | **Degradation** | **Trend** |
|-------------------------|----------------|------------------|------------------|------------------|------------|
|  **GPT-3.5-Turbo**         | 75.8 % | 53.8 % | 63.2 % | −22.0 % | 🔴 High drop |
|  **GPT-3.5-Turbo-16K**     | 75.7 % | 54.1 % | 63.1 % | −21.6 % | 🔴 High drop |
|  **Claude-1.3**            | 59.9 % | 56.8 % | 60.1 % | −3.1 %  | 🟡 Stable |
|  **Claude-1.3-100K**       | 59.8 % | 57.0 % | 60.0 % | −2.8 %  | 🟢 Very stable |
|  **MPT-30B-Instruct**      | 53.7 % | 52.2 % | 56.3 % | −1.5 %  | 🟠 Minimal drop |
|  **LongChat-13B-16K**      | 68.6 % | 55.3 % | 55.0 % | −13.3 % | 🔴 Noticeable drop |
|  **GPT-4**                 | 89 %   | 75 %   | 84 %   | −14.0 % | 🟡 Moderate drop |

**Critical observation** (page 5):

"We find that models often have identical performance to their extended-context counterparts, indicating that extended-context models are not necessarily better at using their input context."

**Extended-context models don't fix the problem:**
- GPT-3.5 vs. GPT-3.5-16K: Within 0.3% at every position
- Claude-1.3 vs. Claude-100K: Within 0.2% at every position

**Implication:** Extending context length lets you fit more information, but doesn't help you use it better.

### Result 3: It's Architectural, Not Fixable by Prompting

<img width="444" height="384" alt="image" src="https://github.com/user-attachments/assets/1447f4aa-e847-4194-8834-1ec0d52664a7" />

**The experiment:** Compare MPT-30B (base model, no instruction tuning) vs. MPT-30B-Instruct

"Even base language models (i.e., without instruction fine-tuning) show a U-shaped performance curve as we vary the position of relevant information in the input context."

Both models show nearly identical U-curves.

**What this means:** The position bias emerges during pre-training, not from instruction tuning.

**Why this happens:**

1. Pre-training creates recency bias:
    - Language models trained to predict next token: P(token_t | previous tokens)
    - In natural text, recent context is most predictive
    - Models learn to attend strongly to recent tokens

2. **Instruction tuning adds primacy bias:**
    - Instructions typically placed at beginning: [System prompt] [Data] [Question]
    - Models learn to attend to task setup at start
    - Nothing emphasizes middle content

3. **Result:** Strong beginning + strong end, weak middle (U-curve)


### Result 4: Query Duplication Only Helps Simple Retrieval

**The Experiment:** Place query before AND after documents
```
Standard:                   Query-Aware:
[Documents]                 [Query]  ← New: query at start
[Query]                     [Documents]
                            [Query]  ← Still at end
```
| **Task**              | **Standard** | **Query-Aware** | **Improvement** | **Notes** |
|------------------------|---------------|------------------|------------------|------------|
|  **Key-Value Retrieval** | 45 % | 100 % | +55 % | ✅ Major gain |
|  **Multi-Doc QA**         | 54 % | 56 %  | +2 %  | ⚠️ Minimal improvement |

**Interpretation:** "Query-aware contextualization (placing the query before and after the documents or key-value pairs) enables near-perfect performance on the synthetic key-value task, but minimally changes trends in multi-document QA."

**Why:** Helps locate exact matches (key-value lookup), doesn't help reason about complex information.

### Result 5: More Documents Can Hurt

**The experiment:** 

**Results:**
| **# Documents** | **Retriever Recall** | **GPT-3.5 Accuracy** | **Claude Accuracy** | **Notes** |
|-----------------|----------------------|----------------------|---------------------|------------|
| 5  | 67 %  | 50 %  | 49 %  | 🟠 Moderate recall, lower accuracy |
| 10 | 75 %  | 53 %  | 52 %  | 🟡 Improved performance |
| 20 | 82 %  | 57 %  | 56 %  | 🟢 Noticeable accuracy gain |
| 30 | 86 %  | 58 %  | 57 %  | 🟢 Stable improvement |
| 50 | 90 %  | 58.5 % | 58 % | 🟢 Plateau reached |

**The critical gap:**
- Adding documents 20→50: Retriever improves +8%, Reader improves only +1.5%

"Using 50 documents instead of 20 retrieved documents only marginally improves performance (∼1.5% for GPT-3.5-Turbo and ∼1% for claude-1.3)."

**Why this matters:** More context can hurt because it creates more "middle" positions where the model struggles.



### Result 6: Encoder-Decoder Architecture Helps (Within Limits)

<img width="1114" height="336" alt="image" src="https://github.com/user-attachments/assets/52cc60b9-91e4-44e2-9ab3-78cf374fbb60" />

**The experiment:** Test Flan-UL2 (encoder-decoder) vs decoder-only models


**Flan-UL2 results:**

**Finding** (page 7):

"When Flan-UL2 is evaluated on sequences within its 2048-token training-time context window, its performance is relatively robust to changes in the position of relevant information within the input context (1.9% absolute difference between best- and worst-case performance)."

**But** (page 7):

"When evaluated on settings with sequences longer than 2048 tokens, Flan-UL2 performance begins to degrade when relevant information is placed in the middle."

## Question 1: Your Own RAG System

**Scenario**: You're building a chatbot that answers questions about your company's internal documentation. You plan to retrieve relevant documents and feed them to GPT-4.

**Question**: Now that you know about the U-shaped performance curve, how would you design your system differently? What's one specific change you'd make?

<details>
<summary><b>Click to reveal answer</b></summary>

**Answer**: Use chunking - split retrieval into multiple focused API calls with ≤10 documents each, ensuring critical information stays at the edges (beginning or end) of each context window.

**Alternative**: Strategically rerank documents to place the most relevant ones at positions 1-3 (beginning) and the last 3 positions (end), accepting that middle documents will be weakly attended to.

</details>



## Why Encoder–Decoder Helps

**Bidirectional Encoder**
- All tokens attend to all other tokens  
- Can process information in the context of the entire sequence  
- Position affects *relative weighting*, not absolute visibility  

**Critical Limitation**
- This advantage only holds **within the training sequence length**


### Model Architecture and Pseudocode
### Conceptual Architecture: How Position Bias Emerges

**Decoder-Only Models (GPT, Claude):**

```
Architecture:
  For each token position t:
    - Can only attend to positions 1 through t (causal masking)
    - Attention weights learned during training
    
  Attention pattern learned from pre-training:
    - Recent tokens (t-1, t-2, ...): HIGH attention
      → Strong signal for next-token prediction
    - Distant tokens (t-100, t-200, ...): LOW attention
      → Weak signal for next-token prediction
    
  Modified by instruction tuning:
    - First few tokens: MEDIUM-HIGH attention
      → Contains task instructions
    - Last few tokens: HIGH attention
      → Contains user query (recency)
    - Middle tokens: LOW attention
      → No special emphasis
      
  Result: U-shaped attention distribution
```


**Encoder-Decoder Models (T5, Flan-UL2):**

```
Architecture:
  Encoder:
    - All tokens attend to all tokens (bidirectional)
    - Position represented via relative position embeddings
    - Can process token i in context of tokens i-k and i+k
    
  Decoder:
    - Attends to all encoder states
    - Can access any encoded information uniformly
    
  Result: More uniform attention distribution
  
  Limitation:
    - Positional embeddings learned up to training length
    - Beyond training length: extrapolation fails
    - U-curve emerges when extrapolating
```

**Pseudocode: Position-Controlled Evaluation**
```
ALGORITHM: EvaluatePositionBias
INPUT: 
  - dataset: List of (question, answer_doc, distractor_docs)
  - model: LanguageModel
  - num_documents: int (e.g., 20)
  - positions_to_test: List[int] (e.g., [1, 5, 10, 15, 20])

OUTPUT:
  - results: Dict[position -> accuracy]

PROCEDURE:
  results = {}
  
  FOR each position p IN positions_to_test:
    correct_count = 0
    total_count = 0
    
    FOR each (question, answer_doc, distractors) IN dataset:
      # Step 1: Construct document list with answer at position p
      documents = []
      distractor_index = 0
      
      FOR i FROM 1 TO num_documents:
        IF i == position:
          documents.append(answer_doc)
        ELSE:
          documents.append(distractors[distractor_index])
          distractor_index += 1
      
      # Step 2: Format prompt
      prompt = "Write a high-quality answer using only the provided search results.\n\n"
      FOR i, doc IN enumerate(documents):
        prompt += f"Document [{i+1}]: {doc}\n\n"
      prompt += f"Question: {question}\nAnswer:"
      
      # Step 3: Generate response
      response = model.generate(prompt)
      
      # Step 4: Check if correct
      ground_truth_answers = get_answers(question)
      is_correct = any(answer.lower() in response.lower() 
                      for answer in ground_truth_answers)
      
      IF is_correct:
        correct_count += 1
      total_count += 1
    
    # Step 5: Compute accuracy for this position
    results[position] = correct_count / total_count
  
  RETURN results
```


## Key Algorithm: Strategic Document Positioning (Novel Contribution)

**Previous work** (e.g., Ivgi et al., 2023): Only tested beginning vs. random positions

**This paper's innovation:** Fine-grained systematic testing of every position

```
ALGORITHM: SystematicPositionTest
INPUT:
  - num_documents: int (e.g., 20)
  - granularity: int (e.g., 5 → test positions 1, 5, 10, 15, 20)

OUTPUT:
  - positions_to_test: List[int]

PROCEDURE:
  positions = []
  current_position = 1
  
  WHILE current_position <= num_documents:
    positions.append(current_position)
    current_position += granularity
  
  # Always include last position
  IF positions[-1] != num_documents:
    positions.append(num_documents)
  
  RETURN positions

EXAMPLE:
  SystematicPositionTest(20, 5) → [1, 5, 10, 15, 20]
  
  Creates 5 test conditions:
    Condition 1: Answer at position 1  (beginning)
    Condition 2: Answer at position 5  (early)
    Condition 3: Answer at position 10 (middle)
    Condition 4: Answer at position 15 (late)
    Condition 5: Answer at position 20 (end)
```
**Why this matters:** Reveals U-shaped curve instead of just "beginning vs. middle vs. end"

**Comparison to Previous Approach**
| **Aspect**              | **Previous Work**                          | **This Paper**                                           |
|--------------------------|--------------------------------------------|----------------------------------------------------------|
| **Position Granularity** | Coarse (beginning / middle / end)          | Fine-grained (every 5th position)                        |
| **Control**              | Random positioning                         | Systematic controlled positioning                        |
| **Task Realism**         | Synthetic or next-word prediction          | Realistic multi-document QA (mimics RAG)                 |
| **Model Coverage**       | Mostly encoder–decoder                     | Decoder-only **+** encoder–decoder                       |
| **Extended Context**     | Not tested                                 | Direct comparison (e.g., 4K vs 16K vs 100K)              |

