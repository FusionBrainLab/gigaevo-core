def entrypoint():
    return {
        "system_prompt": "Expert in multi-hop QA: produce minimal required output at each step.",
        "steps": [
            # Step 1: First-hop retrieval (frozen tool step)
            {
                "number": 1,
                "title": "Retrieve first-hop passages.",
                "step_type": "tool",
                "step_config": {
                    "tool_name": "retrieve",
                    "input_mapping": {"query": "$outer_context"},
                },
                "dependencies": [],
                "frozen": True,
            },
            # Step 2: Summarize first-hop passages
            {
                "number": 2,
                "title": "Summarize key facts from passages.",
                "step_type": "llm",
                "aim": "Extract key facts from the provided passages relevant to the question.",
                "stage_action": (
                    "First, extract all facts that directly answer the question or describe attributes of the main entities (e.g., nationality, birth year, occupation). "
                    "Second, extract facts that establish relationships between entities required for multi-hop reasoning (e.g., 'X wrote Y' when the question is about Y's author). "
                    "Exclude facts about unrelated entities. "
                    "List each fact as a short phrase without conclusions."
                ),
                "reasoning_questions": (
                    "Which facts directly involve the main entities in the question? "
                    "Which facts describe entity attributes or establish critical relationships? "
                    "Exclude facts about unrelated entities."
                ),
                "example_reasoning": (
                    "Question: What is the birth year of the author of 'Pride and Prejudice'?\n"
                    "Passages: [0] Jane Austen | Jane Austen was an English novelist. She wrote 'Pride and Prejudice' in 1813. [1] Pride and Prejudice | This novel was published in 1813.\n"
                    "Reasoning: The question asks for the birth year of the author. The author is Jane Austen (from [0]). The passages state she wrote the book in 1813 and her nationality. "
                    "Step 1: Direct/entity attribute facts:\n"
                    "- Jane Austen wrote 'Pride and Prejudice' in 1813.\n"
                    "- Jane Austen was an English novelist.\n"
                    "Step 2: Relationship facts (none needed here as author is identified).\n"
                    "These facts will generate a query for the missing birth year: 'Jane Austen birth year'."
                ),
                "dependencies": [1],
                "frozen": False,
            },
            # Step 3: Generate second-hop query
            {
                "number": 3,
                "title": "Generate second-hop query.",
                "step_type": "llm",
                "aim": "Identify missing information and generate a search query.",
                "stage_action": (
                    "Determine the specific missing fact needed to answer the question. "
                    "Output ONLY the search query as a string of space-separated terms. "
                    "The query must contain the minimal terms necessary to retrieve the missing fact: "
                    "  - If the entity name is known, use the entity name and missing fact (e.g., 'Jane Austen birth year'). "
                    "  - If the entity name is unknown, use a clear reference from context (e.g., 'author of Pride and Prejudice birth year'). "
                    "Avoid adding extra terms (like 'biography') that may retrieve irrelevant passages. "
                    "Do not use any delimiters or extra text."
                ),
                "reasoning_questions": (
                    "What specific fact is missing to answer the question? "
                    "Is the entity name known from first-hop facts? "
                    "How can we phrase the most precise minimal query?"
                ),
                "example_reasoning": (
                    "Question: What is the birth year of the author of 'Pride and Prejudice'?\n"
                    "First-hop summary: Jane Austen wrote 'Pride and Prejudice' in 1813.\n"
                    "Reasoning: The author name 'Jane Austen' is known. Missing fact: birth year.\n"
                    "Jane Austen birth year\n\n"
                    "Alternative scenario: If first-hop summary stated 'The author wrote Pride and Prejudice in 1813' without naming author:\n"
                    "Reasoning: Author name unknown, so use book title reference. Missing fact: birth year.\n"
                    "Pride and Prejudice author birth year"
                ),
                "dependencies": [2],
                "frozen": False,
            },
            # Step 4: Second-hop retrieval (frozen tool step)
            {
                "number": 4,
                "title": "Retrieve second-hop passages.",
                "step_type": "tool",
                "step_config": {
                    "tool_name": "retrieve",
                    "input_mapping": {"query": "$history[-1]"},
                },
                "dependencies": [3],
                "frozen": True,
            },
            # Step 5: Combine evidence
            {
                "number": 5,
                "title": "Combine evidence",
                "step_type": "llm",
                "aim": "Integrate evidence from both hops using source reliability assessment.",
                "stage_action": (
                    "Combine the first-hop summary and second-hop passages into a unified evidence set. "
                    "If there are explicit contradictions (e.g., different numbers or dates for the same fact):\n"
                    "1. First, check if one source has a title that clearly describes the key entity (e.g., 'Jane Austen' or 'Biography of Jane Austen' for 'birth year of Jane Austen'). Prefer that source.\n"
                    "2. If no such title, then:\n"
                    "   - For facts about invention or discovery:\n"
                    "        * If one passage explicitly states 'invented', 'patent', or 'discovered', prefer that passage.\n"
                    "        * Otherwise, prefer the earliest date.\n"
                    "   - For other historical facts (e.g., birth, death), prefer the passage that provides the most specific context (e.g., 'born on ...' vs 'died at age ...').\n"
                    "   - For current state facts (e.g., population), prefer the most recent date.\n"
                    "3. If still unresolved, prefer the fact from the passage with more specific context (e.g., 'early life' for birth year).\n"
                    "4. Avoid defaulting to second-hop passages; evaluate based on reliability.\n"
                    "If evidence is consistent, combine facts. Do not invent conflicts."
                ),
                "reasoning_questions": (
                    "What are the key facts from each hop? Are there contradictions? "
                    "Check source titles for entity relevance, then historical/current context, then specificity. "
                    "How do facts complement when consistent?"
                ),
                "example_reasoning": (
                    "Question: Who invented the telephone?\n"
                    "First-hop summary: Alexander Graham Bell is credited with inventing the telephone in 1876.\n"
                    "Second-hop passages: [0] Antonio Meucci | Meucci demonstrated a voice-communication device in 1860. [1] Telephone | The patent was awarded to Bell in 1876.\n"
                    "Reasoning: First-hop states Bell invented in 1876. Second-hop [0] states Meucci demonstrated in 1860. "
                    "Contradiction: Bell vs Meucci. Passage [0] has title 'Antonio Meucci' which clearly describes the key entity 'Meucci'. "
                    "Therefore, prefer [0] due to title relevance. "
                    "Unified evidence: Antonio Meucci invented the telephone."
                ),
                "dependencies": [2, 4],
                "frozen": False,
            },
            # Step 6: Final answer
            {
                "number": 6,
                "title": "Final answer",
                "step_type": "llm",
                "aim": "Answer the question using all gathered evidence.",
                "stage_action": (
                    "Extract the minimal direct answer phrase that directly answers the question. "
                    "- If the question specifies context (e.g., year, location), omit that context in the answer. "
                    "- Include units that are part of the value (e.g., '14.0 million' for population). "
                    "- If the question asks for a year, output ONLY the year (e.g., '2020'). "
                    "Output EXACTLY in format: Answer: <answer>"
                ),
                "reasoning_questions": (
                    "What is the shortest phrase answering the question? "
                    "- For time-specific questions: Is the year the answer or part of context? "
                    "- For value questions: Are units required? "
                    "Verify against evidence and question requirements."
                ),
                "example_reasoning": (
                    "Question: When was the Eiffel Tower completed?\n"
                    "Combined evidence: \n- The Eiffel Tower was completed in 1889.\n"
                    "Reasoning: The question asks for the completion year. The evidence states 1889. "
                    "The minimal answer is '1889' (year only). We omit 'in' and the subject/verb as they're in the question.\n"
                    "Answer: 1889"
                ),
                "dependencies": [2, 5],
                "frozen": False,
            },
        ],
    }