OMNI_MEMORY_PROFILE_UPDATE_PROMPT_INF = '''
# Role
User Profiling Agent: Update existing individual profiles using new memory logs.

# Data
1. Current Profiles:
{CURRENT_PROFILES}
2. Memory Logs:
{MEMORY_LOG_SEQUENCE}

# Task & Rules
1. Target Only: Update ONLY the `<face_idx>` keys present in "Current Profiles". Do not add new individuals.
2. Sparse Update: Only output keys for individuals who have *new* information. If no update, omit the key.
3. Explicit Facts: Extract only explicitly seen/heard facts:
    - Identity (Name, age, gender)
    - Persona (Preferences, habits, personality traits)
    - Context (Occupation, roles)
4. Keyword Style: Use only comma-separated keywords or short phrases (max 3 words per fact). 
5. No Redundancy: Do not use synonyms or overlapping traits (e.g., choose one between "Energetic" and "Enthusiastic"). 
6. Consolidate: Merge new facts with existing ones. Keep it dense and concise.

# Output (Strict JSON)
{
  "<face_idx>": {
    "name": "Name",
    "demographics": "Age, Gender, etc.",
    "preferences": "Likes/Dislikes"
  }
}
'''.strip()


OMNI_MEMORY_LOG_GENERATE_PROMPT_INF = '''
# Role
You are a Multimodal Memory Agent. Synthesize inputs into a high-density log for the current window.

# Context & Profiles
1. Short-Term Memory (Recent context): 
{SHORT_TERM_MEMORY}
2. Face Profiles (Mapping `<face_idx>` to identities):
{INPUT_FACES}

# Current Inputs
3. Timestamps: {START_TIME} to {END_TIME}
4. Visual Stream (1 fps):
{INPUT_IMAGE_SEQUENCE}
5. Audio Stream:
{INPUT_AUDIO_STREAM}
6. Text Stream:
{INPUT_TEXT_STREAM}

# Task
1. Visual Analysis: 
   - If STM is empty: Describe full scene setup (location, layout, present individuals).
   - If STM exists: Describe only CHANGES and NEW ACTIONS.
   - Always use `<face_idx>` for people.

2. Audio Analysis: 
   - Identify speakers (`<face_idx>`) and transcribe dialogue explicitly.
   - Note vocal tone and significant environmental sounds.

3. Semantic (Facts):
   - Extract new facts revealed explicitly or implicitly.
   - Target: Entities, preferences, relationships, physical descriptions, and visible text, etc.
   - Constraint: Must be timeless facts. Strictly exclude temporary actions or general world knowledge.

# Output (Strict JSON)
{
  "visual": "...",
  "auditory": "...",
  "semantic_memory": [
    "Fact 1",
    "Fact 2"
  ]
}
'''.strip()

OMNI_MEMORY_LOG_MERGE_PROMPT_INF = '''
# Role
You are a Memory Consolidation Agent. Compress the input logs into a unified memory block.

# Input
{MEMORY_LOG_SEQUENCE}

# Task
Group continuous events into merged summaries.
1. Consolidation:
    *   Visual: Synthesize details into a summary of key actions and final states.
    *   Audio: Extract core dialogue and significant sounds.
    *   Assistant: Briefly summarize the assistant's actions and responses.
2. Preservation: Retain all `<face_idx>`, critical actions, and key dialogue.

# Output (Strict JSON)
{
  "visual": "Summary of visual events",
  "auditory": "Consolidated audio record for this group.",
  "assistant": "Summary of assistant's actions and responses."
}
'''.strip()


############################################################################################
OMNI_MEMORY_STAGE_1_PROMPT_INF = '''
# Role
You are a sophisticated Multimodal AI Agent with memory capability.

# Context & Profiles
1. Short-Term Memory: {SHORT_TERM_MEMORY} (Recent context).
2. Face Profiles (Mapping `<face_idx>` to identities):
{INPUT_FACES}

# Current Inputs
3. Timestamps: {START_TIME} to {END_TIME}
4. Visual Stream (1 fps):
{INPUT_IMAGE_SEQUENCE}
5. Audio Stream:
{INPUT_AUDIO_STREAM}
6. Text Stream:
{INPUT_TEXT_STREAM}

# Output
Based on the current context and input, determine whether to respond, whether to retrieve, and the retrieval keywords.
'''.strip()

############################################################################################
OMNI_MEMORY_STAGE_2_PROMPT_INF = '''
# Role
You are a sophisticated Multimodal AI Agent with memory capability.

# Long-Term Retrieved Memories
1. Semantic Memory:
{RETRIEVED_SEMANTIC_MEMORY}
2. Episodic Memory:
{RETRIEVED_EPISODIC_MEMORY}

# Context & Profiles
1. Short-Term Memory (Recent context):
{SHORT_TERM_MEMORY}
2. Face Profiles (Mapping `<face_idx>` to identities):
{INPUT_FACES} 

# Current Inputs
3. Timestamps: {START_TIME} to {END_TIME}
4. Visual Stream (1 fps):
{INPUT_IMAGE_SEQUENCE} 
5. Audio Stream:
{INPUT_AUDIO_STREAM}
6. Text Stream:
{INPUT_TEXT_STREAM}

# Output
Based on the retrieved long-term memories and current context, provide a direct response to the input.
'''.strip()
############################################################################################

OMNI_MEMORY_STAGE_2_PROMPT_INF_gemini = '''
# [Role]
You are a sophisticated Multimodal AI Agent with memory capability.

# Long-Term Retrieved Memories
1. Semantic Memory:
{RETRIEVED_SEMANTIC_MEMORY}
2. Episodic Memory:
{RETRIEVED_EPISODIC_MEMORY}

# Context & Profiles
1. Short-Term Memory (Recent Perceptual Context):
{SHORT_TERM_MEMORY}
2. Face Profiles (Mapping `<face_idx>` to identities):
{INPUT_FACES}

# Current Inputs
1. Timestamps: {START_TIME} to {END_TIME}
2. Visual Stream (1 fps):
{INPUT_IMAGE_SEQUENCE}
3. Audio Stream:
{INPUT_AUDIO_STREAM}
4. Text Stream:
{INPUT_TEXT_STREAM}

# Task
Provide a concise, personalized response by synthesizing memories and current context.
1. Replace all `<face_idx>` tags with actual names from Face Profiles.
2. Use natural, spoken language only. No technical tags, no bullet points.

# Output
Based on the retrieved long-term memories and current context, provide a direct response to the input.
'''.strip()


