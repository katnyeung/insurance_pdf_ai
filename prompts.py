from langchain.prompts import PromptTemplate

question_refinement_template = PromptTemplate(
    input_variables=["current_info", "missing_info"],
    template="""Based on the following information about the company:
{current_info}

We still need to know:
{missing_info}

INSTRUCTIONS:
1. Select ONE specific missing information category to ask about.
2. Generate a clear, direct question to gather that specific information.
3. Keep your question concise and focused only on the selected category.
4. Format your response as: "CATEGORY: [selected category]\nQUESTION: [your question]"
5. Be confidence, no verification is required.

For example:
CATEGORY: Company size/employee count
QUESTION: How many employees does your company currently have?

Your response:"""
)

recommendation_template = PromptTemplate(
    input_variables=["company_profile", "relevant_policies"],
    template="""Given the following company profile:
{company_profile}

And the following relevant insurance policies:
{relevant_policies}

ROLE: You are an insurance broker helping to identify the best policies for this client.

INSTRUCTIONS:
1. Analyze the company profile to identify their key risks and needs.
2. Review the available policies and select the ones that best address these needs.
3. Clearly name each recommended policy and explain why it's the best fit.

Your response must include:
- POLICY RECOMMENDATIONS: List the specific names of recommended policies in order of priority
- WHY THESE POLICIES: Brief explanation of why each policy is suitable for this client
- COVERAGE HIGHLIGHTS: Key benefits and coverage amounts that address client needs

Keep your response concise and focused on actionable recommendations. Avoid lengthy explanations of policy details unless directly relevant to the client's needs."""
)
    
profile_template = PromptTemplate(
    input_variables=["collected_info"],
    template="""
Based on the following collected information about a company:
{collected_info}

INSTRUCTIONS:
1. Create a single, concise line summarizing all key company details.
2. Include industry, size, revenue, risks, and budget information.
3. Format as a keyword-rich search query.
4. DO NOT include explanations, thinking, or multiple lines.
5. DO NOT use <think> tags or similar markers.

OUTPUT FORMAT:
[Industry] company with [size] employees, [revenue] annual revenue, concerned about [risks], with [budget] constraints.
"""
)