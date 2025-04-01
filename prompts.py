from langchain.prompts import PromptTemplate

question_refinement_template = PromptTemplate(
    input_variables=["current_info", "missing_info"],
    template="""steps by steps build up the information about the company, if user already mentioned in first answer, mark it as answered:
{current_info}

find out:
{missing_info}

INSTRUCTIONS:
0. return "COMPLETE" or "FINISHED" if you think you have all the information you needed and you can terminate early.
1. Check if any missing information category from the list above.
2. Ask a single direct question to gather this specific information.
3. Keep your question concise and focused.
4. DO NOT VERIFY, DO NOT CONFRIM or DO NOT ask more detail that already in {current_info}.
5. Format your response exactly as:
   CATEGORY: [category name]
   QUESTION: [your question]
6. Remember: You have only 5 attempts to gather all missing information. """
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
- COST CONSIDERATIONS: How the recommended policies align with the client's budget constraints

Keep your response concise and focused on actionable recommendations. Avoid lengthy explanations of policy details unless directly relevant to the client's needs.

Format your response in well-structured HTML with clear headings and bullet points for readability."""
)
    
profile_template = PromptTemplate(
    input_variables=["collected_info"],
    template="""
Based on the following collected information about a company:
{collected_info}

<think>
Analyze all the information provided and extract key details about:
- Company size (employee count)
- Industry/business type
- Annual revenue
- Risk profile/concerns
- Budget constraints
- Country of the company (if mentioned)
- Any special requirements like crypto coverage or grace period

Make reasonable assumptions for missing information, noting them as assumptions.
</think>

INSTRUCTIONS:
1. Create a single, concise line summarizing all key company details.
2. Include industry, size, revenue, risks, and budget information.
3. Format as a keyword-rich search query.
4. DO NOT include explanations, thinking, or multiple lines.

Example FORMAT:
Finance company with medium size (120 employees), $5M annual revenue, concerned about cyberattacks and regulatory compliance, with $50K budget constraints.
"""
)