"""
╔══════════════════════════════════════════════════════════════════╗
║      Autonomous LinkedIn Content Manager — Powered by CrewAI     ║
║      5-Agent Pipeline: Research → Write → Critique → Optimize     ║
║                         → Schedule                                ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

# ─────────────────────────────────────────────
# 1. LOAD ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────
load_dotenv()

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY  = os.getenv("SERPER_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    print("❌  OPENAI_API_KEY not found. Please set it in your .env file.")
    sys.exit(1)

if not SERPER_API_KEY:
    print("❌  SERPER_API_KEY not found. Please set it in your .env file.")
    sys.exit(1)

# Export so CrewAI / LangChain can pick them up automatically
os.environ["OPENAI_API_KEY"]  = OPENAI_API_KEY
os.environ["SERPER_API_KEY"]  = SERPER_API_KEY
os.environ["OPENAI_MODEL_NAME"] = OPENAI_MODEL


# ─────────────────────────────────────────────
# 2. TOOLS
# ─────────────────────────────────────────────
search_tool  = SerperDevTool()   # Web search via Serper
scrape_tool  = ScrapeWebsiteTool()  # Web page scraper


# ─────────────────────────────────────────────
# 3. AGENTS
# ─────────────────────────────────────────────

# --- Agent 1: Trend Researcher ---
trend_researcher = Agent(
    role="LinkedIn Trend Researcher",
    goal=(
        "Research the latest trending topics, hashtags, and content themes "
        "relevant to a given niche/industry on LinkedIn."
    ),
    backstory=(
        "You are an expert social media researcher who monitors LinkedIn trends, "
        "viral posts, and industry news in real time. You know exactly what kind "
        "of content drives impressions, comments, and shares on LinkedIn right now. "
        "You have a sharp eye for emerging angles before they hit saturation."
    ),
    tools=[search_tool, scrape_tool],
    verbose=True,
    allow_delegation=False,
)

# --- Agent 2: Content Writer ---
content_writer = Agent(
    role="LinkedIn Content Writer",
    goal=(
        "Write an engaging, high-quality LinkedIn post based on the research "
        "brief provided, following best practices for the platform."
    ),
    backstory=(
        "You are a seasoned LinkedIn ghostwriter who has produced viral posts for "
        "C-suite executives, startup founders, and thought leaders across industries. "
        "You understand LinkedIn's algorithm, hook writing, storytelling frameworks "
        "(AIDA, PAS, etc.), and the perfect placement of a CTA. "
        "Your writing is conversational yet professional — never corporate-speak."
    ),
    verbose=True,
    allow_delegation=False,
)

# --- Agent 3: Content Critic ---
content_critic = Agent(
    role="Content Quality Critic",
    goal=(
        "Review the LinkedIn post draft and provide detailed, honest, constructive "
        "feedback covering engagement potential, tone, structure, clarity, hook "
        "strength, and CTA effectiveness."
    ),
    backstory=(
        "You are a harsh-but-fair content editor who has reviewed thousands of "
        "LinkedIn posts. You know exactly what separates a post that gets 10 likes "
        "from one that earns 10,000 impressions and 200 comments. "
        "You never sugarcoat weaknesses, but every critique you give is specific "
        "and immediately actionable. You think like a scroll-stopping reader, "
        "not a polite colleague."
    ),
    verbose=True,
    allow_delegation=False,
)

# --- Agent 4: Content Optimizer ---
content_optimizer = Agent(
    role="LinkedIn Post Optimizer",
    goal=(
        "Take the original draft and the critic's feedback, then rewrite and "
        "polish the post to maximise LinkedIn engagement."
    ),
    backstory=(
        "You are a LinkedIn growth expert and conversion copywriter. "
        "You know the exact formatting tricks that make posts explode on mobile: "
        "one-sentence lines, strategic white space, power-word hooks, emoji used "
        "sparingly but effectively, and hashtag placement that aids discovery "
        "without looking spammy. You incorporate every piece of feedback to "
        "produce a flawless, publish-ready post."
    ),
    verbose=True,
    allow_delegation=False,
)

# --- Agent 5: Scheduling / Publishing Strategist ---
scheduling_agent = Agent(
    role="LinkedIn Publishing Strategist",
    goal=(
        "Determine the optimal posting time, finalize the formatted post with "
        "hashtags, and produce a complete publishing brief with a first-hour "
        "engagement strategy."
    ),
    backstory=(
        "You are a LinkedIn analytics expert with deep knowledge of platform "
        "algorithms, audience behavior patterns, and timezone-aware posting "
        "windows. You've helped brands grow from zero to hundreds of thousands "
        "of followers by getting the timing and distribution strategy exactly "
        "right. You wrap up every campaign with a tight, actionable brief that "
        "any community manager can execute immediately."
    ),
    verbose=True,
    allow_delegation=False,
)


# ─────────────────────────────────────────────
# 4. TASKS
# ─────────────────────────────────────────────

def build_tasks(topic: str):
    """Construct the five sequential tasks for the given topic."""

    # Task 1 – Research
    research_task = Task(
        description=(
            f"Research the latest trends, viral content patterns, and hot topics "
            f"on LinkedIn for the niche/industry: **{topic}**.\n\n"
            "Use the search and scraping tools to:\n"
            "• Identify 3–5 trending angles or subtopics that are currently popular.\n"
            "• Find 8–12 high-performing hashtags relevant to this niche.\n"
            "• Surface 3–4 compelling content hooks (opening-line patterns) that "
            "are getting traction.\n"
            "• Note any recent news, studies, or data points that could anchor the post."
        ),
        expected_output=(
            "A structured research brief containing:\n"
            "1. Trending angles/subtopics (3–5 bullet points)\n"
            "2. Top-performing hashtags (8–12)\n"
            "3. Proven hook patterns / opening-line ideas (3–4)\n"
            "4. Key data points, stats, or news hooks relevant to the topic"
        ),
        agent=trend_researcher,
    )

    # Task 2 – Writing
    writing_task = Task(
        description=(
            f"Using the research brief provided, write a compelling LinkedIn post "
            f"about **{topic}**.\n\n"
            "Requirements:\n"
            "• Strong hook in the first 1–2 lines (before the 'see more' cut-off).\n"
            "• Value-driven or story-driven body (150–300 words total).\n"
            "• Clear, single call-to-action (CTA) at the end.\n"
            "• 4–6 relevant hashtags appended after the post body.\n"
            "• Conversational, professional tone — no jargon or buzzword soup.\n"
            "• Use the trending angles and hooks from the research."
        ),
        expected_output=(
            "A complete LinkedIn post draft including:\n"
            "• Hook (first 1–2 lines)\n"
            "• Body with storytelling or value delivery\n"
            "• CTA\n"
            "• Hashtags"
        ),
        agent=content_writer,
        context=[research_task],   # fed the output of research_task
    )

    # Task 3 – Critique
    critique_task = Task(
        description=(
            "Critically review the LinkedIn post draft provided.\n\n"
            "Evaluate each dimension and provide an honest score (1–10) plus "
            "specific improvement notes:\n"
            "1. Hook strength — will readers click 'see more'?\n"
            "2. Storytelling / value delivery quality\n"
            "3. Engagement potential (comments, shares, saves)\n"
            "4. CTA clarity and persuasiveness\n"
            "5. Tone consistency and authenticity\n"
            "6. Formatting suitability for LinkedIn mobile\n"
            "7. Hashtag relevance and quantity\n"
            "8. Overall viral potential\n\n"
            "Be direct and specific. Vague praise is useless. "
            "Point to the exact words or sentences that need changing."
        ),
        expected_output=(
            "A detailed critique report with:\n"
            "• Score per dimension (1–10) and one-line reasoning\n"
            "• Overall score out of 10\n"
            "• Top 3 strengths\n"
            "• Top 3–5 specific, actionable improvements\n"
            "• Suggested rewrites for the weakest lines"
        ),
        agent=content_critic,
        context=[writing_task],
    )

    # Task 4 – Optimization
    optimization_task = Task(
        description=(
            "You have the original LinkedIn post draft AND the critic's detailed "
            "feedback. Your job is to produce the definitive, publish-ready version.\n\n"
            "Apply every actionable suggestion from the critique:\n"
            "• Rewrite the hook if needed to guarantee the 'see more' click.\n"
            "• Tighten every sentence — cut fluff, add punch.\n"
            "• Format for mobile: short lines, strategic blank lines between paragraphs.\n"
            "• Use 1–3 emojis maximum, placed where they add visual rhythm.\n"
            "• Sharpen the CTA to a single, unmistakable ask.\n"
            "• Optimise hashtags: 4–6 targeted tags, not a wall of #spam.\n"
            "• Keep total length 150–300 words."
        ),
        expected_output=(
            "The final, publish-ready LinkedIn post — fully formatted, polished, "
            "and optimised for maximum engagement. "
            "Include a brief note (2–3 sentences) explaining the key changes made "
            "and why they improve performance."
        ),
        agent=content_optimizer,
        context=[writing_task, critique_task],
    )

    # Task 5 – Scheduling
    scheduling_task = Task(
        description=(
            f"You have the final optimised LinkedIn post about **{topic}**.\n\n"
            "Produce the complete Publishing Brief:\n"
            "1. Recommended posting day & time (with timezone, e.g. Tuesday 8:30 AM IST).\n"
            "   Base your recommendation on the target audience for this topic.\n"
            "2. The final post formatted exactly as it should be copy-pasted into LinkedIn.\n"
            "3. Hashtag strategy — list all hashtags, group by tier "
            "   (broad / mid-niche / long-tail).\n"
            "4. First-hour engagement playbook: 3–5 actions to take immediately "
            "   after posting to boost early momentum (e.g. reply to every comment, "
            "   share in relevant groups, etc.).\n"
            "5. One-line summary of what makes this post likely to perform well."
        ),
        expected_output=(
            "A complete, formatted Publishing Brief with:\n"
            "• Recommended posting time (day, time, timezone)\n"
            "• Final post (copy-paste ready)\n"
            "• Tiered hashtag list\n"
            "• First-hour engagement playbook (bullet points)\n"
            "• Performance prediction summary (1–2 sentences)"
        ),
        agent=scheduling_agent,
        context=[optimization_task],
    )

    return [
        research_task,
        writing_task,
        critique_task,
        optimization_task,
        scheduling_task,
    ]


# ─────────────────────────────────────────────
# 5. CREW ASSEMBLY & EXECUTION
# ─────────────────────────────────────────────

def run_linkedin_manager(topic: str):
    tasks = build_tasks(topic)

    crew = Crew(
        agents=[
            trend_researcher,
            content_writer,
            content_critic,
            content_optimizer,
            scheduling_agent,
        ],
        tasks=tasks,
        process=Process.sequential,   # strict pipeline order
        verbose=True,
        memory=False,                  # agents share memory across tasks
    )

    print("\n" + "═" * 65)
    print(f"  🚀  Starting LinkedIn Content Pipeline for: {topic!r}")
    print("═" * 65 + "\n")

    result = crew.kickoff(inputs={"topic": topic})

    print("\n" + "═" * 65)
    print("  ✅  FINAL PUBLISHING BRIEF")
    print("═" * 65)
    print(result)
    print("═" * 65 + "\n")

    return result


# ─────────────────────────────────────────────
# 6. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("SCRIPT IS RUNNING")
    import traceback

    try:
        print("Loading .env...")
        load_dotenv()
        print(f"OPENAI_API_KEY set: {bool(os.getenv('OPENAI_API_KEY'))}")
        print(f"SERPER_API_KEY set: {bool(os.getenv('SERPER_API_KEY'))}")

        print("Importing CrewAI tools...")
        from crewai_tools import SerperDevTool, ScrapeWebsiteTool
        print("Tools imported OK")

        if len(sys.argv) > 1:
            topic_input = " ".join(sys.argv[1:])
        else:
            topic_input = input("Enter topic:\n> ").strip()

        print(f"Topic: {topic_input}")
        run_linkedin_manager(topic_input)

    except Exception as e:
        print("\n❌ ERROR CAUGHT:")
        traceback.print_exc()