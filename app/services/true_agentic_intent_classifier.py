"""TRUE Agentic AI flow for intent classification using LangGraph with tool use."""
import json
import logging
import re
from typing import TypedDict, Optional, List, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from openai import AsyncOpenAI

from app.config import settings
from app.models.schemas import IntentType, IntentClassification

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the agentic intent classification flow."""
    query: str
    messages: List[dict]
    analysis_results: dict
    tool_calls: List[str]
    iterations: int
    final_result: Optional[IntentClassification]


class TrueAgenticIntentClassifier:
    """TRUE Agentic AI with tool use and iterative reasoning."""

    def __init__(self):
        """Initialize the agent."""
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model
        self.graph = None
        self.max_iterations = 5

    def _create_graph(self):
        """Create the agentic workflow graph with tool use."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("analyze_query_tool", self._analyze_query_tool)
        workflow.add_node("extract_entities_tool", self._extract_entities_tool)
        workflow.add_node("classify_intent_tool", self._classify_intent_tool)
        workflow.add_node("finalize", self._finalize_node)

        # Set entry point
        workflow.set_entry_point("agent")

        # Agent decides what to do next
        workflow.add_conditional_edges(
            "agent",
            self._agent_decision,
            {
                "analyze_query": "analyze_query_tool",
                "extract_entities": "extract_entities_tool",
                "classify_intent": "classify_intent_tool",
                "finalize": "finalize",
                "continue": "agent",
            }
        )

        # Tools loop back to agent
        workflow.add_edge("analyze_query_tool", "agent")
        workflow.add_edge("extract_entities_tool", "agent")
        workflow.add_edge("classify_intent_tool", "agent")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    async def _agent_node(self, state: AgentState) -> AgentState:
        """Main agent that decides what to do."""
        logger.info(f"🤖 Agent iteration {state['iterations']}")

        # Build system prompt
        system_prompt = """You are an intelligent intent classification agent. Your job is to classify user queries into intent types.

Available tools:
1. analyze_query - Analyze the query structure and content
2. extract_entities - Extract key entities from the query
3. classify_intent - Classify the query intent based on analysis

You MUST use tools to gather information before making a final decision.
After using tools, respond with JSON containing your next action:
{
    "action": "analyze_query|extract_entities|classify_intent|finalize",
    "reasoning": "explanation of why you chose this action"
}

When you have enough information, use action "finalize" to complete the task."""

        # Add user message if first iteration
        if state["iterations"] == 0:
            state["messages"].append({
                "role": "user",
                "content": f"Classify the intent of this query: {state['query']}"
            })

        # Get agent decision
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}] + state["messages"],
            temperature=0.7,
            max_tokens=500
        )

        agent_response = response.choices[0].message.content
        state["messages"].append({"role": "assistant", "content": agent_response})

        logger.info(f"Agent response: {agent_response[:100]}...")
        return state

    def _agent_decision(self, state: AgentState) -> str:
        """Decide what the agent should do next."""
        if state["iterations"] >= self.max_iterations:
            logger.info("Max iterations reached, finalizing...")
            return "finalize"

        # Parse agent response to get action
        try:
            last_message = state["messages"][-1]["content"]
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', last_message, re.DOTALL)
            if json_match:
                action_data = json.loads(json_match.group())
                action = action_data.get("action", "finalize")
                if action in ["analyze_query", "extract_entities", "classify_intent", "finalize"]:
                    return action
        except:
            pass

        # Default to finalize if we can't parse
        return "finalize"

    async def _analyze_query_tool(self, state: AgentState) -> AgentState:
        """Tool: Analyze query structure."""
        logger.info("🔧 Using analyze_query tool...")

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Analyze the query and provide JSON: {\"query_type\": \"question|statement|command\", \"complexity\": \"simple|moderate|complex\", \"domain\": \"technical|general|other\"}"},
                {"role": "user", "content": f"Analyze: {state['query']}"}
            ],
            temperature=0.7,
            max_tokens=200
        )

        try:
            analysis = json.loads(response.choices[0].message.content)
            state["analysis_results"]["query_analysis"] = analysis
            state["tool_calls"].append("analyze_query")
        except:
            state["analysis_results"]["query_analysis"] = {}

        state["messages"].append({
            "role": "user",
            "content": f"Tool result: Query analysis complete. Result: {state['analysis_results'].get('query_analysis', {})}"
        })
        state["iterations"] += 1
        return state

    async def _extract_entities_tool(self, state: AgentState) -> AgentState:
        """Tool: Extract entities from query."""
        logger.info("🔧 Using extract_entities tool...")

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Extract entities from the query. Respond with JSON: {\"entities\": [\"entity1\", \"entity2\"], \"entity_types\": [\"type1\", \"type2\"]}"},
                {"role": "user", "content": f"Extract entities from: {state['query']}"}
            ],
            temperature=0.7,
            max_tokens=200
        )

        try:
            entities = json.loads(response.choices[0].message.content)
            state["analysis_results"]["entities"] = entities
            state["tool_calls"].append("extract_entities")
        except:
            state["analysis_results"]["entities"] = {"entities": [], "entity_types": []}

        state["messages"].append({
            "role": "user",
            "content": f"Tool result: Entity extraction complete. Result: {state['analysis_results'].get('entities', {})}"
        })
        state["iterations"] += 1
        return state

    async def _classify_intent_tool(self, state: AgentState) -> AgentState:
        """Tool: Classify intent based on analysis."""
        logger.info("🔧 Using classify_intent tool...")

        analysis_context = json.dumps(state["analysis_results"], indent=2)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Classify the intent. Respond with JSON: {\"intent\": \"factual_question|comparison|explanation|how_to|troubleshooting|recommendation|summary|clarification|related_topics|opinion\", \"confidence\": 0.0-1.0, \"reasoning\": \"explanation\"}"},
                {"role": "user", "content": f"Query: {state['query']}\n\nAnalysis: {analysis_context}"}
            ],
            temperature=0.7,
            max_tokens=300
        )

        try:
            classification = json.loads(response.choices[0].message.content)
            state["analysis_results"]["classification"] = classification
            state["tool_calls"].append("classify_intent")
        except:
            state["analysis_results"]["classification"] = {"intent": "factual_question", "confidence": 0.5, "reasoning": "Default"}

        state["messages"].append({
            "role": "user",
            "content": f"Tool result: Intent classification complete. Result: {state['analysis_results'].get('classification', {})}"
        })
        state["iterations"] += 1
        return state

    async def _finalize_node(self, state: AgentState) -> AgentState:
        """Finalize the classification."""
        logger.info(f"✔️ Finalizing after {state['iterations']} iterations...")

        classification = state["analysis_results"].get("classification", {})
        intent_str = classification.get("intent", "factual_question")

        intent_map = {
            "factual_question": IntentType.FACTUAL_QUESTION,
            "comparison": IntentType.COMPARISON,
            "explanation": IntentType.EXPLANATION,
            "how_to": IntentType.HOW_TO,
            "troubleshooting": IntentType.TROUBLESHOOTING,
            "recommendation": IntentType.RECOMMENDATION,
            "summary": IntentType.SUMMARY,
            "clarification": IntentType.CLARIFICATION,
            "related_topics": IntentType.RELATED_TOPICS,
            "opinion": IntentType.OPINION,
        }

        intent = intent_map.get(intent_str.lower(), IntentType.FACTUAL_QUESTION)
        confidence = float(classification.get("confidence", 0.7))
        confidence = min(max(confidence, 0.0), 1.0)

        state["final_result"] = IntentClassification(
            intent=intent,
            confidence=confidence,
            entities=state["analysis_results"].get("entities", {}).get("entities", []),
            processing_strategy="agentic_analysis",
            reasoning=classification.get("reasoning", "")
        )

        logger.info(f"✅ Classification: {intent} ({confidence:.2%})")
        logger.info(f"📊 Tools used: {', '.join(state['tool_calls'])}")
        return state

    async def classify_intent(self, query: str) -> IntentClassification:
        """Classify intent using TRUE agentic AI flow."""
        if self.graph is None:
            self.graph = self._create_graph()

        logger.info(f"\n{'='*80}")
        logger.info(f"🤖 TRUE AGENTIC INTENT CLASSIFICATION STARTED")
        logger.info(f"{'='*80}")
        logger.info(f"Query: {query}")

        try:
            initial_state: AgentState = {
                "query": query,
                "messages": [],
                "analysis_results": {},
                "tool_calls": [],
                "iterations": 0,
                "final_result": None,
            }

            final_state = await self.graph.ainvoke(initial_state)

            logger.info(f"{'='*80}")
            logger.info(f"🎉 TRUE AGENTIC INTENT CLASSIFICATION COMPLETE")
            logger.info(f"{'='*80}\n")

            return final_state["final_result"]

        except Exception as e:
            logger.error(f"❌ Error: {str(e)}")
            return IntentClassification(
                intent=IntentType.FACTUAL_QUESTION,
                confidence=0.5,
                entities=[],
                processing_strategy="fallback",
                reasoning=f"Error: {str(e)}"
            )


# Global instance
true_agentic_intent_classifier = TrueAgenticIntentClassifier()

