"""Agentic AI flow for intent classification using LangGraph only."""
import json
import logging
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from openai import AsyncOpenAI

from app.config import settings
from app.models.schemas import IntentType, IntentClassification

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the agentic intent classification flow."""
    query: str
    analysis: Optional[str]
    selected_agent: Optional[str]
    classification: Optional[dict]
    final_result: Optional[IntentClassification]


class IntentClassificationAgent:
    """Agentic AI for intent classification using LangGraph."""

    def __init__(self):
        """Initialize the agent."""
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model
        self.graph = None

    def _create_graph(self):
        """Create the agentic workflow graph using LangGraph."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("route_agent", self._route_agent_node)
        workflow.add_node("factual_agent", self._factual_agent_node)
        workflow.add_node("comparison_agent", self._comparison_agent_node)
        workflow.add_node("procedural_agent", self._procedural_agent_node)
        workflow.add_node("diagnostic_agent", self._diagnostic_agent_node)
        workflow.add_node("recommendation_agent", self._recommendation_agent_node)
        workflow.add_node("analytical_agent", self._analytical_agent_node)
        workflow.add_node("finalize", self._finalize_node)

        # Set entry point
        workflow.set_entry_point("analyze_query")

        # Add edges
        workflow.add_edge("analyze_query", "route_agent")

        # Conditional routing
        workflow.add_conditional_edges(
            "route_agent",
            self._route_decision,
            {
                "FACTUAL": "factual_agent",
                "COMPARISON": "comparison_agent",
                "PROCEDURAL": "procedural_agent",
                "DIAGNOSTIC": "diagnostic_agent",
                "RECOMMENDATION": "recommendation_agent",
                "ANALYTICAL": "analytical_agent",
                "__default__": "factual_agent",
            }
        )

        # All agents lead to finalize
        workflow.add_edge("factual_agent", "finalize")
        workflow.add_edge("comparison_agent", "finalize")
        workflow.add_edge("procedural_agent", "finalize")
        workflow.add_edge("diagnostic_agent", "finalize")
        workflow.add_edge("recommendation_agent", "finalize")
        workflow.add_edge("analytical_agent", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    async def _analyze_query_node(self, state: AgentState) -> AgentState:
        """Analyze the query."""
        logger.info(f"📊 Analyzing query...")
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Analyze briefly in one line."},
                {"role": "user", "content": f"Analyze: {state['query']}"}
            ],
            temperature=0.7,
            max_tokens=100
        )
        state["analysis"] = response.choices[0].message.content
        logger.info(f"✅ Analysis: {state['analysis']}")
        return state

    async def _route_agent_node(self, state: AgentState) -> AgentState:
        """Route to appropriate agent."""
        logger.info(f"🔀 Routing to agent...")
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Choose ONE: FACTUAL, COMPARISON, PROCEDURAL, DIAGNOSTIC, RECOMMENDATION, ANALYTICAL. Respond with ONLY the word."},
                {"role": "user", "content": f"Query: {state['query']}"}
            ],
            temperature=0.7,
            max_tokens=20
        )
        agent = response.choices[0].message.content.strip().upper()
        state["selected_agent"] = agent
        logger.info(f"✅ Routed to: {agent}")
        return state

    def _route_decision(self, state: AgentState) -> str:
        """Decide which agent to route to."""
        agent = state.get("selected_agent", "FACTUAL")
        valid = ["FACTUAL", "COMPARISON", "PROCEDURAL", "DIAGNOSTIC", "RECOMMENDATION", "ANALYTICAL"]
        return agent if agent in valid else "FACTUAL"

    async def _factual_agent_node(self, state: AgentState) -> AgentState:
        """Factual question agent."""
        logger.info(f"🔍 Factual Agent processing...")
        state["classification"] = await self._classify_intent(state["query"], "factual")
        return state

    async def _comparison_agent_node(self, state: AgentState) -> AgentState:
        """Comparison agent."""
        logger.info(f"🔍 Comparison Agent processing...")
        state["classification"] = await self._classify_intent(state["query"], "comparison")
        return state

    async def _procedural_agent_node(self, state: AgentState) -> AgentState:
        """Procedural agent."""
        logger.info(f"🔍 Procedural Agent processing...")
        state["classification"] = await self._classify_intent(state["query"], "procedural")
        return state

    async def _diagnostic_agent_node(self, state: AgentState) -> AgentState:
        """Diagnostic agent."""
        logger.info(f"🔍 Diagnostic Agent processing...")
        state["classification"] = await self._classify_intent(state["query"], "diagnostic")
        return state

    async def _recommendation_agent_node(self, state: AgentState) -> AgentState:
        """Recommendation agent."""
        logger.info(f"🔍 Recommendation Agent processing...")
        state["classification"] = await self._classify_intent(state["query"], "recommendation")
        return state

    async def _analytical_agent_node(self, state: AgentState) -> AgentState:
        """Analytical agent."""
        logger.info(f"🔍 Analytical Agent processing...")
        state["classification"] = await self._classify_intent(state["query"], "analytical")
        return state

    async def _classify_intent(self, query: str, agent_type: str) -> dict:
        """Classify intent using OpenAI."""
        prompt = f"""Classify as {agent_type}. Respond with JSON:
{{"intent": "factual_question|comparison|explanation|how_to|troubleshooting|recommendation|summary|clarification|related_topics|opinion", "confidence": 0.0-1.0, "entities": [], "processing_strategy": "standard_retrieval|comparative_analysis|step_by_step|diagnostic_analysis|recommendation_engine|analytical_summary", "reasoning": "explanation"}}

Query: {query}"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"You are a {agent_type} intent classifier."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )

        try:
            content = response.choices[0].message.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except:
            pass

        return {"intent": "factual_question", "confidence": 0.7, "entities": [], "processing_strategy": "standard_retrieval", "reasoning": "Default"}

    async def _finalize_node(self, state: AgentState) -> AgentState:
        """Finalize the classification."""
        logger.info(f"✔️ Finalizing...")
        classification = state.get("classification", {})
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
            entities=classification.get("entities", []),
            processing_strategy=classification.get("processing_strategy", "standard_retrieval"),
            reasoning=classification.get("reasoning", "")
        )
        logger.info(f"✅ Finalized: {intent} ({confidence:.2%})")
        return state

    async def classify_intent(self, query: str) -> IntentClassification:
        """Classify intent using agentic AI flow."""
        if self.graph is None:
            self.graph = self._create_graph()

        logger.info(f"\n{'='*80}")
        logger.info(f"🤖 AGENTIC INTENT CLASSIFICATION STARTED")
        logger.info(f"{'='*80}")
        logger.info(f"Query: {query}")

        try:
            initial_state: AgentState = {
                "query": query,
                "analysis": None,
                "selected_agent": None,
                "classification": None,
                "final_result": None,
            }

            final_state = await self.graph.ainvoke(initial_state)

            logger.info(f"{'='*80}")
            logger.info(f"🎉 AGENTIC INTENT CLASSIFICATION COMPLETE")
            logger.info(f"{'='*80}\n")

            return final_state["final_result"]

        except Exception as e:
            logger.error(f"❌ Error: {str(e)}")
            return IntentClassification(
                intent=IntentType.FACTUAL_QUESTION,
                confidence=0.5,
                entities=[],
                processing_strategy="standard_retrieval",
                reasoning=f"Fallback: {str(e)}"
            )


# Global instance
agentic_intent_classifier = IntentClassificationAgent()

