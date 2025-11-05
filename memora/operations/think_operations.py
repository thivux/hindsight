"""
Think operations for formulating answers based on agent and world facts.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ThinkOperationsMixin:
    """Mixin class for think operations."""

    async def think_async(
        self,
        agent_id: str,
        query: str,
        thinking_budget: int = 50,
        top_k: int = 10,
        model: str = "openai/gpt-oss-120b",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        """
        Think and formulate an answer using agent identity, world facts, and opinions.

        This method:
        1. Retrieves agent facts (agent's identity and past actions)
        2. Retrieves world facts (general knowledge)
        3. Retrieves existing opinions (agent's formed perspectives)
        4. Uses Groq LLM to formulate an answer
        5. Extracts and stores any new opinions formed during thinking
        6. Returns plain text answer and the facts used

        Args:
            agent_id: Agent identifier
            query: Question to answer
            thinking_budget: Number of memory units to explore
            top_k: Maximum facts to retrieve
            model: LLM model to use (default: openai/gpt-oss-120b)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Dict with:
                - text: Plain text answer (no markdown)
                - based_on: Dict with 'world', 'agent', and 'opinion' fact lists
                - new_opinions: List of newly formed opinions
        """
        # Use cached LLM client
        if self._llm_client is None:
            raise ValueError("GROQ_API_KEY environment variable not set")

        client = self._llm_client

        # Steps 1-3: Run all three searches in parallel
        (agent_results, _), (world_results, _), (opinion_results, _) = await asyncio.gather(
            # Get agent facts (identity)
            self.search_async(
                agent_id=agent_id,
                query=query,
                thinking_budget=thinking_budget,
                top_k=top_k,
                enable_trace=False,
                fact_type='agent'
            ),
            # Get world facts
            self.search_async(
                agent_id=agent_id,
                query=query,
                thinking_budget=thinking_budget,
                top_k=top_k,
                enable_trace=False,
                fact_type='world'
            ),
            # Get existing opinions
            self.search_async(
                agent_id=agent_id,
                query=query,
                thinking_budget=thinking_budget,
                top_k=top_k,
                enable_trace=False,
                fact_type='opinion'
            )
        )

        # Step 4: Format facts for LLM with full details as JSON
        import json

        def format_facts(facts):
            if not facts:
                return "[]"
            formatted = []
            for fact in facts:
                fact_obj = {
                    "text": fact['text']
                }

                # Add context if available
                if fact.get('context'):
                    fact_obj["context"] = fact['context']

                # Add event_date if available
                if fact.get('event_date'):
                    from datetime import datetime
                    event_date = fact['event_date']
                    if isinstance(event_date, str):
                        fact_obj["event_date"] = event_date
                    elif isinstance(event_date, datetime):
                        fact_obj["event_date"] = event_date.strftime('%Y-%m-%d %H:%M:%S')

                # Add score if available
                if fact.get('score') is not None:
                    fact_obj["score"] = fact['score']

                formatted.append(fact_obj)

            return json.dumps(formatted, indent=2)

        agent_facts_text = format_facts(agent_results)
        world_facts_text = format_facts(world_results)
        opinion_facts_text = format_facts(opinion_results)

        # Step 5: Call Groq to formulate answer
        prompt = f"""You are an AI assistant answering a question based on retrieved facts provided in JSON format.

AGENT IDENTITY (what the agent has done):
{agent_facts_text}

WORLD FACTS (general knowledge):
{world_facts_text}

YOUR EXISTING OPINIONS (perspectives you've formed):
{opinion_facts_text}

QUESTION: {query}

The facts above are provided as JSON arrays. Each fact may include:
- text: The fact content
- context: Additional context information
- event_date: When the fact occurred
- score: Relevance score

Provide a helpful, accurate answer based on the facts above. Be consistent with your existing opinions. If the facts don't contain enough information to answer the question, say so clearly. Do not use markdown formatting - respond in plain text only.

If you form any new opinions while thinking about this question, state them clearly in your answer."""

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Always respond in plain text without markdown formatting. You can form and express opinions based on facts."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        answer_text = response.choices[0].message.content.strip()

        # Step 6: Extract new opinions from the answer
        new_opinions = await self._extract_opinions_from_text(
            client=client,
            text=answer_text,
            model=model
        )

        # Step 7: Store new opinions (schedule as background tasks, don't wait)
        if new_opinions:
            current_time = datetime.now(timezone.utc)
            for opinion_dict in new_opinions:
                task = asyncio.create_task(
                    self.put_async(
                        agent_id=agent_id,
                        content=opinion_dict["text"],
                        context=f"formed during thinking about: {query}",
                        event_date=current_time,
                        fact_type_override='opinion',
                        confidence_score=opinion_dict["confidence"]
                    )
                )
                # Track task and auto-remove when done
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

        # Step 8: Return response with facts split by type
        return {
            "text": answer_text,
            "based_on": {
                "world": world_results,
                "agent": agent_results,
                "opinion": opinion_results
            },
            "new_opinions": new_opinions
        }

    async def _extract_opinions_from_text(
        self,
        client,
        text: str,
        model: str
    ) -> List[Dict[str, Any]]:
        """
        Extract opinions with reasons and confidence from text using LLM.

        Args:
            client: OpenAI client
            text: Text to extract opinions from
            model: LLM model to use

        Returns:
            List of dicts with keys: 'text' (opinion with reasons), 'confidence' (score 0-1)
        """
        class Opinion(BaseModel):
            """An opinion formed by the agent."""
            opinion: str = Field(description="The opinion or perspective formed")
            reasons: str = Field(description="The reasons supporting this opinion")
            confidence: float = Field(description="Confidence score for this opinion (0.0 to 1.0, where 1.0 is very confident)")

        class OpinionExtractionResponse(BaseModel):
            """Response containing extracted opinions."""
            opinions: List[Opinion] = Field(
                default_factory=list,
                description="List of opinions formed with their supporting reasons and confidence scores"
            )

        extraction_prompt = f"""Extract any opinions or perspectives that were formed in the following text.
An opinion is a judgment, viewpoint, or conclusion that goes beyond just stating facts.

TEXT:
{text}

For each opinion found, provide:
1. The opinion itself
2. The reasons or facts that support it
3. A confidence score (0.0 to 1.0) indicating how confident the agent is in this opinion based on the available information

If no clear opinions are expressed, return an empty list."""

        try:
            response = await client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": "You extract opinions and perspectives from text."},
                    {"role": "user", "content": extraction_prompt}
                ],
                response_format=OpinionExtractionResponse
            )

            result = response.choices[0].message.parsed

            # Format opinions with reasons included in the text and confidence score
            formatted_opinions = []
            for op in result.opinions:
                # Combine opinion and reasons into a single statement
                opinion_with_reasons = f"{op.opinion} (Reasons: {op.reasons})"
                formatted_opinions.append({
                    "text": opinion_with_reasons,
                    "confidence": op.confidence
                })

            return formatted_opinions

        except Exception as e:
            logger.warning(f"Failed to extract opinions: {str(e)}")
            return []
