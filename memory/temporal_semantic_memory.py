"""
Temporal + Semantic + Entity Memory System for AI Agents.

This implements a sophisticated memory architecture that combines:
1. Temporal links: Memories connected by time proximity
2. Semantic links: Memories connected by meaning/similarity
3. Entity links: Memories connected by shared entities (PERSON, ORG, etc.)
4. Spreading activation: Search through the graph with activation decay
5. Dynamic weighting: Recency and frequency-based importance
"""
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import asyncpg
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import asyncio
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from .utils import (
    extract_facts,
    calculate_recency_weight,
    calculate_frequency_weight,
)
from .entity_resolver import EntityResolver


def utcnow():
    """Get current UTC time with timezone info."""
    return datetime.now(timezone.utc)


# Global process pool for parallel embedding generation
# Each process loads its own copy of the embedding model
# This provides TRUE parallelism for CPU-bound embedding operations
_PROCESS_POOL = None
_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Process-local model cache (one per worker process)
_worker_model = None


def _get_worker_model():
    """Get or load the embedding model in worker process."""
    global _worker_model
    if _worker_model is None:
        _worker_model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    return _worker_model


def _encode_batch_worker(texts: List[str]) -> List[List[float]]:
    """
    Worker function for process pool - encodes texts to embeddings.

    This function runs in a separate process and loads its own model.
    """
    model = _get_worker_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [emb.tolist() for emb in embeddings]


def _get_process_pool():
    """Get or create the global process pool."""
    global _PROCESS_POOL
    if _PROCESS_POOL is None:
        # Use 4 worker processes for true parallelism
        # Adjust based on your CPU cores (each process loads ~500MB model)
        _PROCESS_POOL = ProcessPoolExecutor(max_workers=4)
    return _PROCESS_POOL


class TemporalSemanticMemory:
    """
    Advanced memory system using temporal and semantic linking with PostgreSQL.
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ):
        """
        Initialize the temporal + semantic memory system.

        Args:
            db_url: PostgreSQL connection URL (postgresql://user:pass@host:port/dbname)
            embedding_model: Name of the SentenceTransformer model to use
        """
        load_dotenv()

        # Initialize PostgreSQL connection URL
        self.db_url = db_url or os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError(
                "Database URL not found. "
                "Set DATABASE_URL environment variable."
            )

        # Connection pool (created lazily on first use)
        self._pool = None
        self._pool_lock = asyncio.Lock()

        # Initialize entity resolver (will be created with pool)
        self.entity_resolver = None

        # Initialize local embedding model (384 dimensions)
        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"✓ Model loaded (embedding dim: {self.embedding_model.get_sentence_embedding_dimension()})")

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create the connection pool (lazy initialization)."""
        if self._pool is None:
            async with self._pool_lock:
                if self._pool is None:
                    self._pool = await asyncpg.create_pool(
                        self.db_url,
                        min_size=2,
                        max_size=10,
                        command_timeout=60,
                        statement_cache_size=0  # Disable prepared statement cache
                    )
                    # Initialize entity resolver with pool
                    if self.entity_resolver is None:
                        self.entity_resolver = EntityResolver(self._pool)
        return self._pool

    async def close(self):
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using local SentenceTransformer model.

        Args:
            text: Text to embed

        Returns:
            384-dimensional embedding vector (bge-small-en-v1.5)
        """
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            return embedding.tolist()
        except Exception as e:
            raise Exception(f"Failed to generate embedding: {str(e)}")

    async def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using local model in parallel.

        Uses a ProcessPoolExecutor to achieve TRUE parallelism for CPU-bound
        embedding generation. Each worker process loads its own model copy.

        When multiple put_async calls run in parallel, each can generate
        embeddings concurrently in separate processes (no GIL contention).

        Args:
            texts: List of texts to embed

        Returns:
            List of 384-dimensional embeddings in same order as input texts
        """
        try:
            # Run in process pool for true parallelism
            loop = asyncio.get_event_loop()
            pool = _get_process_pool()
            embeddings = await loop.run_in_executor(
                pool,
                _encode_batch_worker,
                texts
            )
            return embeddings
        except Exception as e:
            raise Exception(f"Failed to generate batch embeddings: {str(e)}")

    async def _find_duplicate_facts_batch(
        self,
        conn,
        agent_id: str,
        texts: List[str],
        embeddings: List[List[float]],
        event_date: datetime,
        time_window_hours: int = 24,
        similarity_threshold: float = 0.95
    ) -> List[bool]:
        """
        Check which facts are duplicates using semantic similarity + temporal window.

        For each new fact, checks if a semantically similar fact already exists
        within the time window. Uses pgvector cosine similarity for efficiency.

        Args:
            conn: Database connection
            agent_id: Agent identifier
            texts: List of fact texts to check
            embeddings: Corresponding embeddings
            event_date: Event date for temporal filtering
            time_window_hours: Hours before/after event_date to search (default: 24)
            similarity_threshold: Minimum cosine similarity to consider duplicate (default: 0.95)

        Returns:
            List of booleans - True if fact is a duplicate (should skip), False if new
        """
        is_duplicate = []

        time_lower = event_date - timedelta(hours=time_window_hours)
        time_upper = event_date + timedelta(hours=time_window_hours)

        for text, embedding in zip(texts, embeddings):
            # Query for similar facts within time window
            # Convert embedding list to string for asyncpg vector type
            embedding_str = str(embedding)
            result = await conn.fetchrow(
                """
                SELECT id, text, 1 - (embedding <=> $1::vector) AS similarity
                FROM memory_units
                WHERE agent_id = $2
                  AND event_date BETWEEN $3 AND $4
                  AND 1 - (embedding <=> $1::vector) > $5
                ORDER BY similarity DESC
                LIMIT 1
                """,
                embedding_str, agent_id, time_lower, time_upper, similarity_threshold
            )

            if result:
                is_duplicate.append(True)
            else:
                is_duplicate.append(False)

        return is_duplicate

    def put(
        self,
        agent_id: str,
        content: str,
        context: str = "",
        event_date: Optional[datetime] = None,
    ) -> List[str]:
        """
        Store content as memory units (synchronous wrapper).

        This is a synchronous wrapper around put_async() for convenience.
        For best performance, use put_async() directly.

        Args:
            agent_id: Unique identifier for the agent
            content: Text content to store
            context: Context about when/why this memory was formed
            event_date: When the event occurred (defaults to now)

        Returns:
            List of created unit IDs
        """
        # Run async version synchronously
        return asyncio.run(self.put_async(agent_id, content, context, event_date))

    async def put_async(
        self,
        agent_id: str,
        content: str,
        context: str = "",
        event_date: Optional[datetime] = None,
    ) -> List[str]:
        """
        Store content as memory units with temporal and semantic links (ASYNC version).

        This is a convenience wrapper around put_batch_async for a single content item.

        Args:
            agent_id: Unique identifier for the agent
            content: Text content to store
            context: Context about when/why this memory was formed
            event_date: When the event occurred (defaults to now)

        Returns:
            List of created unit IDs
        """
        # Use put_batch_async with a single item (avoids code duplication)
        result = await self.put_batch_async(
            agent_id=agent_id,
            contents=[{
                "content": content,
                "context": context,
                "event_date": event_date
            }]
        )

        # Return the first (and only) list of unit IDs
        return result[0] if result else []

    async def put_batch_async(
        self,
        agent_id: str,
        contents: List[Dict[str, Any]],
    ) -> List[List[str]]:
        """
        Store multiple content items as memory units in ONE batch operation.

        This is MUCH more efficient than calling put_async multiple times:
        - Extracts facts from all contents in parallel
        - Generates ALL embeddings in ONE batch
        - Does ALL database operations in ONE transaction

        Args:
            agent_id: Unique identifier for the agent
            contents: List of dicts with keys:
                - "content" (required): Text content to store
                - "context" (optional): Context about the memory
                - "event_date" (optional): When the event occurred

        Returns:
            List of lists of unit IDs (one list per content item)

        Example:
            unit_ids = await memory.put_batch_async(
                agent_id="user123",
                contents=[
                    {"content": "Alice works at Google", "context": "conversation"},
                    {"content": "Bob loves Python", "context": "conversation"},
                ]
            )
            # Returns: [["unit-id-1"], ["unit-id-2"]]
        """
        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"PUT_BATCH_ASYNC START: {agent_id}")
        print(f"Batch size: {len(contents)} content items")
        print(f"{'='*60}")

        if not contents:
            return []

        # Step 1: Extract facts from ALL contents in parallel
        step_start = time.time()

        # Create tasks for parallel fact extraction
        fact_extraction_tasks = []
        for item in contents:
            content = item["content"]
            context = item.get("context", "")
            event_date = item.get("event_date") or utcnow()

            task = extract_facts(content, event_date, context)
            fact_extraction_tasks.append((task, event_date, context))

        # Wait for all fact extractions to complete
        all_fact_results = await asyncio.gather(*[task for task, _, _ in fact_extraction_tasks])

        # Flatten and track which facts belong to which content
        all_fact_texts = []
        all_fact_dates = []
        all_contexts = []
        content_boundaries = []  # [(start_idx, end_idx), ...]

        current_idx = 0
        for i, ((_, event_date, context), fact_dicts) in enumerate(zip(fact_extraction_tasks, all_fact_results)):
            start_idx = current_idx

            for fact_dict in fact_dicts:
                all_fact_texts.append(fact_dict['fact'])
                try:
                    from dateutil import parser as date_parser
                    fact_date = date_parser.isoparse(fact_dict['date'])
                    all_fact_dates.append(fact_date)
                except Exception:
                    all_fact_dates.append(event_date)
                all_contexts.append(context)

            end_idx = current_idx + len(fact_dicts)
            content_boundaries.append((start_idx, end_idx))
            current_idx = end_idx

        total_facts = len(all_fact_texts)

        if total_facts == 0:
            return [[] for _ in contents]

        # Step 2: Generate ALL embeddings in ONE batch (HUGE speedup!)
        step_start = time.time()
        all_embeddings = await self._generate_embeddings_batch(all_fact_texts)
        print(f"[2] Generate embeddings (parallel): {len(all_embeddings)} embeddings in {time.time() - step_start:.3f}s")

        # Step 3: Process everything in ONE database transaction
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                try:
                    # Deduplication check for all facts
                    step_start = time.time()
                    all_is_duplicate = []
                    for sentence, embedding, fact_date in zip(all_fact_texts, all_embeddings, all_fact_dates):
                        dup_flags = await self._find_duplicate_facts_batch(
                            conn, agent_id, [sentence], [embedding], fact_date
                        )
                        all_is_duplicate.extend(dup_flags)

                    duplicates_filtered = sum(all_is_duplicate)
                    new_facts = total_facts - duplicates_filtered
                    print(f"[3] Deduplication check: {duplicates_filtered} duplicates filtered, {new_facts} new facts in {time.time() - step_start:.3f}s")

                    # Filter out duplicates
                    filtered_sentences = [s for s, is_dup in zip(all_fact_texts, all_is_duplicate) if not is_dup]
                    filtered_embeddings = [e for e, is_dup in zip(all_embeddings, all_is_duplicate) if not is_dup]
                    filtered_dates = [d for d, is_dup in zip(all_fact_dates, all_is_duplicate) if not is_dup]
                    filtered_contexts = [c for c, is_dup in zip(all_contexts, all_is_duplicate) if not is_dup]

                    if not filtered_sentences:
                        print(f"[PUT_BATCH_ASYNC] All facts were duplicates, returning empty")
                        return [[] for _ in contents]

                    # Batch insert ALL units
                    step_start = time.time()
                    # Convert embeddings to strings for asyncpg vector type
                    filtered_embeddings_str = [str(emb) for emb in filtered_embeddings]
                    results = await conn.fetch(
                        """
                        INSERT INTO memory_units (agent_id, text, context, embedding, event_date, access_count)
                        SELECT * FROM unnest($1::text[], $2::text[], $3::text[], $4::vector[], $5::timestamptz[], $6::integer[])
                        RETURNING id
                        """,
                        [agent_id] * len(filtered_sentences),
                        filtered_sentences,
                        filtered_contexts,
                        filtered_embeddings_str,
                        filtered_dates,
                        [0] * len(filtered_sentences)
                    )

                    created_unit_ids = [str(row['id']) for row in results]
                    print(f"[5] Batch insert units: {len(created_unit_ids)} units in {time.time() - step_start:.3f}s")

                    # Process entities for ALL units
                    step_start = time.time()
                    all_entity_links = await self._extract_entities_batch_optimized(
                        conn, agent_id, created_unit_ids, filtered_sentences, "", filtered_dates
                    )
                    print(f"[6] Extract entities (batched): {time.time() - step_start:.3f}s")

                    # Create temporal links
                    step_start = time.time()
                    await self._create_temporal_links_batch_per_fact(conn, agent_id, created_unit_ids)
                    print(f"[7] Batch create temporal links: {time.time() - step_start:.3f}s")

                    # Create semantic links
                    step_start = time.time()
                    await self._create_semantic_links_batch(conn, agent_id, created_unit_ids, filtered_embeddings)
                    print(f"[8] Batch create semantic links: {time.time() - step_start:.3f}s")

                    # Insert entity links
                    step_start = time.time()
                    if all_entity_links:
                        await self._insert_entity_links_batch(conn, all_entity_links)
                    print(f"[9] Batch insert entity links: {time.time() - step_start:.3f}s")

                    # Transaction auto-commits on success
                    commit_start = time.time()
                    print(f"[10] Commit: {time.time() - commit_start:.3f}s")

                    # Map created unit IDs back to original content items
                    # Account for duplicates when mapping back
                    result_unit_ids = []
                    filtered_idx = 0

                    for start_idx, end_idx in content_boundaries:
                        content_unit_ids = []
                        for i in range(start_idx, end_idx):
                            if not all_is_duplicate[i]:
                                content_unit_ids.append(created_unit_ids[filtered_idx])
                                filtered_idx += 1
                        result_unit_ids.append(content_unit_ids)

                    total_time = time.time() - start_time
                    print(f"\n{'='*60}")
                    print(f"PUT_BATCH_ASYNC COMPLETE: {len(created_unit_ids)} units from {len(contents)} contents in {total_time:.3f}s")
                    print(f"{'='*60}\n")

                    return result_unit_ids

                except Exception as e:
                    # Transaction auto-rolls back on exception
                    import traceback
                    traceback.print_exc()
                    raise Exception(f"Failed to store batch memory: {str(e)}")

    def search(
        self,
        agent_id: str,
        query: str,
        thinking_budget: int = 50,
        top_k: int = 10,
        live_tracer=None,
    ) -> List[Dict[str, Any]]:
        """
        Search memories using spreading activation (synchronous wrapper).

        This is a synchronous wrapper around search_async() for convenience.
        For best performance, use search_async() directly.

        Args:
            agent_id: Agent ID to search for
            query: Search query
            thinking_budget: How many units to explore (computational budget)
            top_k: Number of results to return
            live_tracer: Optional LiveSearchTracer for visualization

        Returns:
            List of memory units with their weights, sorted by relevance
        """
        # Run async version synchronously
        return asyncio.run(self.search_async(agent_id, query, thinking_budget, top_k, live_tracer))

    async def search_async(
        self,
        agent_id: str,
        query: str,
        thinking_budget: int = 50,
        top_k: int = 10,
        live_tracer=None,
    ) -> List[Dict[str, Any]]:
        """
        Search memories using spreading activation (ASYNC version).

        This implements the core SEARCH operation:
        1. Find entry points (most relevant units via vector search)
        2. Spread activation through the graph
        3. Weight results by activation + recency + frequency
        4. Return top results

        Args:
            agent_id: Agent ID to search for
            query: Search query
            thinking_budget: How many units to explore (computational budget)
            top_k: Number of results to return
            live_tracer: Optional LiveSearchTracer for visualization

        Returns:
            List of memory units with their weights, sorted by relevance
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            search_start = time.time()
            print(f"\n[SEARCH] Starting search for query: '{query[:50]}...' (thinking_budget={thinking_budget}, top_k={top_k})")

            try:
                # Step 1: Generate query embedding
                step_start = time.time()
                query_embedding = self._generate_embedding(query)
                print(f"  [1] Generate query embedding: {time.time() - step_start:.3f}s")

                # Step 2: Find entry points
                step_start = time.time()
                # Convert embedding to string for asyncpg
                query_embedding_str = str(query_embedding)
                entry_points = await conn.fetch(
                    """
                    SELECT id, text, context, event_date, access_count,
                           1 - (embedding <=> $1::vector) AS similarity
                    FROM memory_units
                    WHERE agent_id = $2
                      AND embedding IS NOT NULL
                      AND (1 - (embedding <=> $1::vector)) >= 0.5
                    ORDER BY embedding <=> $1::vector
                    LIMIT 3
                    """,
                    query_embedding_str, agent_id
                )

                print(f"  [2] Find entry points: {len(entry_points)} found in {time.time() - step_start:.3f}s")

                if not entry_points:
                    print(f"[SEARCH] Complete: 0 results in {time.time() - search_start:.3f}s")
                    return []

                # Step 3: Spreading activation with budget
                step_start = time.time()
                visited = set()
                results = []
                budget_remaining = thinking_budget
                # Initialize entry points with their actual similarity scores instead of 1.0
                queue = [(dict(unit), unit["similarity"], True) for unit in entry_points]  # (unit, activation, is_entry)

                # Track substep timings
                update_access_time = 0
                calculate_weight_time = 0
                query_neighbors_time = 0
                process_neighbors_time = 0

                # Process nodes in batches for efficient neighbor querying
                BATCH_SIZE = 50
                nodes_to_process = []  # (unit, activation, is_entry_point)

                while queue and budget_remaining > 0:
                    # Collect a batch of nodes to process
                    while queue and len(nodes_to_process) < BATCH_SIZE and budget_remaining > 0:
                        current_unit, activation, is_entry_point = queue.pop(0)
                        unit_id = str(current_unit["id"])

                        if unit_id not in visited:
                            visited.add(unit_id)
                            budget_remaining -= 1
                            nodes_to_process.append((current_unit, activation, is_entry_point))

                    if not nodes_to_process:
                        break

                    # Update access counts for batch
                    substep_start = time.time()
                    node_ids = [str(node[0]["id"]) for node in nodes_to_process]
                    await conn.execute(
                        "UPDATE memory_units SET access_count = access_count + 1 WHERE id::text = ANY($1)",
                        node_ids
                    )
                    update_access_time += time.time() - substep_start

                    # Query neighbors for ALL nodes in batch at once
                    substep_start = time.time()
                    all_neighbors = await conn.fetch(
                        """
                        SELECT ml.from_unit_id, ml.to_unit_id, ml.weight,
                               mu.text, mu.context, mu.event_date, mu.access_count
                        FROM memory_links ml
                        JOIN memory_units mu ON ml.to_unit_id = mu.id
                        WHERE ml.from_unit_id::text = ANY($1)
                          AND ml.weight >= 0.1
                        ORDER BY ml.from_unit_id, ml.weight DESC
                        """,
                        node_ids
                    )
                    query_neighbors_time += time.time() - substep_start

                    # Group neighbors by from_unit_id
                    substep_start = time.time()
                    neighbors_by_node = {}
                    for neighbor in all_neighbors:
                        from_id = str(neighbor["from_unit_id"])
                        if from_id not in neighbors_by_node:
                            neighbors_by_node[from_id] = []
                        neighbors_by_node[from_id].append(neighbor)

                    # Process each node in the batch
                    for current_unit, activation, is_entry_point in nodes_to_process:
                        unit_id = str(current_unit["id"])

                        # Calculate combined weight
                        event_date = current_unit["event_date"]
                        days_since = (utcnow() - event_date).total_seconds() / 86400

                        recency_weight = calculate_recency_weight(days_since)
                        frequency_weight = calculate_frequency_weight(current_unit.get("access_count", 0))

                        # Normalize frequency to [0, 1] range
                        frequency_normalized = (frequency_weight - 1.0) / 1.0

                        # Calculate semantic similarity between query and this memory
                        memory_embedding = current_unit.get("embedding")
                        if memory_embedding is not None:
                            # Cosine similarity = 1 - cosine distance
                            query_vec = np.array(query_embedding)
                            memory_vec = np.array(memory_embedding)
                            # Cosine similarity
                            dot_product = np.dot(query_vec, memory_vec)
                            norm_query = np.linalg.norm(query_vec)
                            norm_memory = np.linalg.norm(memory_vec)
                            semantic_similarity = dot_product / (norm_query * norm_memory) if norm_query > 0 and norm_memory > 0 else 0.0
                        else:
                            semantic_similarity = 0.0

                        # Combined weight: 30% activation, 30% semantic similarity, 25% recency, 15% frequency
                        final_weight = 0.3 * activation + 0.3 * semantic_similarity + 0.25 * recency_weight + 0.15 * frequency_normalized

                        # Notify tracer
                        if live_tracer:
                            live_tracer.visit_node(
                                node_id=unit_id,
                                text=current_unit["text"],
                                activation=activation,
                                recency=recency_weight,
                                frequency=frequency_weight,
                                weight=final_weight,
                                is_entry_point=is_entry_point,
                            )

                        results.append({
                            "id": unit_id,
                            "text": current_unit["text"],
                            "context": current_unit.get("context", ""),
                            "event_date": event_date.isoformat(),
                            "weight": final_weight,
                            "activation": activation,
                            "semantic_similarity": semantic_similarity,
                            "recency": recency_weight,
                            "frequency": frequency_weight,
                        })

                        # Spread to neighbors (from batch query results)
                        neighbors = neighbors_by_node.get(unit_id, [])
                        for neighbor in neighbors:
                            neighbor_id = str(neighbor["to_unit_id"])
                            if neighbor_id not in visited:
                                link_weight = neighbor["weight"]
                                new_activation = activation * link_weight * 0.8  # 0.8 = decay factor

                                if new_activation > 0.1:
                                    queue.append(({
                                        "id": neighbor["to_unit_id"],
                                        "text": neighbor["text"],
                                        "context": neighbor.get("context", ""),
                                        "event_date": neighbor["event_date"],
                                        "access_count": neighbor["access_count"],
                                        "embedding": neighbor.get("embedding"),
                                    }, new_activation, False))  # Not an entry point

                    calculate_weight_time += time.time() - substep_start
                    process_neighbors_time += time.time() - substep_start

                    # Clear batch for next iteration
                    nodes_to_process = []

                    spreading_activation_time = time.time() - step_start
                    num_batches = (len(visited) + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
                    print(f"  [3] Spreading activation: {len(visited)} nodes visited in {spreading_activation_time:.3f}s")
                    print(f"      [3.1] Update access counts: {update_access_time:.3f}s")
                    print(f"      [3.2] Calculate weights: {calculate_weight_time:.3f}s")
                    print(f"      [3.3] Query neighbors: {query_neighbors_time:.3f}s ({num_batches} batched queries)")
                    print(f"      [3.4] Process neighbors: {process_neighbors_time:.3f}s")

                    # Step 4: Sort by final weight and return top results
                    step_start = time.time()
                    results.sort(key=lambda x: x["weight"], reverse=True)
                    top_results = results[:top_k]
                    print(f"  [4] Sort and return top {top_k}: {time.time() - step_start:.3f}s")

                    print(f"[SEARCH] Complete: {len(top_results)} results in {time.time() - search_start:.3f}s\n")
                    return top_results

            except Exception as e:
                print(f"[SEARCH] ERROR after {time.time() - search_start:.3f}s: {str(e)}")
                raise Exception(f"Failed to search memories: {str(e)}")

    async def delete_agent(self, agent_id: str) -> Dict[str, int]:
        """
        Delete all data for a specific agent (multi-tenant cleanup).

        This is much more efficient than dropping all tables and allows
        multiple agents to coexist in the same database.

        Deletes (with CASCADE):
        - All memory units for this agent
        - All entities for this agent
        - All associated links, unit-entity associations, and co-occurrences

        Args:
            agent_id: Agent ID to delete

        Returns:
            Dictionary with counts of deleted items
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                try:
                    # Count before deletion for reporting
                    units_count = await conn.fetchval("SELECT COUNT(*) FROM memory_units WHERE agent_id = $1", agent_id)
                    entities_count = await conn.fetchval("SELECT COUNT(*) FROM entities WHERE agent_id = $1", agent_id)

                    # Delete memory units (cascades to unit_entities, memory_links)
                    await conn.execute("DELETE FROM memory_units WHERE agent_id = $1", agent_id)

                    # Delete entities (cascades to unit_entities, entity_cooccurrences, memory_links with entity_id)
                    await conn.execute("DELETE FROM entities WHERE agent_id = $1", agent_id)

                    return {
                        "memory_units_deleted": units_count,
                        "entities_deleted": entities_count
                    }

                except Exception as e:
                    raise Exception(f"Failed to delete agent data: {str(e)}")

    async def _extract_entities_batch_optimized(
        self,
        conn,
        agent_id: str,
        unit_ids: List[str],
        sentences: List[str],
        context: str,
        fact_dates: List,
    ) -> List[tuple]:
        """
        Extract entities from ALL sentences in one batch (MUCH faster than sequential).

        Uses spaCy's batch processing to extract entities from all texts at once,
        then resolves and links them in bulk.

        Returns list of tuples for batch insertion: (from_unit_id, to_unit_id, link_type, weight, entity_id)
        """
        from .entity_resolver import extract_entities_batch

        try:
            # Step 1: Extract entities from ALL sentences in one batch (fast!)
            substep_start = time.time()
            all_entities = extract_entities_batch(sentences)
            total_entities = sum(len(ents) for ents in all_entities)
            print(f"  [6.1] spaCy NER (batch): {total_entities} entities from {len(sentences)} sentences in {time.time() - substep_start:.3f}s")

            # Step 2: Resolve entities in BATCH (much faster!)
            substep_start = time.time()
            step_6_2_start = time.time()

            # [6.2.1] Prepare all entities for batch resolution
            substep_6_2_1_start = time.time()
            all_entities_flat = []
            entity_to_unit = []  # Maps flat index to (unit_id, local_index)

            for unit_id, entities, fact_date in zip(unit_ids, all_entities, fact_dates):
                if not entities:
                    continue

                for local_idx, entity in enumerate(entities):
                    all_entities_flat.append({
                        'text': entity['text'],
                        'type': entity['type'],
                        'nearby_entities': entities,
                    })
                    entity_to_unit.append((unit_id, local_idx, fact_date))
            print(f"    [6.2.1] Prepare entities: {len(all_entities_flat)} entities in {time.time() - substep_6_2_1_start:.3f}s")

            # Resolve ALL entities in one batch call
            if all_entities_flat:
                # [6.2.2] Batch resolve entities
                substep_6_2_2_start = time.time()
                # Group by date for batch resolution (most will have same date)
                entities_by_date = {}
                for idx, (unit_id, local_idx, fact_date) in enumerate(entity_to_unit):
                    date_key = fact_date
                    if date_key not in entities_by_date:
                        entities_by_date[date_key] = []
                    entities_by_date[date_key].append((idx, all_entities_flat[idx]))

                # Resolve each date group in batch
                resolved_entity_ids = [None] * len(all_entities_flat)
                for fact_date, entities_group in entities_by_date.items():
                    indices = [idx for idx, _ in entities_group]
                    entities_data = [entity_data for _, entity_data in entities_group]

                    batch_resolved = await self.entity_resolver.resolve_entities_batch(
                        agent_id=agent_id,
                        entities_data=entities_data,
                        context=context,
                        unit_event_date=fact_date,
                        conn=conn
                    )

                    for idx, entity_id in zip(indices, batch_resolved):
                        resolved_entity_ids[idx] = entity_id
                print(f"    [6.2.2] Resolve entities: {len(all_entities_flat)} entities in {time.time() - substep_6_2_2_start:.3f}s")

                # [6.2.3] Create unit-entity links in BATCH
                substep_6_2_3_start = time.time()
                # Map resolved entities back to units and collect all (unit, entity) pairs
                unit_to_entity_ids = {}
                unit_entity_pairs = []
                for idx, (unit_id, local_idx, fact_date) in enumerate(entity_to_unit):
                    if unit_id not in unit_to_entity_ids:
                        unit_to_entity_ids[unit_id] = []

                    entity_id = resolved_entity_ids[idx]
                    unit_to_entity_ids[unit_id].append(entity_id)
                    unit_entity_pairs.append((unit_id, entity_id))

                # Batch insert all unit-entity links (MUCH faster!)
                await self.entity_resolver.link_units_to_entities_batch(unit_entity_pairs, conn=conn)
                print(f"    [6.2.3] Create unit-entity links (batched): {len(unit_entity_pairs)} links in {time.time() - substep_6_2_3_start:.3f}s")

                print(f"  [6.2] Entity resolution (batched): {len(all_entities_flat)} entities resolved in {time.time() - step_6_2_start:.3f}s")
            else:
                unit_to_entity_ids = {}
                print(f"  [6.2] Entity resolution (batched): 0 entities in {time.time() - step_6_2_start:.3f}s")

            # Step 3: Create entity links between units that share entities
            substep_start = time.time()
            # Collect all unique entity IDs
            all_entity_ids = set()
            for entity_ids in unit_to_entity_ids.values():
                all_entity_ids.update(entity_ids)

            # For each entity, find all units that reference it (one query per entity)
            entity_to_units = {}
            for entity_id in all_entity_ids:
                rows = await conn.fetch(
                    """
                    SELECT unit_id
                    FROM unit_entities
                    WHERE entity_id = $1
                    """,
                    entity_id
                )
                entity_to_units[entity_id] = [row['unit_id'] for row in rows]

            # Create bidirectional links between units that share entities
            links = []
            for entity_id, units_with_entity in entity_to_units.items():
                # For each pair of units with this entity, create bidirectional links
                for i, unit_id_1 in enumerate(units_with_entity):
                    for unit_id_2 in units_with_entity[i+1:]:
                        # Bidirectional links
                        links.append((unit_id_1, unit_id_2, 'entity', 1.0, entity_id))
                        links.append((unit_id_2, unit_id_1, 'entity', 1.0, entity_id))

            print(f"  [6.3] Entity link creation: {len(links)} links for {len(all_entity_ids)} unique entities in {time.time() - substep_start:.3f}s")

            return links

        except Exception as e:
            print(f"ERROR: Failed to extract entities in batch: {str(e)}")
            import traceback
            traceback.print_exc()
            # Re-raise to trigger rollback at put_async level
            raise

    async def _create_temporal_links_batch_per_fact(
        self,
        conn,
        agent_id: str,
        unit_ids: List[str],
        time_window_hours: int = 24,
    ):
        """
        Create temporal links for multiple units, each with their own event_date.

        Queries the event_date for each unit from the database and creates temporal
        links based on individual dates (supports per-fact dating).
        """
        if not unit_ids:
            return

        try:
            # Get the event_date for each new unit
            rows = await conn.fetch(
                """
                SELECT id, event_date
                FROM memory_units
                WHERE id::text = ANY($1)
                """,
                unit_ids
            )
            new_units = {str(row['id']): row['event_date'] for row in rows}

            # Create links based on each unit's individual event_date
            links = []
            for unit_id, unit_event_date in new_units.items():
                # Find units within the time window of THIS specific unit
                recent_units = await conn.fetch(
                    """
                    SELECT id, event_date
                    FROM memory_units
                    WHERE agent_id = $1
                      AND id != $2
                      AND event_date BETWEEN $3 AND $4
                    ORDER BY event_date DESC
                    LIMIT 10
                    """,
                    agent_id,
                    unit_id,
                    unit_event_date - timedelta(hours=time_window_hours),
                    unit_event_date + timedelta(hours=time_window_hours)
                )

                for recent_row in recent_units:
                    recent_id = recent_row['id']
                    recent_event_date = recent_row['event_date']
                    # Calculate temporal proximity weight
                    time_diff_hours = abs((unit_event_date - recent_event_date).total_seconds() / 3600)
                    weight = max(0.3, 1.0 - (time_diff_hours / time_window_hours))
                    links.append((unit_id, str(recent_id), 'temporal', weight, None))

            if links:
                await conn.executemany(
                    """
                    INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, entity_id)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::uuid)) DO NOTHING
                    """,
                    links
                )

        except Exception as e:
            print(f"ERROR: Failed to create temporal links: {str(e)}")
            import traceback
            traceback.print_exc()
            # Re-raise to trigger rollback at put_async level
            raise

    async def _create_semantic_links_batch(
        self,
        conn,
        agent_id: str,
        unit_ids: List[str],
        embeddings: List[List[float]],
        top_k: int = 5,
        threshold: float = 0.7,
    ):
        """
        Create semantic links for multiple units efficiently.

        For each unit, finds similar units and creates links.
        """
        if not unit_ids or not embeddings:
            return

        try:
            all_links = []

            for unit_id, embedding in zip(unit_ids, embeddings):
                # Find similar units using vector similarity
                # Convert embedding to string for asyncpg
                embedding_str = str(embedding)
                similar_units = await conn.fetch(
                    """
                    SELECT id, 1 - (embedding <=> $1::vector) AS similarity
                    FROM memory_units
                    WHERE agent_id = $2
                      AND id != $3
                      AND embedding IS NOT NULL
                      AND (1 - (embedding <=> $1::vector)) >= $4
                    ORDER BY embedding <=> $1::vector
                    LIMIT $5
                    """,
                    embedding_str, agent_id, unit_id, threshold, top_k
                )

                for row in similar_units:
                    similar_id = row['id']
                    similarity = row['similarity']
                    all_links.append((unit_id, str(similar_id), 'semantic', float(similarity), None))

            if all_links:
                await conn.executemany(
                    """
                    INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, entity_id)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::uuid)) DO NOTHING
                    """,
                    all_links
                )

        except Exception as e:
            print(f"ERROR: Failed to create semantic links: {str(e)}")
            import traceback
            traceback.print_exc()
            # Re-raise to trigger rollback at put_async level
            raise

    async def _insert_entity_links_batch(self, conn, links: List[tuple]):
        """Insert all entity links in a single batch."""
        if not links:
            return

        try:
            await conn.executemany(
                """
                INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, entity_id)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::uuid)) DO NOTHING
                """,
                links
            )
        except Exception as e:
            print(f"Warning: Failed to insert entity links: {str(e)}")
