from langchain.tools import tool
from qdrant_client.http.models import Filter, FieldCondition, MatchText
from config import client, collection_name, embedding

@tool
def retrieve_grouped_context(query: str):
    """Retrieve candidate CV context for multiple candidates"""
    embedded_query = embedding.embed_query(query)

    response = client.query_points_groups(
        collection_name=collection_name,
        query=embedded_query,
        group_by="metadata.source",
        limit=4,
        group_size=4
    )

    context = []

    for group in response.groups:
        content = [hit.payload["page_content"] for hit in group.hits if hit.score > 0.65]
        metadata = group.hits[0].payload["metadata"]

        if len(content) > 0:
            joined_content = "\n".join(content)
            context.append(
                f"Candidate: {metadata['name']}\n"
                f"Context:\n{joined_content}\n"
                f"------------------"
            )

    return "\n".join(context)

@tool
def retrieve_candidate_context(query: str, candidate_name: str):
    """Retrieve detailed CV context for a specific candidate"""

    embedded_query = embedding.embed_query(query)

    search_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.name",
                match=MatchText(text=candidate_name)
            )
        ]
    )

    results = client.query_points(
        collection_name=collection_name,
        query=embedded_query,
        query_filter=search_filter,
        limit=5
    )

    context = []

    for point in results.points:
        payload = point.payload
        content = payload["page_content"]
        metadata = payload["metadata"]

        context.append(
            f"Candidate: {metadata['name']}\n"
            f"Context:\n{content}\n"
            f"------------------"
        )

    return "\n".join(context)
