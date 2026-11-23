"""Legal KB provisions (3.4).

GET /v1/legal/provisions/{provision_id}   200 ProvisionResponse; 404 PROVISION_NOT_FOUND for an
unknown id or a split-part id (A-45). Shared Legal KB, so no case scope; legal nodes are
read-only through the API (R-18).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path

from app.agents.nodes.graph_expander import ConceptIndex
from app.deps import get_concepts, get_neo4j
from app.errors import error_responses
from app.ingestion.legal.legal_processor import get_provision
from app.schemas.api import ProvisionResponse
from app.services.retrieval.neo4j_service import Neo4jService

router = APIRouter(prefix="/v1/legal", tags=["legal"], responses=error_responses('PROVISION_NOT_FOUND'))


@router.get("/provisions/{provision_id}", response_model=ProvisionResponse)
async def provision(
    provision_id: str = Path(min_length=1, max_length=200),
    neo4j: Neo4jService = Depends(get_neo4j),
    concepts: ConceptIndex = Depends(get_concepts),
) -> ProvisionResponse:
    return ProvisionResponse(**await get_provision(neo4j, provision_id, concepts.terms_in))
