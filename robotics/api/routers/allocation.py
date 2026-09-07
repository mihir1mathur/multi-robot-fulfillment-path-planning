"""Task-allocation endpoint.

    POST /allocation/run   assign pending tasks to available robots with the
                           greedy baseline or the CP-SAT optimiser

An allocation that assigns nothing because no pairing is feasible is a valid
result (200, empty ``assignments``, every task in ``unassigned_task_ids``),
not an error.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from robotics.api.dependencies import get_session, require_role
from robotics.api.security import Role
from robotics.api.schemas.allocation import AllocationRequest, AllocationResponse
from robotics.services.allocation_service import AllocationService

router = APIRouter(
    prefix="/allocation", tags=["allocation"],
    dependencies=[Depends(require_role(Role.operator))],
)


@router.post("/run", response_model=AllocationResponse)
def run_allocation(
    body: AllocationRequest, session: Session = Depends(get_session)
) -> AllocationResponse:
    outcome = AllocationService(session).run_allocation(
        algorithm=body.algorithm.value,
        robots=[r.to_spec() for r in body.robots] or None,
        tasks=[t.to_spec() for t in body.tasks] or None,
        from_persisted=body.from_persisted,
        commit=body.commit,
    )
    return AllocationResponse.from_outcome(outcome)
