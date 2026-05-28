from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from app.models.conversation import CaseSession


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, CaseSession] = {}

    def create(self) -> CaseSession:
        session = CaseSession(case_id=str(uuid4()))
        self._sessions[session.case_id] = session
        return session

    def get(self, case_id: str) -> CaseSession | None:
        return self._sessions.get(case_id)

    def get_or_create(self, case_id: str | None) -> CaseSession:
        if case_id:
            existing = self.get(case_id)
            if existing:
                return existing
        return self.create()

    def save(self, session: CaseSession) -> CaseSession:
        session.updated_at = datetime.utcnow()
        self._sessions[session.case_id] = session
        return session


session_store = InMemorySessionStore()
