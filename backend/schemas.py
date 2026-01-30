from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional
import uuid


class ImportPreviewRequest(BaseModel):
    raw_text: str


class ImportPart(BaseModel):
    message_id: int
    text_id: int
    speaker_id: str
    speaker_name: str
    contents: str
    conversation_at: Optional[str] = None


class ImportPreviewResponse(BaseModel):
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    split_version: int = 1
    parts: List[ImportPart]


class ImportCommitRequest(BaseModel):
    thread_id: str
    split_version: int = 1
    parts: List[ImportPart]


class SpeakerBase(BaseModel):
    speaker_name: str
    speaker_role: Optional[str] = None
    canonical_role: str = "other"
    speaker_type_detail: Optional[str] = None


class SpeakerCreate(SpeakerBase):
    pass


class SpeakerUpdate(SpeakerBase):
    pass


class Speaker(SpeakerBase):
    speaker_id: str
    created_at: str
    updated_at: str


class UtteranceRoleBase(BaseModel):
    utterance_role_name: str


class UtteranceRoleCreate(UtteranceRoleBase):
    pass


class UtteranceRoleUpdate(UtteranceRoleBase):
    pass


class UtteranceRole(UtteranceRoleBase):
    utterance_role_id: int
    created_at: str
    updated_at: str


class WorkerJob(BaseModel):
    job_id: str
    job_type: str
    target_table: str
    target_id: str
    status: str
    updated_at: str
