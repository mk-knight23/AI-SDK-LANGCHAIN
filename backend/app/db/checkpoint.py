"""PostgreSQL checkpointing for LangGraph state persistence.

This module implements state checkpointing using PostgreSQL as the backend.
Checkpoints allow LangGraph agents to resume from any point in execution
and enable time-travel debugging.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import json
import logging

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, DateTime, JSON, Text
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.config import get_config

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


class CheckpointModel(Base):
    """SQLAlchemy model for checkpoint storage."""

    __tablename__ = "checkpoints"

    id: Mapped[int] = mapped_column(primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(255), index=True)
    checkpoint_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    parent_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    state: Mapped[dict] = mapped_column(JSON, nullable=False)
    step: Mapped[int] = mapped_column(Integer, default=0)
    checkpoint_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


@dataclass(frozen=True)
class Checkpoint:
    """Immutable checkpoint data structure.

    Attributes:
        thread_id: Unique identifier for the conversation thread
        checkpoint_id: Unique identifier for this checkpoint
        state: Agent state at this checkpoint
        step: Step number in the execution flow
        parent_id: Optional parent checkpoint ID for version tracking
        metadata: Optional metadata about the checkpoint
        created_at: Timestamp when checkpoint was created
    """

    thread_id: str
    checkpoint_id: str
    state: dict[str, Any]
    step: int
    parent_id: str | None = None
    metadata: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class CheckpointMetadata:
    """Metadata about a checkpoint.

    Attributes:
        thread_id: Thread identifier
        checkpoint_id: Checkpoint identifier
        step: Step number
        source: What created this checkpoint (agent, system, etc.)
        writes: What writes were performed at this checkpoint
    """

    thread_id: str
    checkpoint_id: str
    step: int
    source: str
    writes: dict[str, Any] | None = None


class PostgresCheckpointSaver:
    """PostgreSQL-based checkpoint saver for LangGraph.

    Provides persistent storage for agent state checkpoints, enabling
    resumption and time-travel debugging of agent executions.

    Example:
        >>> saver = PostgresCheckpointSaver()
        >>> checkpoint = Checkpoint(thread_id="abc", checkpoint_id="1", state={...}, step=1)
        >>> await saver.save_checkpoint(checkpoint)
        >>> retrieved = await saver.get_checkpoint("abc", "1")
    """

    def __init__(self, database_url: str | None = None):
        """Initialize the checkpoint saver.

        Args:
            database_url: Optional database URL. If not provided, uses config.
        """
        config = get_config()
        self._database_url = database_url or config.database.url

        # Create async engine
        self._engine = create_async_engine(
            self._database_url,
            pool_size=config.database.pool_size,
            max_overflow=config.database.max_overflow,
            pool_timeout=config.database.pool_timeout,
            echo=config.database.echo,
        )

        # Create session factory
        self._session_maker = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        logger.info(f"PostgresCheckpointSaver initialized with database: {self._database_url}")

    async def create_tables(self) -> None:
        """Create database tables if they don't exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created/verified")

    async def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint to the database.

        Args:
            checkpoint: Checkpoint to save

        Raises:
            Exception: If database operation fails
        """
        async with self._session_maker() as session:
            db_checkpoint = CheckpointModel(
                thread_id=checkpoint.thread_id,
                checkpoint_id=checkpoint.checkpoint_id,
                parent_id=checkpoint.parent_id,
                state=checkpoint.state,
                step=checkpoint.step,
                checkpoint_metadata=checkpoint.metadata,
                created_at=checkpoint.created_at,
            )

            session.add(db_checkpoint)
            await session.commit()

            logger.debug(
                f"Saved checkpoint {checkpoint.checkpoint_id} "
                f"for thread {checkpoint.thread_id}"
            )

    async def get_checkpoint(
        self, thread_id: str, checkpoint_id: str
    ) -> Checkpoint | None:
        """Retrieve a checkpoint from the database.

        Args:
            thread_id: Thread identifier
            checkpoint_id: Checkpoint identifier

        Returns:
            Checkpoint if found, None otherwise
        """
        async with self._session_maker() as session:
            result = await session.execute(
                SELECT(CheckpointModel).where(
                    CheckpointModel.thread_id == thread_id,
                    CheckpointModel.checkpoint_id == checkpoint_id,
                )
            )
            db_checkpoint = result.scalar_one_or_none()

            if db_checkpoint is None:
                return None

            return Checkpoint(
                thread_id=db_checkpoint.thread_id,
                checkpoint_id=db_checkpoint.checkpoint_id,
                parent_id=db_checkpoint.parent_id,
                state=db_checkpoint.state,
                step=db_checkpoint.step,
                metadata=db_checkpoint.checkpoint_metadata,
                created_at=db_checkpoint.created_at,
            )

    async def list_checkpoints(
        self,
        thread_id: str,
        limit: int = 100,
        before: datetime | None = None,
    ) -> list[Checkpoint]:
        """List checkpoints for a thread.

        Args:
            thread_id: Thread identifier
            limit: Maximum number of checkpoints to return
            before: Only return checkpoints before this timestamp

        Returns:
            List of checkpoints, ordered by creation time (newest first)
        """
        async with self._session_maker() as session:
            from sqlalchemy import select, desc

            query = select(CheckpointModel).where(
                CheckpointModel.thread_id == thread_id
            )

            if before is not None:
                query = query.where(CheckpointModel.created_at < before)

            query = query.order_by(desc(CheckpointModel.created_at)).limit(limit)

            result = await session.execute(query)
            db_checkpoints = result.scalars().all()

            return [
                Checkpoint(
                    thread_id=cp.thread_id,
                    checkpoint_id=cp.checkpoint_id,
                    parent_id=cp.parent_id,
                    state=cp.state,
                    step=cp.step,
                    metadata=cp.checkpoint_metadata,
                    created_at=cp.created_at,
                )
                for cp in db_checkpoints
            ]

    async def delete_checkpoint(self, thread_id: str, checkpoint_id: str) -> bool:
        """Delete a checkpoint from the database.

        Args:
            thread_id: Thread identifier
            checkpoint_id: Checkpoint identifier

        Returns:
            True if checkpoint was deleted, False if not found
        """
        async with self._session_maker() as session:
            from sqlalchemy import delete

            result = await session.execute(
                delete(CheckpointModel).where(
                    CheckpointModel.thread_id == thread_id,
                    CheckpointModel.checkpoint_id == checkpoint_id,
                )
            )
            await session.commit()

            deleted = result.rowcount > 0
            if deleted:
                logger.debug(
                    f"Deleted checkpoint {checkpoint_id} for thread {thread_id}"
                )

            return deleted

    async def get_latest_checkpoint(self, thread_id: str) -> Checkpoint | None:
        """Get the latest checkpoint for a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            Latest checkpoint if found, None otherwise
        """
        checkpoints = await self.list_checkpoints(thread_id, limit=1)
        return checkpoints[0] if checkpoints else None

    async def close(self) -> None:
        """Close the database connection."""
        await self._engine.dispose()
        logger.info("PostgresCheckpointSaver closed")


def create_checkpoint_saver(database_url: str | None = None) -> PostgresCheckpointSaver:
    """Factory function to create a checkpoint saver.

    Args:
        database_url: Optional custom database URL

    Returns:
        Configured PostgresCheckpointSaver instance
    """
    return PostgresCheckpointSaver(database_url=database_url)


# Import at end to avoid circular dependencies
from sqlalchemy import select
