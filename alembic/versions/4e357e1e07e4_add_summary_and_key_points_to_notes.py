"""add_summary_and_key_points_to_notes

Revision ID: 4e357e1e07e4
Revises: 9f1f200b0d56
Create Date: 2025-11-27 14:14:22.405278

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '4e357e1e07e4'
down_revision: Union[str, None] = '9f1f200b0d56'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('notes', sa.Column('summary', sa.Text(), nullable=True))
    op.add_column('notes', sa.Column('key_points', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column('notes', 'key_points')
    op.drop_column('notes', 'summary')
