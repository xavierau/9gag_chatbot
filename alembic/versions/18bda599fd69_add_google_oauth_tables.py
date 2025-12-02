"""add google oauth tables

Revision ID: 18bda599fd69
Revises: 4e357e1e07e4
Create Date: 2025-12-02

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '18bda599fd69'
down_revision: Union[str, None] = '4e357e1e07e4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create google_oauth_tokens table
    op.create_table('google_oauth_tokens',
        sa.Column('id', sa.String(length=50), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('access_token_encrypted', sa.Text(), nullable=False),
        sa.Column('refresh_token_encrypted', sa.Text(), nullable=False),
        sa.Column('token_expiry', postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('scopes', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('google_email', sa.String(length=255), nullable=True),
        sa.Column('google_name', sa.String(length=255), nullable=True),
        sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('updated_at', postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('is_revoked', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('revoked_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_google_oauth_tokens_user_id'), 'google_oauth_tokens', ['user_id'], unique=True)

    # Create oauth_states table
    op.create_table('oauth_states',
        sa.Column('id', sa.String(length=50), nullable=False),
        sa.Column('state', sa.String(length=64), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('whatsapp_phone', sa.String(length=20), nullable=False),
        sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('expires_at', postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_oauth_states_state'), 'oauth_states', ['state'], unique=True)


def downgrade() -> None:
    op.drop_index(op.f('ix_oauth_states_state'), table_name='oauth_states')
    op.drop_table('oauth_states')
    op.drop_index(op.f('ix_google_oauth_tokens_user_id'), table_name='google_oauth_tokens')
    op.drop_table('google_oauth_tokens')
