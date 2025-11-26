"""add_expenses_tables

Revision ID: 2f913cd689fc
Revises: 64172141a0fc
Create Date: 2025-11-26 18:10:35.203064

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '2f913cd689fc'
down_revision: Union[str, None] = '64172141a0fc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create expense tracking tables."""
    # Create expense_categories table
    op.create_table('expense_categories',
        sa.Column('id', sa.String(length=50), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_expense_categories_user_id'), 'expense_categories', ['user_id'], unique=False)
    op.create_index('ix_expense_categories_user_id_name', 'expense_categories', ['user_id', 'name'], unique=True)

    # Create expenses table
    op.create_table('expenses',
        sa.Column('id', sa.String(length=50), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('amount', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('currency', sa.String(length=3), nullable=False, server_default='HKD'),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('category_name', sa.String(length=100), nullable=False),
        sa.Column('expense_date', sa.Date(), nullable=False),
        sa.Column('merchant_name', sa.String(length=255), nullable=True),
        sa.Column('payment_method', sa.String(length=50), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', postgresql.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_expenses_expense_date'), 'expenses', ['expense_date'], unique=False)
    op.create_index(op.f('ix_expenses_user_id'), 'expenses', ['user_id'], unique=False)
    op.create_index('ix_expenses_user_id_category_name', 'expenses', ['user_id', 'category_name'], unique=False)
    op.create_index('ix_expenses_user_id_expense_date', 'expenses', ['user_id', 'expense_date'], unique=False)


def downgrade() -> None:
    """Remove expense tracking tables."""
    op.drop_index('ix_expenses_user_id_expense_date', table_name='expenses')
    op.drop_index('ix_expenses_user_id_category_name', table_name='expenses')
    op.drop_index(op.f('ix_expenses_user_id'), table_name='expenses')
    op.drop_index(op.f('ix_expenses_expense_date'), table_name='expenses')
    op.drop_table('expenses')
    op.drop_index('ix_expense_categories_user_id_name', table_name='expense_categories')
    op.drop_index(op.f('ix_expense_categories_user_id'), table_name='expense_categories')
    op.drop_table('expense_categories')
