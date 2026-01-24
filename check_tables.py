from app.core.database import engine
from sqlalchemy import text

with engine.connect() as conn:
    result = conn.execute(text(
        "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
        "WHERE TABLE_SCHEMA = 'dbo' AND TABLE_TYPE = 'BASE TABLE' "
        "ORDER BY TABLE_NAME"
    ))
    tables = [row[0] for row in result]
    print("\n".join(tables))
