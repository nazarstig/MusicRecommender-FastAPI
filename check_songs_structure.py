from app.core.database import engine
from sqlalchemy import text

with engine.connect() as conn:
    result = conn.execute(text(
        "SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE "
        "FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'Songs' "
        "ORDER BY ORDINAL_POSITION"
    ))
    print("Songs table columns:")
    for row in result:
        print(f"  {row[0]}: {row[1]}({row[2] if row[2] else ''}) {'NULL' if row[3] == 'YES' else 'NOT NULL'}")
