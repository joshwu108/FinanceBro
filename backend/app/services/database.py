from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import os
from typing import Optional

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./financebro.db")

# Create engine
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
else:
    engine = create_engine(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Metadata for migrations
metadata = MetaData()


async def init_db():
    """Initialize database tables"""
    try:
        # Import models to register them
        #from app.routers.stocks import Stock
        #from app.models.alert import Alert
        #from app.models.portfolio import Portfolio
        #from app.models.prediction import Prediction
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DatabaseManager:
    """Database manager for common operations"""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
    
    def get_session(self):
        """Get a new database session"""
        return self.SessionLocal()
    
    def close_session(self, session):
        """Close a database session"""
        session.close()
    
    def execute_query(self, query):
        """Execute a raw SQL query"""
        with self.engine.connect() as connection:
            result = connection.execute(query)
            return result
    
    def health_check(self) -> bool:
        """Check database connectivity"""
        try:
            with self.engine.connect() as connection:
                connection.execute("SELECT 1")
            return True
        except Exception:
            return False


# Global database manager instance
db_manager = DatabaseManager() 