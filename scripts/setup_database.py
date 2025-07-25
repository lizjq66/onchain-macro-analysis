#!/usr/bin/env python3
from src.data.storage.database import create_tables
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    logger.info("Setting up database...")
    create_tables()
    logger.info("Database setup complete!")

if __name__ == "__main__":
    main()
