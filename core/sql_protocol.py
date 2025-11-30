"""
FOR-CODE-04: SQL Client Protocol for type safety.

Defines a minimal Protocol for SQL client operations used throughout ACM.
This replaces `Any` type hints with proper structural typing, improving
code documentation and enabling static type checking without requiring
concrete base classes.
"""

from typing import Protocol, Any, Optional, Tuple, List, runtime_checkable


@runtime_checkable
class Cursor(Protocol):
    """Protocol for database cursor objects."""
    
    def execute(self, sql: str, params: Optional[Tuple[Any, ...]] = None) -> Any:
        """Execute a SQL query with optional parameters."""
        ...
    
    def fetchall(self) -> List[Any]:
        """Fetch all remaining rows of a query result."""
        ...
    
    def fetchone(self) -> Optional[Tuple[Any, ...]]:
        """Fetch the next row of a query result."""
        ...
    
    def close(self) -> None:
        """Close the cursor."""
        ...


@runtime_checkable
class Connection(Protocol):
    """Protocol for database connection objects."""
    
    autocommit: bool
    
    def commit(self) -> None:
        """Commit the current transaction."""
        ...
    
    def rollback(self) -> None:
        """Rollback the current transaction."""
        ...
    
    def close(self) -> None:
        """Close the connection."""
        ...


@runtime_checkable
class SqlClient(Protocol):
    """
    Protocol defining the SQL client interface used by ACM modules.
    
    This represents the minimal contract required by forecasting, RUL,
    output_manager, and other modules that interact with SQL databases.
    
    Implementations: core.sql_client.SQLClient (pyodbc-based)
    """
    
    conn: Connection
    
    def cursor(self) -> Cursor:
        """
        Create and return a new database cursor.
        
        Returns:
            Cursor object for executing queries
        """
        ...
    
    def execute(self, sql: str, params: Optional[Tuple[Any, ...]] = None) -> Cursor:
        """
        Execute a SQL statement and return a cursor.
        
        Args:
            sql: SQL statement to execute
            params: Optional tuple of parameters for parameterized query
        
        Returns:
            Cursor with query results
        """
        ...
    
    def fetchall(self, sql: str, params: Optional[Tuple[Any, ...]] = None) -> List[Tuple[Any, ...]]:
        """
        Execute query and fetch all results.
        
        Args:
            sql: SQL SELECT statement
            params: Optional query parameters
        
        Returns:
            List of result rows as tuples
        """
        ...
    
    def fetchone(self, sql: str, params: Optional[Tuple[Any, ...]] = None) -> Optional[Tuple[Any, ...]]:
        """
        Execute query and fetch one result.
        
        Args:
            sql: SQL SELECT statement
            params: Optional query parameters
        
        Returns:
            Single result row as tuple, or None if no results
        """
        ...
    
    def insert_many(
        self,
        table: str,
        rows: List[Tuple[Any, ...]],
        columns: Optional[List[str]] = None
    ) -> int:
        """
        Bulk insert rows into a table.
        
        Args:
            table: Target table name
            rows: List of value tuples to insert
            columns: Optional list of column names (if None, assumes all columns)
        
        Returns:
            Number of rows inserted
        """
        ...
    
    def close(self) -> None:
        """Close the database connection."""
        ...
