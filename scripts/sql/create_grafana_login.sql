-- Create Grafana SQL Login for ACM Database Access
-- ===================================================
-- Run this script in SQL Server Management Studio (SSMS)
-- or via sqlcmd to create a read-only login for Grafana.
--
-- Usage:
--   sqlcmd -S "localhost\B19CL3PCQLSERVER" -E -i scripts/sql/create_grafana_login.sql
--
-- After running, configure Grafana datasource with:
--   User: grafana_reader
--   Password: ACMGrafana2024!
--   Server: host.docker.internal:1433 (from Docker)
--   Database: ACM

USE master;
GO

-- Drop login if exists (for clean reinstall)
IF EXISTS (SELECT * FROM sys.server_principals WHERE name = 'grafana_reader')
BEGIN
    DROP LOGIN grafana_reader;
    PRINT 'Dropped existing login: grafana_reader';
END
GO

-- Create login with secure password
CREATE LOGIN grafana_reader WITH PASSWORD = 'ACMGrafana2024!';
PRINT 'Created login: grafana_reader';
GO

-- Switch to ACM database
USE ACM;
GO

-- Drop user if exists
IF EXISTS (SELECT * FROM sys.database_principals WHERE name = 'grafana_reader')
BEGIN
    DROP USER grafana_reader;
    PRINT 'Dropped existing user: grafana_reader';
END
GO

-- Create user for login
CREATE USER grafana_reader FOR LOGIN grafana_reader;
PRINT 'Created user: grafana_reader in ACM database';
GO

-- Grant read-only access
ALTER ROLE db_datareader ADD MEMBER grafana_reader;
PRINT 'Granted db_datareader role to grafana_reader';
GO

-- Verify setup
SELECT 
    'Login created' AS Status,
    'grafana_reader' AS Username,
    'ACMGrafana2024!' AS Password,
    'host.docker.internal:1433' AS ServerFromDocker,
    'ACM' AS [Database];
GO

PRINT '';
PRINT '================================================================';
PRINT 'Grafana SQL login setup complete!';
PRINT '';
PRINT 'Configure Grafana datasource with:';
PRINT '  Host: host.docker.internal:1433';
PRINT '  Database: ACM';
PRINT '  User: grafana_reader';
PRINT '  Password: ACMGrafana2024!';
PRINT '  Encrypt: disable (for local development)';
PRINT '================================================================';
GO
