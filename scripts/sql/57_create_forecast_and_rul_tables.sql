/*
 * Script: 57_create_forecast_and_rul_tables.sql
 * Purpose: SQL tables for health forecasting, failure probability, maintenance windows, and RUL.
 * Context: Backed by AR(1) forecast logic in core/forecast.py and RUL estimator in core/rul_estimator.py.
 */

USE ACM;
GO

-- Health forecast over time (equipment-level)
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_HealthForecast_TS' AND schema_id = SCHEMA_ID('dbo'))
BEGIN
    CREATE TABLE dbo.ACM_HealthForecast_TS (
        RunID           UNIQUEIDENTIFIER NOT NULL,
        EquipID         INT NOT NULL,
        Timestamp       DATETIME2 NOT NULL,
        ForecastHealth  FLOAT NULL,
        CiLower         FLOAT NULL,
        CiUpper         FLOAT NULL,
        ForecastStd     FLOAT NULL,
        Method          NVARCHAR(50) NOT NULL,
        CreatedAt       DATETIME2 NOT NULL CONSTRAINT DF_ACM_HealthForecast_TS_CreatedAt DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT PK_ACM_HealthForecast_TS PRIMARY KEY CLUSTERED (RunID, EquipID, Timestamp)
    );
END
GO

-- Failure probability over forecast horizon
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_FailureForecast_TS' AND schema_id = SCHEMA_ID('dbo'))
BEGIN
    CREATE TABLE dbo.ACM_FailureForecast_TS (
        RunID           UNIQUEIDENTIFIER NOT NULL,
        EquipID         INT NOT NULL,
        Timestamp       DATETIME2 NOT NULL,
        FailureProb     FLOAT NOT NULL,
        ThresholdUsed   FLOAT NOT NULL,
        Method          NVARCHAR(50) NOT NULL,
        CreatedAt       DATETIME2 NOT NULL CONSTRAINT DF_ACM_FailureForecast_TS_CreatedAt DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT PK_ACM_FailureForecast_TS PRIMARY KEY CLUSTERED (RunID, EquipID, Timestamp)
    );
END
GO

-- Remaining Useful Life time series (per forecast run)
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_RUL_TS' AND schema_id = SCHEMA_ID('dbo'))
BEGIN
    CREATE TABLE dbo.ACM_RUL_TS (
        RunID           UNIQUEIDENTIFIER NOT NULL,
        EquipID         INT NOT NULL,
        Timestamp       DATETIME2 NOT NULL,
        RUL_Hours       FLOAT NOT NULL,
        LowerBound      FLOAT NULL,
        UpperBound      FLOAT NULL,
        Confidence      FLOAT NULL,
        Method          NVARCHAR(50) NOT NULL,
        CreatedAt       DATETIME2 NOT NULL CONSTRAINT DF_ACM_RUL_TS_CreatedAt DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT PK_ACM_RUL_TS PRIMARY KEY CLUSTERED (RunID, EquipID, Timestamp)
    );
END
GO

-- RUL summary (one row per run/equipment)
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_RUL_Summary' AND schema_id = SCHEMA_ID('dbo'))
BEGIN
    CREATE TABLE dbo.ACM_RUL_Summary (
        RunID           UNIQUEIDENTIFIER NOT NULL,
        EquipID         INT NOT NULL,
        RUL_Hours       FLOAT NOT NULL,
        LowerBound      FLOAT NULL,
        UpperBound      FLOAT NULL,
        Confidence      FLOAT NULL,
        Method          NVARCHAR(50) NOT NULL,
        LastUpdate      DATETIME2 NOT NULL,
        CreatedAt       DATETIME2 NOT NULL CONSTRAINT DF_ACM_RUL_Summary_CreatedAt DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT PK_ACM_RUL_Summary PRIMARY KEY CLUSTERED (RunID, EquipID)
    );
END
GO

-- Sensor-level attribution at predicted failure time
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_RUL_Attribution' AND schema_id = SCHEMA_ID('dbo'))
BEGIN
    CREATE TABLE dbo.ACM_RUL_Attribution (
        RunID               UNIQUEIDENTIFIER NOT NULL,
        EquipID             INT NOT NULL,
        FailureTime         DATETIME2 NOT NULL,
        SensorName          NVARCHAR(255) NOT NULL,
        FailureContribution FLOAT NOT NULL,
        ZScoreAtFailure     FLOAT NULL,
        AlertCount          INT NULL,
        Comment             NVARCHAR(400) NULL,
        CreatedAt           DATETIME2 NOT NULL CONSTRAINT DF_ACM_RUL_Attribution_CreatedAt DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT PK_ACM_RUL_Attribution PRIMARY KEY CLUSTERED (RunID, EquipID, FailureTime, SensorName)
    );
END
GO

-- Optional: simple per-sensor forecast (for selected hot sensors)
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_SensorForecast_TS' AND schema_id = SCHEMA_ID('dbo'))
BEGIN
    CREATE TABLE dbo.ACM_SensorForecast_TS (
        RunID           UNIQUEIDENTIFIER NOT NULL,
        EquipID         INT NOT NULL,
        SensorName      NVARCHAR(255) NOT NULL,
        Timestamp       DATETIME2 NOT NULL,
        ForecastValue   FLOAT NOT NULL,
        CiLower         FLOAT NULL,
        CiUpper         FLOAT NULL,
        ForecastStd     FLOAT NULL,
        Method          NVARCHAR(50) NOT NULL,
        CreatedAt       DATETIME2 NOT NULL CONSTRAINT DF_ACM_SensorForecast_TS_CreatedAt DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT PK_ACM_SensorForecast_TS PRIMARY KEY CLUSTERED (RunID, EquipID, SensorName, Timestamp)
    );
END
GO

-- Maintenance recommendation window
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_MaintenanceRecommendation' AND schema_id = SCHEMA_ID('dbo'))
BEGIN
    CREATE TABLE dbo.ACM_MaintenanceRecommendation (
        RunID                 UNIQUEIDENTIFIER NOT NULL,
        EquipID               INT NOT NULL,
        EarliestMaintenance   DATETIME2 NOT NULL,
        PreferredWindowStart  DATETIME2 NOT NULL,
        PreferredWindowEnd    DATETIME2 NOT NULL,
        FailureProbAtWindowEnd FLOAT NOT NULL,
        Comment               NVARCHAR(400) NULL,
        CreatedAt             DATETIME2 NOT NULL CONSTRAINT DF_ACM_MaintenanceRecommendation_CreatedAt DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT PK_ACM_MaintenanceRecommendation PRIMARY KEY CLUSTERED (RunID, EquipID)
    );
END
GO

PRINT 'Forecast and RUL tables created/ensured successfully.';
GO

