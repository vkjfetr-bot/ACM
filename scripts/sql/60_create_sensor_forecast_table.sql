-- =============================================================================
-- ACM_SensorForecast Table - Physical Sensor Value Predictions
-- =============================================================================
-- Purpose: Store forecasted values for physical sensors (Motor Current,
--          Bearing Temperature, Pressure, etc.) to predict equipment behavior
-- Version: v10.0.0
-- =============================================================================

USE ACM;
GO

-- Drop existing table if present (for clean migrations)
IF OBJECT_ID('dbo.ACM_SensorForecast', 'U') IS NOT NULL
    DROP TABLE dbo.ACM_SensorForecast;
GO

CREATE TABLE dbo.ACM_SensorForecast (
    -- Primary identification
    RunID           UNIQUEIDENTIFIER NOT NULL,          -- FK to ACM_Runs
    EquipID         INT NOT NULL,                       -- FK to Equipment
    Timestamp       DATETIME2 NOT NULL,                 -- Forecast timestamp
    SensorName      NVARCHAR(255) NOT NULL,             -- Physical sensor name
    
    -- Forecast values
    ForecastValue   FLOAT NOT NULL,                     -- Predicted sensor value
    CiLower         FLOAT NULL,                         -- Lower confidence interval
    CiUpper         FLOAT NULL,                         -- Upper confidence interval
    ForecastStd     FLOAT NULL,                         -- Standard deviation of forecast
    
    -- Metadata
    Method          NVARCHAR(50) NOT NULL,              -- Forecast method (LinearTrend, VAR, etc.)
    RegimeLabel     INT NULL,                           -- Operating regime at forecast time
    CreatedAt       DATETIME2 NOT NULL DEFAULT GETDATE(),
    
    -- Composite primary key (unique per run/equipment/timestamp/sensor)
    CONSTRAINT PK_ACM_SensorForecast PRIMARY KEY CLUSTERED (RunID, EquipID, Timestamp, SensorName),
    
    -- Foreign keys
    CONSTRAINT FK_ACM_SensorForecast_Runs FOREIGN KEY (RunID) REFERENCES dbo.ACM_Runs(RunID) ON DELETE CASCADE,
    CONSTRAINT FK_ACM_SensorForecast_Equipment FOREIGN KEY (EquipID) REFERENCES dbo.Equipment(EquipID)
);
GO

-- Performance indexes
CREATE NONCLUSTERED INDEX IX_ACM_SensorForecast_EquipID_Timestamp
    ON dbo.ACM_SensorForecast(EquipID, Timestamp)
    INCLUDE (SensorName, ForecastValue, CiLower, CiUpper);
GO

CREATE NONCLUSTERED INDEX IX_ACM_SensorForecast_SensorName
    ON dbo.ACM_SensorForecast(SensorName, EquipID, Timestamp)
    INCLUDE (ForecastValue, Method);
GO

PRINT 'ACM_SensorForecast table created successfully with indexes';
GO
