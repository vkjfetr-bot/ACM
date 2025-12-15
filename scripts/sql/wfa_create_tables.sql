-- Wind Farm A per-turbine table creation (CARE_To_Compare data)
-- Run in SQL Server (sqlcmd recommended). Creates one _Data table per asset_id.

SET NOCOUNT ON;
SET QUOTED_IDENTIFIER ON;

DECLARE @assetIds TABLE(asset_id INT NOT NULL);
INSERT INTO @assetIds(asset_id) VALUES
 (0),(3),(10),(11),(13),(14),(17),(21),(22),(24),(25),(26),
 (38),(40),(42),(45),(51),(68),(69),(71),(72),(73),(84),(92);

DECLARE @sql NVARCHAR(MAX);

DECLARE cur CURSOR FAST_FORWARD FOR
SELECT asset_id FROM @assetIds;

OPEN cur;
DECLARE @asset INT;
FETCH NEXT FROM cur INTO @asset;

WHILE @@FETCH_STATUS = 0
BEGIN
    DECLARE @table SYSNAME = CONCAT('WFA_TURBINE_', @asset, '_Data');
    DECLARE @qualifiedTable NVARCHAR(300) = QUOTENAME('dbo') + '.' + QUOTENAME(@table);

    -- Drop old table if rerunning
    IF OBJECT_ID(@qualifiedTable) IS NOT NULL
        EXEC('DROP TABLE ' + @qualifiedTable + ';');

    SET @sql = N'
    CREATE TABLE ' + @qualifiedTable + N' (
        EntryDateTime      DATETIME2       NOT NULL,
        asset_id           INT             NOT NULL,
        [id]               INT             NOT NULL,
        train_test         VARCHAR(16)     NULL,
        status_type_id     INT             NULL,
        sensor_0_avg       FLOAT           NULL,
        sensor_1_avg       FLOAT           NULL,
        sensor_2_avg       FLOAT           NULL,
        wind_speed_3_avg   FLOAT           NULL,
        wind_speed_4_avg   FLOAT           NULL,
        wind_speed_3_max   FLOAT           NULL,
        wind_speed_3_min   FLOAT           NULL,
        wind_speed_3_std   FLOAT           NULL,
        sensor_5_avg       FLOAT           NULL,
        sensor_5_max       FLOAT           NULL,
        sensor_5_min       FLOAT           NULL,
        sensor_5_std       FLOAT           NULL,
        sensor_6_avg       FLOAT           NULL,
        sensor_7_avg       FLOAT           NULL,
        sensor_8_avg       FLOAT           NULL,
        sensor_9_avg       FLOAT           NULL,
        sensor_10_avg      FLOAT           NULL,
        sensor_11_avg      FLOAT           NULL,
        sensor_12_avg      FLOAT           NULL,
        sensor_13_avg      FLOAT           NULL,
        sensor_14_avg      FLOAT           NULL,
        sensor_15_avg      FLOAT           NULL,
        sensor_16_avg      FLOAT           NULL,
        sensor_17_avg      FLOAT           NULL,
        sensor_18_avg      FLOAT           NULL,
        sensor_18_max      FLOAT           NULL,
        sensor_18_min      FLOAT           NULL,
        sensor_18_std      FLOAT           NULL,
        sensor_19_avg      FLOAT           NULL,
        sensor_20_avg      FLOAT           NULL,
        sensor_21_avg      FLOAT           NULL,
        sensor_22_avg      FLOAT           NULL,
        sensor_23_avg      FLOAT           NULL,
        sensor_24_avg      FLOAT           NULL,
        sensor_25_avg      FLOAT           NULL,
        sensor_26_avg      FLOAT           NULL,
        reactive_power_27_avg FLOAT        NULL,
        reactive_power_27_max FLOAT        NULL,
        reactive_power_27_min FLOAT        NULL,
        reactive_power_27_std FLOAT        NULL,
        reactive_power_28_avg FLOAT        NULL,
        reactive_power_28_max FLOAT        NULL,
        reactive_power_28_min FLOAT        NULL,
        reactive_power_28_std FLOAT        NULL,
        power_29_avg       FLOAT           NULL,
        power_29_max       FLOAT           NULL,
        power_29_min       FLOAT           NULL,
        power_29_std       FLOAT           NULL,
        power_30_avg       FLOAT           NULL,
        power_30_max       FLOAT           NULL,
        power_30_min       FLOAT           NULL,
        power_30_std       FLOAT           NULL,';

    SET @sql += N'
        sensor_31_avg      FLOAT           NULL,
        sensor_31_max      FLOAT           NULL,
        sensor_31_min      FLOAT           NULL,
        sensor_31_std      FLOAT           NULL,
        sensor_32_avg      FLOAT           NULL,
        sensor_33_avg      FLOAT           NULL,
        sensor_34_avg      FLOAT           NULL,
        sensor_35_avg      FLOAT           NULL,
        sensor_36_avg      FLOAT           NULL,
        sensor_37_avg      FLOAT           NULL,
        sensor_38_avg      FLOAT           NULL,
        sensor_39_avg      FLOAT           NULL,
        sensor_40_avg      FLOAT           NULL,
        sensor_41_avg      FLOAT           NULL,
        sensor_42_avg      FLOAT           NULL,
        sensor_43_avg      FLOAT           NULL,
        [sensor_44]       FLOAT           NULL,
        [sensor_45]       FLOAT           NULL,
        [sensor_46]       FLOAT           NULL,
        [sensor_47]       FLOAT           NULL,
        [sensor_48]       FLOAT           NULL,
        [sensor_49]       FLOAT           NULL,
        [sensor_50]       FLOAT           NULL,
        [sensor_51]       FLOAT           NULL,';

    SET @sql += N'
        sensor_52_avg      FLOAT           NULL,
        sensor_52_max      FLOAT           NULL,
        sensor_52_min      FLOAT           NULL,
        sensor_52_std      FLOAT           NULL,
        sensor_53_avg      FLOAT           NULL,
        QualityFlag        INT             NOT NULL CONSTRAINT ' + QUOTENAME(CONCAT('DF_', @table, '_QualityFlag')) + N' DEFAULT (0),
        CONSTRAINT ' + QUOTENAME(CONCAT('PK_', @table)) + N' PRIMARY KEY (EntryDateTime)
    );
    ';

    EXEC(@sql);

    FETCH NEXT FROM cur INTO @asset;
END

CLOSE cur;
DEALLOCATE cur;

PRINT 'Wind Farm A tables created.';