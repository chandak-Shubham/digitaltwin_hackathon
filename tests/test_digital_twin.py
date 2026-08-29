import pandas as pd

from digital_twin import build_station_history, explain_vehicle_anomaly


sample_df = pd.DataFrame(
    [
        {
            'vehicle_id': 'VEH-00001',
            'station_id': 'S1',
            'cycle_time_sec': 60.0,
            'torque_nm': 140.0,
            'temperature_c': 22.0,
            'vibration_rms': 1.0,
            'pressure_bar': 6.0,
            'force_n': 450.0,
            'position_error_mm': 0.5,
            'anomaly_flag': 1,
            'root_cause_station': 'S1',
        },
        {
            'vehicle_id': 'VEH-00001',
            'station_id': 'S2',
            'cycle_time_sec': 65.0,
            'torque_nm': 138.0,
            'temperature_c': 23.0,
            'vibration_rms': 1.1,
            'pressure_bar': 6.1,
            'force_n': 440.0,
            'position_error_mm': 0.6,
            'anomaly_flag': 1,
            'root_cause_station': 'S1',
        },
        {
            'vehicle_id': 'VEH-00001',
            'station_id': 'S3',
            'cycle_time_sec': 70.0,
            'torque_nm': 110.0,
            'temperature_c': 25.0,
            'vibration_rms': 2.2,
            'pressure_bar': 5.2,
            'force_n': 360.0,
            'position_error_mm': 1.8,
            'anomaly_flag': 1,
            'root_cause_station': 'S1',
        },
        {
            'vehicle_id': 'VEH-00002',
            'station_id': 'S1',
            'cycle_time_sec': 50.0,
            'torque_nm': 130.0,
            'temperature_c': 21.0,
            'vibration_rms': 0.8,
            'pressure_bar': 6.2,
            'force_n': 420.0,
            'position_error_mm': 0.2,
            'anomaly_flag': 0,
            'root_cause_station': 'NONE',
        },
        {
            'vehicle_id': 'VEH-00002',
            'station_id': 'S2',
            'cycle_time_sec': 52.0,
            'torque_nm': 131.0,
            'temperature_c': 21.0,
            'vibration_rms': 0.7,
            'pressure_bar': 6.1,
            'force_n': 424.0,
            'position_error_mm': 0.2,
            'anomaly_flag': 0,
            'root_cause_station': 'NONE',
        },
    ]
)


def test_build_station_history_adds_sequences():
    result = build_station_history(sample_df)

    assert 'station_index' in result.columns
    assert 'vehicle_sequence' in result.columns
    assert 'is_anomalous_vehicle' in result.columns
    assert result['vehicle_sequence'].tolist() == [1, 2, 3, 1, 2]


def test_explain_vehicle_anomaly_returns_contributors():
    result = explain_vehicle_anomaly(sample_df, 'VEH-00001', 'S3')

    assert result['predicted_station'] == 'S3'
    assert result['root_cause_station'] == 'S1'
    assert 'contributors' in result
    assert any(item['station_id'] == 'S1' for item in result['contributors'])
    assert any(item['feature'] == 'torque_nm' for item in result['contributors'])
