import unittest
from reservoir_control.systems.ottawa.historical_pilot import extract, symbol, requests_for


class HistoricalAcquisitionChecks(unittest.TestCase):
    def fixture(self):
        return dict(features=[dict(properties=dict(STATION_NUMBER='TEST', DATE='2000-01-02', LEVEL=None)),
                              dict(properties=dict(STATION_NUMBER='TEST', DATE='2000-01-01', LEVEL=1.))])

    def test_source_dates_and_missing_values_are_preserved(self):
        rows = extract(self.fixture(), 'TEST')
        self.assertEqual(rows.DATE.tolist(), ['2000-01-01', '2000-01-02'])
        self.assertTrue(rows.LEVEL.isna().iloc[1])
        self.assertEqual(symbol(None), 'HYDAT_unflagged')
        self.assertEqual(symbol('Partial Day'), 'HYDAT_symbol:Partial Day')

    def test_wrong_station_duplicates_and_incomplete_pages_fail(self):
        data = self.fixture()
        with self.assertRaises(ValueError): extract(data, 'OTHER')
        data['features'].append(data['features'][0])
        with self.assertRaises(ValueError): extract(data, 'TEST')
        data = self.fixture(); data['links'] = [dict(rel='next')]
        with self.assertRaises(ValueError): extract(data, 'TEST')

    def test_downloads_target_official_historical_not_realtime_endpoint(self):
        c = dict(api_root='https://api.weather.gc.ca/collections/', dataset='hydrometric-daily-mean',
                 level_station='02LG007', discharge_station='02LG006',
                 target=dict(start_date='1970-01-01', end_date_exclusive='1971-01-01'))
        urls = requests_for(c)
        self.assertEqual(len(urls), 5)
        self.assertIn('1970-01-01', urls['02LG007_daily.json'])
        self.assertIn('1971-01-01', urls['02LG007_daily.json'])
        self.assertNotIn('realtime', str(urls))


if __name__ == '__main__': unittest.main()
