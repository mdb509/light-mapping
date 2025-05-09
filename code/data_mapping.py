import os
import gpxpy
import pandas as pd

class DataMapper():
    def gpx_to_csv(self, gpx_file):
        """ reads gpx file and converts it to csv and saves it"""
        # Parse GPX file
        with open(gpx_file, 'r') as gpx_file:
            gpx = gpxpy.parse(gpx_file)

        # Extract data points
        data = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    data.append({
                        'timestamp': point.time.timestamp() if point.time else None,
                        'latitude': point.latitude,
                        'longitude': point.longitude,
                        'elevation': point.elevation
                    })

        # Convert to DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv("output_gpx.csv", index=False)
        print("CSV saved as 'output.csv'")
    
    def biolog_to_csv(self, biolog_file):
        with open(biolog_file, 'r') as file:
            c = 0
            data = []
            for line in file:
                if c > 10:
                    break
                c += 1
                print(line)
                line = ' '.join(line.split())
                point = line.split(' ')
                data.append({
                    'unix_timestamp': point.time.timestamp() if point.time else None,
                    'date': point.latitude,
                    'time': point.longitude,
                    'battery_voltage_mV': point.elevation
                })


    
if __name__ == '__main__':
    d = DataMapper()
    current_folder = os.path.dirname(os.path.abspath(__file__))
    # filename = current_folder +"/data/Wanderung.gpx"
    # d.gpx_to_csv(filename)
    filename = current_folder + "/data/BIOLOG.TXT"
    d.biolog_to_csv(filename)
