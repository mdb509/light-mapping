import os
import gpxpy
import pandas as pd

class GPX():
    def gpx_to_csv(self, gpx_file):
        # Parse GPX file
        with open(gpx_file, 'r') as gpx_file:
            gpx = gpxpy.parse(gpx_file)

        # Extract data points
        data = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    data.append({
                        'time': point.time,
                        'latitude': point.latitude,
                        'longitude': point.longitude,
                        'elevation': point.elevation
                    })

        # Convert to DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv('output.csv', index=False)
        print("CSV saved as 'output.csv'")
    
    
if __name__ == '__main__':
    g = GPX()
    current_folder = os.path.dirname(os.path.abspath(__file__))
    filename = current_folder +'/data/Wanderung.gpx'
    g.gpx_to_csv(filename)
