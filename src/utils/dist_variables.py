from datetime import date

# VEG-DIST-DATE encodes days since this date (per HLSDIST documentation)
VEGDISTDATE_EPOCH = date(2020, 12, 31)

# Annual Disturbance Product Folder Locations
FOLDERSET = {
        '2025': "projects/glad/HLSDIST/DIST-ANN_v1_2025",
        '2024': "projects/glad/HLSDIST/DIST-ANN_v1_2024",
        '2023': "projects/glad/HLSDIST/DIST-ANN_v1"
    }

# DIST-STATUS color palette
STATUSPAL = [
    "121212",  # black
    "E48727",  # light orange
    "E01B07",  # red
    "777777",  # grey
    "DDDDDD",  # light grey
    "005555",  # dark slate
    "008888"   # dark cyan
]

# VEG-ANOM-MAX color palette
ANOMPAL = ["222222",
           "FEE187FF",
           "FEC965FF",
           "FEAB49FF",
           "FD8D3CFF",
           "FC5B2EFF",
           "ED2F22FF",
           "D41020FF",
           "B10026FF",
           "800026FF"]

# VEG-DIST-CONF color palette 
CONFPAL = ['000000',
           "FFFFCCFF",
           "FFEFA5FF",
           "FEDD7FFF",
           "FEBF5AFF",
           "FD9D43FF",
           "FD7134FF",
           "F43D25FF",
           "DB141EFF",
           "B60026FF",
           "800026FF"]

# DIST-STATSU Legend Visualization Dictionary
LEGEND_DICT = {
    "No disturbance" : "121212",
    "confirmed <50% ongoing" : "E48727",
    "confirmed ≥50% ongoing" : "E01B07",
    "confirmed <50% finished" : "777777",
    "confirmed ≥50% finished" : "DDDDDD",
    "confirmed previous year <50%" : "005555",
    "confirmed previous year ≥50%" : "008888"
}

DW_CLASSES = {
    0: 'Water', 1: 'Trees', 2: 'Grass', 3: 'Flooded Veg',
    4: 'Crops', 5: 'Shrub/Scrub', 6: 'Built Area', 7: 'Bare Ground', 8: 'Snow/Ice'
}
DW_PALETTE = [
    '#419BDF', '#397D49', '#88B053', '#7A87C6', '#E49635',
    '#DFC35A', '#C4281B', '#A59B8F', '#B39FE1'
]
DW_VIS = {'min': 0, 'max': 8, 'palette': [c.lstrip('#') for c in DW_PALETTE]}

