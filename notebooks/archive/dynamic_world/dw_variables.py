DW_CLASSES = {
    0: 'Water', 1: 'Trees', 2: 'Grass', 3: 'Flooded Veg',
    4: 'Crops', 5: 'Shrub/Scrub', 6: 'Built Area', 7: 'Bare Ground', 8: 'Snow/Ice'
}
DW_PALETTE = [
    '#419BDF', '#397D49', '#88B053', '#7A87C6', '#E49635',
    '#DFC35A', '#C4281B', '#A59B8F', '#B39FE1'
]
DW_VIS = {'min': 0, 'max': 8, 'palette': [c.lstrip('#') for c in DW_PALETTE]}