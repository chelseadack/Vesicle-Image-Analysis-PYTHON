## ----------------- FUNCTION TO SUMMARISE THE VESICLES IN EACH CONDITION & IMAGE ----------------------------------------------------------------

# Makes 2 tables from the measurements of every vesicle:
#   - conditionTable = one row per condition, all repeats pooled.
#   - imageTable = one row per image (ie. each repeat), including any images where no vesicles were found.
# Both give the number of vesicles (GUVs) and the mean & median measured fluorescence
# (the asterisks & notches on the violin plots, if no vesicles are left out of the plots).
# For 'MEMBRANE-INNER', they also give the number & % of GUVs localised (membrane/lumen ratio > localised_ratio).

# Run this function with:
# conditionTable, imageTable = func_summary_tables_vesicle(vesicleData, imageList)

localised_ratio = 1  # a GUV is localised if its membrane/lumen ratio is above this


def func_summary_tables_vesicle(vesicleData, imageList):
    #   Input parameters:
    #       - vesicleData, imageList = tables returned by func_detect_measure_vesicle.
    #   Output:
    #       - conditionTable, imageTable = see above.

    is_ratio = vesicleData.attrs['settings']['measure_fluorescence_type'] == 'MEMBRANE-INNER'  # % localised only means something for a ratio
    vesicles = vesicleData.assign(Localised=vesicleData['MES'] > localised_ratio)
    summary = dict(TotalGUVs=('MES', 'size'), LocalisedGUVs=('Localised', 'sum'), MeanMES=('MES', 'mean'), MedianMES=('MES', 'median'))
    count_columns = ['TotalGUVs', 'LocalisedGUVs']

    conditionTable = vesicles.groupby('Condition', observed=False).agg(**summary)
    imageTable = vesicles.groupby(['Condition', 'Image'], observed=True).agg(**summary).reset_index()
    imageTable = imageList.merge(imageTable, on=['Condition', 'Image'], how='left')  # keeps images with no GUVs
    imageTable[count_columns] = imageTable[count_columns].fillna(0).astype(int)

    tables = []
    for table in [conditionTable, imageTable]:
        table.insert(table.columns.get_loc('MeanMES'), 'PercentLocalised', table['LocalisedGUVs'] / table['TotalGUVs'] * 100)
        tables.append(table if is_ratio else table.drop(columns=['LocalisedGUVs', 'PercentLocalised']))

    return tables[0], tables[1]
