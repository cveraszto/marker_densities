import numpy as np
import sys
import xlrd
from IPython.display import clear_output


def scale_density_dataset(
    density_dataset,
    scaling_factor,
    maximum_density,
    full_voxels=None,
    empty_voxels=None,
):
    """
    Scales density_dataset by a positive scaling_factor.
    The upper limit for each voxel is given by maximum_density.
    full_voxels is a subset of density_dataset, density_dataset for these voxels will be set
    with the value of maximum_density.
    empty_voxels is a subset of density_dataset, density_dataset for these voxels will be set to 0.
    This function allows to scale the total density in re-distributing it across every voxels
    while remaining under the upper limit and matching specific voxels conditions.

    Arguments:
        density_dataset: 3D numpy ndarray of float
        scfac: float factor to to scale the dataset
        maximum_density: 3D numpy ndarray of float. Should have the same shape as density_dataset.
        full_voxels: list of transposed voxel ids of shape (3, m).
            Will be set to maximum_density in density_dataset
        empty_voxels: list of transposed voxel ids of shape (3, n).
            Will be set to 0 in density_dataset.

    Returns:
        3d numpy ndarray of float, scaled density dataset
    Raises:
        ValueError if maximum_density and density_dataset does not have the same shape.
            if scaling_factor is negative.
    """

    if density_dataset.shape != maximum_density.shape:
        raise ValueError("Dataset and dataset limit should have the same size")
    if scaling_factor < 0:
        raise ValueError("The scaling factor should be positive or null.")

    scaled_density = np.copy(density_dataset)
    scaled_density *= scaling_factor
    impossible_voxels = np.where(scaled_density > maximum_density)
    scaled_density[impossible_voxels] = maximum_density[impossible_voxels]
    if full_voxels is not None:
        scaled_density[
            full_voxels[0], full_voxels[1], full_voxels[2]
        ] = maximum_density[full_voxels[0], full_voxels[1], full_voxels[2]]
    if empty_voxels is not None:
        scaled_density[empty_voxels[0], empty_voxels[1], empty_voxels[2]] = 0
    return scaled_density


def fill_density_dataset(
    target_num,
    marker_dataset,
    maximum_density,
    full_voxels=None,
    empty_voxels=None,
    epsilon=1e-3,
):
    """
    Fills a 3d numpy ndarray so that the sum of its voxels equals a target number (+-epsilon).
    The final densities will follow the marker expression distribution.
    The value of each voxel of the final densities is limited by maximum_density.
    The final densities values are set to maximum_density values for full_voxels.
    The final densities values are set to 0 for empty_voxels.

    Arguments:
        target_num: Target number corresponding to the sum of all voxels
        marker_dataset: 3D numpy ndarray of float marker intensity.
        maximum_density: 3D numpy ndarray of float maximum density.
            Should have the same shape as marker_dataset.
        full_voxels: list of transposed voxel ids of shape (3, m).
            Will be set to maximum_density in the resulting density dataset
        empty_voxels: list of transposed voxel ids of shape  (3, m).
            Will be set to 0 in the resulting density dataset.
        epsilon: maximum difference between the sum of the final densities
            and the target number

    Returns:
        3d numpy ndarray of float, dataset filled with the target number of cells
    Raises:
        ValueError if the target number is greater the sum of the voxels of maximum_density
            if the sum of the full_voxels values is greater than the target_number
            if the marker dataset contains negative values or is filled with 0.
    """

    if target_num < epsilon:
        return np.zeros(marker_dataset.shape, dtype=np.float64)

    excessive_number = target_num - np.sum(maximum_density)
    if (
        empty_voxels is not None
        and excessive_number
        + np.sum(maximum_density[empty_voxels[0], empty_voxels[1], empty_voxels[2]])
        >= epsilon
        or empty_voxels is None
        and excessive_number >= epsilon
    ):
        raise ValueError("Trying to place more cells than the density limit.")
    if (
        full_voxels is not None
        and np.sum(maximum_density[full_voxels[0], full_voxels[1], full_voxels[2]])
        - target_num
        >= epsilon
    ):
        raise ValueError(
            "The sum of the full voxels values is greater than the target number."
        )
    sum_marker = np.sum(marker_dataset)
    if (marker_dataset < 0).any() or sum_marker <= 0:
        raise ValueError("The marker dataset should contain only positive values.")

    final_density = np.copy(marker_dataset)
    actual_num = 0
    scaling_factor = float(target_num) / sum_marker
    while abs(actual_num - target_num) >= epsilon:
        final_density = scale_density_dataset(
            marker_dataset, scaling_factor, maximum_density, full_voxels, empty_voxels
        )
        actual_num = np.sum(final_density)
        # print(actual_num, "/", target_num)
        scaling_factor *= float(target_num) / actual_num
    return final_density


def read_densities_sheet(filename, region_keys, columns_mean, sheet_indices=[0], num_first_row=1, column_name=0):
    """
    Opens and reads an excel file containing mean and standard deviation values of cell densities
    Read the 'sheet_indices' sheets of the file, skip the num_first_row first rows of the sheet (Header)
    The region names will be read in the column column_name
    Rows where mean or std is defined as "N/D" are skipped.
    The mean densities will be read in the column listed in columns_mean and averaged.
    The standard deviations of the densities will be read in the columns following the ones listed in columns_mean
    Region names absent of the region_keys list will be returned in a dictionary

    Arguments:
        filename: Path to Excel file containing densities
        region_keys: List of regions name
        columns_mean: 2D array of int: For each marker, list of columns where to read the mean value.
        For each mean value, its standard deviation should be stored in the following column.
        sheet_indices: indices of the sheet to read in the excel file
        num_first_row: number of row to skip (header)
        column_name: indice of the column containing region names

    Returns:
        List of region names
        List of mean densities
        List of minimum values (mean - standard deviation)
        List of maximum values (mean + standard deviation)
        Dictionary where each key is a name of a region not in region_key and its value is []
    """

    names = []
    num_marker = len(columns_mean)

    mean_literature = [[] for _ in range(num_marker)]
    down_std_literature = [[] for _ in range(num_marker)]
    up_std_literature = [[] for _ in range(num_marker)]
    convert = {}
    wb = xlrd.open_workbook(filename)
    for i_sheet in sheet_indices:
        sheet = wb.sheet_by_index(i_sheet)
        for i_region in range(sheet.nrows-num_first_row): 
            if "N/D" not in [sheet.cell_value(i_region+num_first_row, i) for i in range(np.min(columns_mean), np.max(columns_mean)+1)]:
                names.append(sheet.cell_value(i_region+num_first_row, column_name).replace("È", "e"))
                for i_marker in range(num_marker):
                    mean_value = 0.
                    std_value = 0.
                    for column in columns_mean[i_marker]:
                        mean_value += sheet.cell_value(i_region + num_first_row, column)
                        std_value += sheet.cell_value(i_region + num_first_row, column+1)
                    std_value/=len(columns_mean[i_marker])
                    mean_value/=len(columns_mean[i_marker])
                    mean_literature[i_marker].append(mean_value)
                    down_std_literature[i_marker].append(max(mean_value - std_value, 0.))
                    up_std_literature[i_marker].append(mean_value + std_value)
                if names[-1] not in region_keys:
                    convert[names[-1]] = []
    return np.array(names), np.array(mean_literature), np.array(down_std_literature), np.array(up_std_literature), convert


def process_numbers(mean_markers, min_markers, max_markers, top_lim_num):
    """
    Scaled down the mean counts estimated for each marker so that it matches a top_lim_num constraint
    If the sum of the minimum estimates is greater than the top_lim_num constraint
        The mean values are scaled so that their sum is equal to top_lim_num
        The ratios of the different markers is kept.
        Minimum and maximum estimates are set to the mean estimate
    Else if the sum of the mean estimates is greater than the top_lim_num constraint
        The mean values are scaled so that their sum is equal to top_lim_num
        The ratios of the different markers is adapted to maintain the minimum estimates
        The max values are set to the mean values
    Else
        The max value for each marker is set to the top_lim_num - the sum of the mean estimates of the other markers
    Arguments:
        mean_markers: List of mean counts estimates for each marker
        min_markers: List of minimums counts estimates for each marker
        max_markers: List of maximums counts estimates for each marker
        top_lim_num: Maximum count of cell in the region of interest

    Returns:
        List of mean counts estimates for each marker after scaling
        List of minimums counts estimates for each marker after scaling
        List of maximums counts estimates for each marker after scaling
    """
    sum_ = np.sum(mean_markers)
    if np.sum(min_markers) > top_lim_num:
        diff_ = sum_ - top_lim_num
        mean_markers -= diff_ * mean_markers / sum_
        min_markers = np.copy(mean_markers)
        max_markers = np.copy(mean_markers)
    elif sum_ > top_lim_num:
        marker_sums = np.ones(mean_markers.shape) * sum_
        while sum_ > top_lim_num:
            ratios = np.divide(mean_markers, marker_sums, out=np.zeros_like(mean_markers), where=marker_sums!=0)
            diff_ = sum_ - top_lim_num
            mean_markers = np.maximum(min_markers, mean_markers - ratios * diff_)
            sum_ = np.sum(mean_markers)
            marker_sums =  np.ones(mean_markers.shape) * sum_
            marker_sums -= np.sum(mean_markers[mean_markers == min_markers])
        max_markers = np.copy(mean_markers)
    else:
        num_marker = len(max_markers)
        indices = np.arange(num_marker)
        for i_marker in range(num_marker): 
            max_markers[i_marker] = min(max_markers[i_marker], top_lim_num - np.sum(mean_markers[indices!=i_marker]))
    return min_markers, mean_markers, max_markers


def progress_bar(current_percent, previous_message=None, next_message=None):
    """
    Displays a progress bar with additional messages.
    Can be recalled to update the progress bar.
    Parameters:
        current_percent: Current percentage of completion
        previous_message: Text to display before the progress bar
        next_message: Text to display after the progress bar
    """

    color = "\033[91m"  # red
    current_percent = int(current_percent)
    if current_percent > 33:
        color = "\033[93m" # yellow
    if current_percent > 66:
        color = "\033[92m" # green
    progress = "\r["+ color + "%s" % ("=" * current_percent) + \
                "\033[0m"+"%s" % ("-" * (100 - current_percent)) + "] " + str(current_percent) + "%\n"
    clear_output(wait=True)
    if previous_message is not None:
        print(previous_message)
    sys.stdout.write(progress)
    sys.stdout.flush()
    if next_message is not None:
        print(next_message)


def find_unique_regions(annotation, 
                        id_to_region_dictionary_ALLNAME, 
                        region_dictionary_to_id_ALLNAME,
                        region_dictionary_to_id_ALLNAME_parent, 
                        name2allname,
                        top_region_name="Basic cell groups and regions"):
    """
    Finds unique regions ids that are present in an annotation file and are contained in the top_region_name
    Adds also to the list each parent of the regions present in the annotation file.
    Dictionaries parameters correspond to the ones produced in JSONread

    Parameters:
        annotation: 3D numpy ndarray of integers ids of the regions
        id_to_region_dictionary_ALLNAME: dictionary from region id to region complete name
        region_dictionary_to_id_ALLNAME: dictionary from region complete name to region id
        region_dictionary_to_id_ALLNAME_parent: dictionary from region complete name to its parent complete name
        name2allname: dictionary from region name to region complete name
        top_region_name: name of the most broader region included in the uniques

    Returns:
        List of unique regions id in the annotation file that are included in top_region_name
    """

    # Take the parent of the top region to stop the loop 
    root_allname = region_dictionary_to_id_ALLNAME_parent[name2allname[top_region_name]]
    uniques = []
    for uniq in np.unique(annotation)[1:]: # Cell regions without outside
        allname = id_to_region_dictionary_ALLNAME[uniq]
        if top_region_name in id_to_region_dictionary_ALLNAME[uniq] and uniq not in uniques:
            uniques.append(uniq)
            parent_allname = region_dictionary_to_id_ALLNAME_parent[allname]
            id_parent = region_dictionary_to_id_ALLNAME[parent_allname]
            while id_parent not in uniques and parent_allname != root_allname:
                uniques.append(id_parent)
                parent_allname = region_dictionary_to_id_ALLNAME_parent[parent_allname]
                if parent_allname == "":
                    break
                id_parent = region_dictionary_to_id_ALLNAME[parent_allname]

    return np.array(uniques)


def get_neuron_counts(annotation, density,
                      id_to_region_dictionary_ALLNAME, is_leaf,
                      region_dictionary_to_id_ALLNAME,
                      region_dictionary_to_id_ALLNAME_parent,
                      name2allname):
    """
    Sum all the voxels values from a density dataset sorted by region
    Dictionaries parameters correspond to the ones produced in JSONread

    Parameters:
        annotation: 3D numpy ndarray of integers ids of the regions
        density: 3D numpy ndarray of floars counts of neurons in the brain

        id_to_region_dictionary_ALLNAME: dictionary from region id to region complete name
        is_leaf: dictionary from region complete name to boolean, True if the region is a leaf region.
        region_dictionary_to_id_ALLNAME: dictionary from region complete name to region id
        region_dictionary_to_id_ALLNAME_parent: dictionary from region complete name to its parent complete name
        name2allname: dictionary from region name to region complete name

    Returns:
        Dictionary from region complete name to number of neurons in that region
    """
    uniques = find_unique_regions(annotation,
                        id_to_region_dictionary_ALLNAME,
                        region_dictionary_to_id_ALLNAME,
                        region_dictionary_to_id_ALLNAME_parent,
                        name2allname)
    children, order_ = find_children(uniques, id_to_region_dictionary_ALLNAME, is_leaf,
                                     region_dictionary_to_id_ALLNAME_parent,
                                     region_dictionary_to_id_ALLNAME)
    child_neuron = {}
    for uniq in uniques:
        child_neuron[id_to_region_dictionary_ALLNAME[uniq]] = np.sum(density[annotation == uniq])
    num_neurons = {}
    for parent, child in children.items():
        children[parent] = np.unique(child)
        num_neurons[parent] = 0.0
        for id_reg in children[parent]:
            if id_to_region_dictionary_ALLNAME[id_reg] in child_neuron:
                num_neurons[parent] += child_neuron[id_to_region_dictionary_ALLNAME[id_reg]]
    for k, v in child_neuron.items():
        if k not in num_neurons.keys():
            num_neurons[k] = v
    num_neurons["|root"] += np.sum(density[annotation == 997])
    return num_neurons


def find_children(uniques, id_to_region_dictionary_ALLNAME, is_leaf,
                  region_dictionary_to_id_ALLNAME_parent, 
                  region_dictionary_to_id_ALLNAME):
    """
    Finds the children regions of each region id in uniques and its distance from a leaf region in the hierarchy tree.
    Non leaf regions are included in the children list
    Dictionaries parameters correspond to the ones produced in JSONread

    Parameters:
        uniques: List of unique region ids
        id_to_region_dictionary_ALLNAME: dictionary from region id to region complete name
        is_leaf: dictionary from region complete name to boolean, True if the region is a leaf region.
        region_dictionary_to_id_ALLNAME_parent: dictionary from region complete name to its parent complete name
        region_dictionary_to_id_ALLNAME: dictionary from region complete name to region id

    Returns:
         Dictionary of region complete name to list of child region ids
         List of distances from a leaf region in the hierarchy tree for each region in uniques.
    """

    children = {}
    order_ = np.zeros(uniques.shape)
    for id_reg, allname in id_to_region_dictionary_ALLNAME.items():
        if is_leaf[allname]:
            inc = 0
            ids_reg = [id_reg]
            parentname = region_dictionary_to_id_ALLNAME_parent[allname]
            while parentname != '':
                if parentname not in children:
                    children[parentname] = []
                children[parentname] += ids_reg
                inc+=1
                id_parent = region_dictionary_to_id_ALLNAME[parentname]
                if id_parent in uniques:
                    ids_reg.append(id_parent)
                    place_ = np.where(uniques==id_parent)
                    order_[place_] = max(order_[place_], inc)
                allname = parentname
                parentname = region_dictionary_to_id_ALLNAME_parent[allname]
                
    for parent, child in children.items():
        children[parent] = np.unique(child)
    return children, order_


def export_volumes(annotation, is_leaf, region_dictionary_to_id_ALLNAME, resolution=25.0):
    """
    Computes the volume in um3 of each region of the brain.
    Dictionaries parameters correspond to the ones produced in JSONread.

    Parameters:
        annotation: 3D numpy ndarray of integers ids of the regions
        is_leaf: dictionary from region complete name to boolean, True if the region is a leaf region.
        region_dictionary_to_id_ALLNAME: dictionary from region complete name to region id
        resolution: size of each voxel in um

    Returns:
        Dictionary of region complete name to its volume in um3
    """

    leafs = {}
    for k, v in is_leaf.items():
        if v:
            leafs[k] = region_dictionary_to_id_ALLNAME[k]

    rv_LEAF = {}
    for jn, idreg in leafs.items():
        rv_LEAF[ jn ] = np.nonzero(annotation==idreg)[0].shape[0] * (resolution**3.0)

    rv = {}
    for rnk in region_dictionary_to_id_ALLNAME.keys():
        rv[ rnk ] = 0.0
    for rnk in rv_LEAF.keys():
        split_name = rnk.split("|")
        for isn in range(1,len(split_name)):
            name_TMP = ""
            for iisn in range(isn+1):
                name_TMP += split_name[iisn]+"|"
            if len(name_TMP[:-1])>0:
                rv[ name_TMP[:-1] ] += rv_LEAF[ rnk ]
    return rv


def place_cells(annotation, uniques, children, 
                rv, neu_dens, num_neurons,
                markers_intensity, alphas, std_fitt, 
                names, mean_literature, down_std_literature, up_std_literature,
                id_to_region_dictionary_ALLNAME, is_leaf, id_to_region_dictionary):
    """
    Loops through all the unique region ids and compute the densities of each marker according to literature value if available
    or the transfer function from marker mean expresion in the region to region densities.
    The minimum, maximum and mean values are constrained by the densities of neuron in the brain.
    The uniques region ids should be sorted from leaf regions to main brain region.
    Dictionaries parameters correspond to the ones produced in JSONread.

    Parameters:
        annotation: 3D numpy ndarray of integers ids of the regions
        uniques: List of unique region ids
        children: Dictionary of region complete name to list of child region ids
        rv: Dictionary of region complete name to its volume in um3
        neu_dens: 3D numpy ndarray of float neurons counts in the brain (cell/voxel)
        num_neurons: Dictionary from region complete name to number of neurons in that region
        markers_intensity: List of 3d numpy ndarray of float processed expression of the different markers
        alphas: List of factors from marker mean intensity to region density
        std_fitt: Standard deviation of the above alphas factors.
        names: List of region names for which a literature value is available for each marker
        mean_literature: List of region mean density values for each marker according to literature
        down_std_literature: List of region minimal density values (mean - std) for each marker according to literature
        up_std_literature: List of region maximal density values (mean + std) for each marker according to literature
        id_to_region_dictionary_ALLNAME: dictionary from region id to region complete name
        is_leaf: dictionary from region complete name to boolean, True if the region is a leaf region.
        id_to_region_dictionary: dictionary from region id to region name

    Returns:
        List of 3D numpy ndarray of float marker positive cell counts in the brain (cell/voxel)
        List of dictionaries from region ids (in string to allow writing) to a list with the minimum and maximum marker positive cell counts
    """


    num_marker = len(markers_intensity)
    dens_markers = np.zeros((num_marker,) + annotation.shape, dtype=np.float32)
    std_markers = [{} for _ in range(num_marker)]

    percent_done = -1
    length_ = float(len(uniques))
    with np.errstate(divide='raise', invalid='raise', over='raise'):
        for iunique, id_reg in enumerate(uniques):
            # print current percentage
            current_percent = min(int(float(iunique) / length_ * 100.), 100)
            if current_percent > percent_done:
                progress_bar(current_percent)
            percent_done = current_percent

            # filter voxels in region
            allname = id_to_region_dictionary_ALLNAME[id_reg]
            if is_leaf[allname]:
                filter_ = annotation == id_reg
            else:
                ids_reg = np.concatenate((children[allname], [id_reg]))
                filter_ = np.in1d(annotation, ids_reg).reshape(annotation.shape)
            name = id_to_region_dictionary[id_reg]
            volume = rv[allname] / 1.0e9
            loc_neu_dens = neu_dens[filter_]
            num_neuron = num_neurons[allname]

            # choose source
            if name in names:  # Density from Literature
                loc_mean_markers = mean_literature[:, np.where(names == name)[0][0]] * volume
                loc_min_markers = down_std_literature[:, np.where(names == name)[0][0]] * volume
                loc_max_markers = up_std_literature[:, np.where(names == name)[0][0]] * volume
            else:  # Use marker intensity
                if allname.find("Cerebellum") >= 0 or allname.find("arbor vitae") >= 0:
                    place_ = 0
                elif allname.find("Isocortex") >= 0 or allname.find("Entorhinal area") >= 0 or allname.find(
                        "Piriform area") >= 0:
                    place_ = 1
                else:
                    place_ = 2
                mean_intensities = np.mean(markers_intensity[:, filter_], axis=1) * volume
                loc_mean_markers = alphas[:, place_] * mean_intensities
                loc_min_markers = (alphas[:, place_] - std_fitt[:, place_]) * mean_intensities
                loc_max_markers = (alphas[:, place_] + std_fitt[:, place_]) * mean_intensities
            loc_min_markers, loc_mean_markers, loc_max_markers = process_numbers(loc_mean_markers, loc_min_markers,
                                                                                 loc_max_markers, num_neuron)
            if not is_leaf[allname]:
                # Remaining neurons to place in the whole region
                num_placed = np.sum(dens_markers[:, filter_], axis=1)
                filter_ = annotation == id_reg
                loc_mean_markers = np.maximum(loc_mean_markers - num_placed, 0)
                loc_min_markers = np.maximum(loc_min_markers - num_placed, 0)
                loc_max_markers = np.maximum(loc_max_markers - num_placed, 0)

                # number of neuron to place in regions where annotation = id_reg
                loc_neu_dens = neu_dens[filter_]
                num_neuron = np.sum(loc_neu_dens)
                loc_min_markers, loc_mean_markers, loc_max_markers = process_numbers(loc_mean_markers, loc_min_markers,
                                                                                     loc_max_markers, num_neuron)
                loc_min_markers += num_placed
                loc_max_markers += num_placed
            else:
                num_placed = [0.] * num_marker

            sum_dens = np.zeros(annotation.shape)[filter_]
            for i_marker in range(num_marker):
                dens_placed = fill_density_dataset(loc_mean_markers[i_marker],
                                                   np.ones(annotation.shape)[filter_],
                                                   loc_neu_dens - sum_dens, None)
                dens_markers[i_marker, filter_] = np.copy(dens_placed)
                sum_dens += dens_placed
                std_markers[i_marker][str(id_reg)] = [
                    min(loc_min_markers[i_marker], num_placed[i_marker] + np.sum(dens_placed)),
                    max(loc_max_markers[i_marker], num_placed[i_marker] + np.sum(dens_placed))]
        progress_bar(100)
    return dens_markers, std_markers
