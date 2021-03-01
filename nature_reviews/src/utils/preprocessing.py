SPECIAL_SECTIONS = ['INTRODUCTION', 'CONCLUSION', 'PERSPECTIVES']
UNIQUENESS_METHODS = ['first_occurrence', 'unique_only']


def is_box_section(title):
    """
    Box sections have specific titles: "Box N | Title goes here".
    This function determines whether title belongs to a box section.
    """
    if '|' not in title:
        return False
    return 'Box' in title.split('|')[0]


def flatten_clustering_impl(clustering, max_level, current_level):
    """
    Recursive traversal of multi-level dict-like clustering.
    Clusters on level larger than `max_level` are merged.
    """
    current_level_elements = []
    clusters = []
    num_clusters = 0

    for el in clustering:
        if isinstance(el, dict):
            inner_clusters, num_inner_clusters = flatten_clustering_impl(el['references'],
                                                                         max_level, current_level + 1)
            if max_level <= current_level:
                current_level_elements.extend(inner_clusters)
            else:
                if num_inner_clusters == 1:
                    clusters.append(inner_clusters)
                else:
                    clusters.extend(inner_clusters)
                num_clusters += num_inner_clusters
        else:
            current_level_elements.append(el)

    if max_level <= current_level or not clusters:
        return current_level_elements, 1
    else:
        if current_level_elements:
            clusters.insert(0, current_level_elements)
            num_clusters += 1
        if num_clusters == 1:
            return clusters[0], num_clusters
        else:
            return clusters, num_clusters


def flatten_clustering(clustering, max_level,
                       include_special_sections,
                       include_box_sections):
    """
    Builds a flat version of multi-level clustering sliced at `max_level`.
    max_level is 1-based.
    """
    flat_clustering = []
    for section in clustering:
        is_special_section = section['title'] in SPECIAL_SECTIONS
        if not include_special_sections and is_special_section:
            continue
        if not include_box_sections and is_box_section(section['title']):
            continue
        clusters, num_clusters = flatten_clustering_impl(section['references'], max_level, 1)
        if num_clusters == 1:
            flat_clustering.append(clusters)
        else:
            flat_clustering.extend(clusters)
    return flat_clustering


def unique_ids_clustering(clustering, method):
    """
    Ensures that each paper is assigned only to one cluster by one of the methods:
     * 'first_occurence' - paper is assigned to the first cluster
     * 'unique_only' - only papers belonging to one cluster are taken into account
    """
    id_cluster = {}
    for i, cluster in enumerate(clustering):
        for pmid in cluster:
            if pmid not in id_cluster:
                id_cluster[pmid] = []
            id_cluster[pmid].append(i)
    if method == 'first_occurrence':
        return {str(k): v[0] for k, v in id_cluster.items()}
    elif method == 'unique_only':
        return {str(k): v[0] for k, v in id_cluster.items() if len(set(v)) == 1}


def preprocess_clustering(clustering, max_level,
                          include_special_sections=False,
                          include_box_sections=True,
                          uniqueness_method=None):
    """
    Convert raw multi-level clustering to a flat clustering with unique ids.
    max_level is 1-based.
    """
    flat_clustering = flatten_clustering(clustering, max_level,
                                         include_special_sections,
                                         include_box_sections)

    if uniqueness_method:
        if uniqueness_method not in UNIQUENESS_METHODS:
            raise ValueError('Unrecognized uniqueness method')
        return unique_ids_clustering(flat_clustering, uniqueness_method)

    return flat_clustering


def get_clustering_level(clustering):
    """
    Returns the numbers of levels in the clustering.
    Intended to be used as an upper bound for 1-based `max_level` in clustering preprocessing.
    """
    levels = []
    for el in clustering:
        if isinstance(el, dict):
            levels.append(get_clustering_level(el['references']))
    if not levels:
        return 1
    else:
        return max(levels) + 1
