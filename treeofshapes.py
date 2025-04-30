import random
import numpy as np
np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))

from PIL import Image
from enum import Enum, auto
import higra as hg
import statistics

def split_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def set_px(nparr, pix, value):
    if len(nparr.shape) == 3:
        x = int(pix / (nparr.shape[1] * nparr.shape[2]))
        rest = pix - (x * (nparr.shape[1] * nparr.shape[2]))
        y = int(rest / nparr.shape[2])
        z = rest - y * nparr.shape[2]
        nparr[x][y][z] = value
    else:
        x = int(pix / nparr.shape[1])
        y = pix - x * nparr.shape[1]
        nparr[x][y] = value

def bound(low, high, value):
    if value > high:
        return high
    if value < low:
        return low
    return value

class Feature(Enum):
    NB_NODES = auto()
    DEPTH = auto()
    AVG_NODE_DEGREE = auto()
    # Maximum number of nodes at any single level.
    WIDTH = auto()
    NB_LEAVES = auto()
    MEAN_INTENSITY = auto()
    # Ratio of parent intensity to child intensity for all nodes.
    MEAN_DYNAMIC = auto()
    MEAN_AREA = auto()
    # Correlation between node area and its depth in the tree.
    AREA_TO_DEPTH_CORRELATION = auto()
    # Nb of switches between ascending and descending grey-level values
    NB_POLARITY_SWITCH = auto()
    # Nb of nodes having an area lower than n
    NB_SMALL_NODES = auto()
    # Topological clustering factor
    TTOS_NB_NODE_DIFFERENCE = auto()
    # Nb of nodes having more than one child
    NB_BRANCHING = auto()
    EUCLIDEAN_DISTANCE = auto()

class Node:
    def __init__(self, name, alt, parent):
        self.name = name
        self.alt = alt
        self.parent = parent
        self.children = []
        self.proper_part = []
        self.area = 0
        self.depth = 0

        # Min or max tree
        self.ct_class = None

        # GoS only attributes: multiple adj parents
        self.adj_parents = {}

        self.ct_parent = None
        self.adj_children = []
        self.ct_children = []

        self.interval_to_parent = None

        # Bounds of the node, i.e. the "closest" smaller/higher altitude of its parent/children
        self.lower_bound = None
        self.upper_bound = None

        # Children nodes corresponding to bounds (M+/M-)
        self.children_lower_bound = None
        self.children_upper_bound = None

        self.nb_gray_levels = 0

        # ctos to ttos information
        self.last_compressed = None
        self.compressed_proper_part = {}

        self.removed = False

    def enrich_node(self):
        # a shift is necessary: a hole 'lives' a grey level above
        self.__compute_ct_class()
        if self.ct_class == 'min':
            self.alt += 1
        self.__compute_interval_to_parent()
        if self.ct_class == 'max':
            self.nb_gray_levels = self.interval_to_parent[1] - self.interval_to_parent[0] + 1
        else:
            self.nb_gray_levels = self.interval_to_parent[0] - self.interval_to_parent[1] + 1

    def __compute_ct_class(self):
        if self.is_root():
            self.ct_class = 'min'
            return
        if self.parent.ct_class == 'max':
            if self.alt > self.parent.alt:
                self.ct_class = 'max'
            else:
                self.ct_class = 'min'
        elif self.parent.ct_class == 'min':
            if self.alt < self.parent.alt:
                self.ct_class = 'min'
            else:
                self.ct_class = 'max'

    def __compute_interval_to_parent(self):
        if self.is_root():
            self.interval_to_parent = (self.alt, self.alt)
            return
        if self.ct_class == self.parent.ct_class:
            if self.ct_class == 'max':
                self.interval_to_parent = (self.parent.alt + 1, self.alt)
            else:
                self.interval_to_parent = (self.parent.alt - 1, self.alt)
        else:
            self.interval_to_parent = (self.parent.alt, self.alt)

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return not self.children

    # returns the parent of self at altitude alt
    def get_ct_parent(self, alt):
        # self is defined over interval [alpha, omega]

        # if alt == alpha, the parent at altitude alt is the parent of self in the tos
        if alt == self.interval_to_parent[0]:
            return self.ct_parent
        # if alt is within ]alpha, omega], the parent is self
        if self.ct_class == 'max':
            if self.interval_to_parent[0] < alt <= self.interval_to_parent[1]:
                return self
        if self.interval_to_parent[0] > alt >= self.interval_to_parent[1]:
            return self
        # the altitude is strictly outside of self's interval: return None
        return None

    def get_adj_parent(self, alt):
        if self.ct_class == 'max':
            if self.interval_to_parent[0] <= alt <= self.interval_to_parent[1]:
                return self.adj_parents[alt]
            else:
                return None
        if self.ct_class == 'min':
            if self.interval_to_parent[0] >= alt >= self.interval_to_parent[1]:
                return self.adj_parents[alt]
            else:
                return None
        return None

        # GoS only

    def get_ct_children(self, altitude):
        children = []
        if self.ct_class == 'max':
            if not self.interval_to_parent[0] <= altitude <= self.interval_to_parent[1]:
                return []
            if self.interval_to_parent[0] <= altitude < self.interval_to_parent[1]:
                children.append(self)
            for c in self.ct_children:
                if c.interval_to_parent[0] == altitude + 1:
                    children.append(c)
        if self.ct_class == 'min':
            if not self.interval_to_parent[0] >= altitude >= self.interval_to_parent[1]:
                return []
            if self.interval_to_parent[0] >= altitude > self.interval_to_parent[1]:
                children.append(self)
            for c in self.ct_children:
                if c.interval_to_parent[0] == altitude - 1:
                    children.append(c)
        return children

    def get_adj_children(self, altitude):
        children = []
        for adj_c in self.adj_children:
            interval = adj_c[1]
            if self.ct_class == 'max':
                if not self.interval_to_parent[0] <= altitude <= self.interval_to_parent[1]:
                    return []
                if interval[0] <= altitude <= interval[1]:
                    children.append(adj_c[0])
            else:
                if not self.interval_to_parent[0] >= altitude >= self.interval_to_parent[1]:
                    return []
                if interval[0] >= altitude >= interval[1]:
                    children.append(adj_c[0])
        return children

    def compute_gos_relations(self):
        if self.is_root():
            return

        parent = self.parent
        # Case 1: node is the child of the root:
        # Node is added to both ct and adj relation of its tos parent (root)
        if parent.is_root():
            # CT
            self.__add_ct_parent(parent)
            # ADJ
            self.__add_adj_parent(self.interval_to_parent, self.parent)
            return

        z = parent.adj_parents[parent.interval_to_parent[1]].get_ct_parent(parent.interval_to_parent[1])

        # Case 2: node has a different ct class as its parent
        if self.ct_class != parent.ct_class:
            self.__add_ct_parent(z)
            first_adj = parent
        # Case 3: node has the same ct class
        else:
            self.__add_ct_parent(parent)
            first_adj = z
        # Common treatment: adding the adjacency for all greylevel values of self
        current_alpha = self.interval_to_parent[0]
        current_omega = -1
        omega = self.interval_to_parent[1]
        adj_parent = first_adj
        while current_omega != omega:
            if self.ct_class == 'max':
                if self.interval_to_parent[1] <= adj_parent.interval_to_parent[0]:
                    current_omega = omega
                else:
                    current_omega = adj_parent.interval_to_parent[0]
            else:
                if self.interval_to_parent[1] >= adj_parent.interval_to_parent[0]:
                    current_omega = omega
                else:
                    current_omega = adj_parent.interval_to_parent[0]
            adj_interval = (current_alpha, current_omega)
            self.__add_adj_parent(adj_interval, adj_parent)
            current_alpha = current_omega + 1 if self.ct_class == 'max' else current_omega - 1
            adj_parent = adj_parent.get_ct_parent(adj_parent.interval_to_parent[0])

    def __add_ct_parent(self, parent):
        self.ct_parent = parent
        parent.ct_children.append(self)

    # add adj parent to the attribute list over the specified interval
    def __add_adj_parent(self, interval, parent):
        if self.ct_class == 'max':
            for i in range(interval[0], interval[1] + 1):
                self.adj_parents[i] = parent
        else:
            for i in range(interval[1], interval[0] + 1):
                self.adj_parents[i] = parent
        mirrored_interval = (interval[1], interval[0])
        parent.adj_children.append((self, mirrored_interval))

    # computes the necessary divisions of self depending on its configuration in the gos (gos_node)
    def compute_ctos_division(self, gos_node):
        gos_node.adj_children.sort(key=lambda x: x[1][0])
        base_interval = self.interval_to_parent
        pp = [] + self.proper_part
        if self.ct_class == 'min':
            gos_node.adj_children.reverse()
        current_alpha = self.interval_to_parent[0]
        current_node = self
        nbdiv = 0
        children_list = []
        for c in self.children:
            children_list.append(c)

        self.children.clear()
        # Go through all the holes of self and divide it accordingly
        for h in gos_node.adj_children:
            hole_interval = h[1]
            if hole_interval[0] != current_alpha:
                omega = hole_interval[0] - 1 if self.ct_class == 'max' else hole_interval[0] + 1
                current_node = self.divide_node(current_node, (current_alpha, omega))
                current_alpha = hole_interval[0]
                nbdiv += 1
        if nbdiv > 0:
            current_node.alt = base_interval[1]
            current_node.interval_to_parent = (current_alpha, base_interval[1])
            current_node.ct_class = self.ct_class

        for c in children_list:
            c.parent = current_node
        current_node.children += children_list

        current_node.compressed_proper_part[base_interval[1]] = pp

        return current_node

    def divide_node(self, node, new_interval):
        omega = node.interval_to_parent[1]
        node.interval_to_parent = new_interval

        child = Node(node.name, node.interval_to_parent[1], node)
        node.alt = new_interval[1]
        node.children.append(child)

        child.ct_class = node.ct_class
        child.parent = node
        child_alpha = new_interval[1] + 1 if node.ct_class == 'max' else new_interval[1] - 1
        child.interval_to_parent = (child_alpha, omega)

        return child

    def compress_in(self, node):
        node.interval_to_parent = (node.interval_to_parent[0], self.interval_to_parent[1])
        node.children += self.children
        node.last_compressed = self
        node.proper_part += self.proper_part
        if self.compressed_proper_part.get(self.interval_to_parent[1]) is not None:
            node.compressed_proper_part[self.interval_to_parent[1]] = (
                    [] + self.compressed_proper_part[self.interval_to_parent[1]])
        node.children.remove(self)

        self.parent = None
        for children in self.children:
            children.parent = node
        self.children.clear()
        self.removed = True

    # Change the node's altitude. Depending on the node's configuration,
    # this operation can lead to fusing with the parent (hence removal of the node) and/or the children nodes.
    # Returns:
    # If the node is not removed by the altitude change, returns itself
    # Else return its parent
    def change_node_altitude_in_bounds(self, new_alt):
        if self.alt == new_alt:
            return self
        parent_alt = self.__get_parent_alt()
        self.__compute_bounds()

        # Bounding of the new altitude
        # -1 for lower/upper bounds represents infinite
        # new_alt must remain between the lower and upper bounds of the node
        new_alt = self.__bound_value(new_alt)

        # if the new altitude is strictly between the lower and upper bound of the node,
        # there is no impact on the relations of the node
        if self.__is_value_strictly_between_bounds(new_alt):
            self.alt = new_alt
            return self

        # Look for impacted children (i.e. M+ / M-)
        if new_alt == self.lower_bound:
            self.__add_lower_bound_children()
            impacted_children = self.children_lower_bound
        else:
            self.__add_upper_bound_children()
            impacted_children = self.children_upper_bound

        # If the new altitude of the self node reaches some of its children,
        # these children fuse with the self node and are removed.
        # The children of the fused nodes are now children of the self node
        for impacted_child in impacted_children:
            impacted_child.__fuse_to_parent()

        self.alt = new_alt

        # if the new altitude of the self node reaches its parent node, the parent absorbs the self node,
        # and each child of the self node is now a child of the parent node
        if new_alt == parent_alt:
            parent = self.parent
            self.__fuse_to_parent()
            return parent
        return self

    # Returns the altitude closest to the node between the upper and lower bound
    def closest_bound(self):
        self.__compute_bounds()
        if self.parent is None:
            print(self)
        if self.lower_bound == -1 or self.upper_bound == self.alt:
            return self.upper_bound
        if self.upper_bound == -1:
            return self.lower_bound

        distance_lower = self.alt - self.lower_bound
        distance_upper = self.upper_bound - self.alt

        return self.upper_bound if distance_upper > distance_lower else self.lower_bound

    # The size of the proper part, i.e. the nb of pixels in the set
    def get_nb_proper_part(self):
        return len(self.proper_part)

    # m
    def distance_to_parent(self):
        par_alt = self.__get_parent_alt()
        return par_alt - self.alt if par_alt > self.alt else self.alt - par_alt

    # nb of gray levels separating the node from its closest node
    def distance_to_closest(self):
        self.__compute_bounds()

        dist_up = self.upper_bound - self.alt
        dist_low = self.alt - self.lower_bound

        if self.lower_bound == -1:
            return dist_up
        if self.upper_bound == -1:
            return dist_low

        return dist_up if dist_up <= dist_low else dist_low

    # Returns the mean value between all the neighbours (parent/children) of the node
    # weighted: do not use, keep False
    def get_mean_neighboring_value(self, weighted=False, rounding=False):
        alts = []
        if self.parent is not None:
            if weighted:
                alts = [(self.__get_parent_alt(), self.parent.get_nb_proper_part())]
            else:
                alts = [(self.__get_parent_alt(), 1)]
        for child in self.children:
            if weighted:
                alts.append((child.alt, child.get_nb_proper_part()))
            else:
                alts.append((child.alt, 1))

        if weighted:
            alts.append((self.alt, self.get_nb_proper_part()))
        else:
            alts.append((self.alt, 1))
        sum_alt = 0
        weight = 0
        for a, w in alts:
            sum_alt += a * w
            weight += w

        return int(round(sum_alt / weight, 0)) if rounding else sum_alt / weight

    def get_median_neighboring_value(self):
        alts = [self.alt]
        if not self.is_root():
            alts.append(self.__get_parent_alt())
        for child in self.children:
            alts.append(child.alt)
        return statistics.median(alts)

    def get_closest_extrema(self):
        if self.is_root():
            return
        neighbourhood = self.children.copy()
        neighbourhood.append(self.parent)
        neighbourhood.sort(key=lambda x: x.alt)

        if len(neighbourhood) == 1:
            return self.parent.alt

        lowest_alt = neighbourhood[0].alt
        highest_alt = neighbourhood[len(neighbourhood) - 1].alt
        return lowest_alt if abs(self.alt - lowest_alt) < abs(highest_alt - self.alt) else highest_alt

    def __fuse_to_parent(self):
        # for each child, remove the relation to self and link child to parent
        for child in self.children:
            child.parent = self.parent
            self.parent.children.append(child)
        self.children.clear()

        # proper part is given to parent
        self.parent.proper_part += self.proper_part
        self.proper_part.clear()

        # relation to parent is removed
        self.parent.children.remove(self)
        self.parent = None
        self.removed = True

    # Compute the bounds of the node
    # Bounds are defined depending on the neighbouring nodes' (parent and children) altitudes
    def __compute_bounds(self):
        parent_alt = self.__get_parent_alt()
        self.upper_bound = None
        self.lower_bound = None
        for child in self.children:
            if child.alt > self.alt:
                if self.upper_bound is None or child.alt < self.upper_bound:
                    self.upper_bound = child.alt
            elif child.alt < self.alt:
                if self.lower_bound is None or child.alt > self.lower_bound:
                    self.lower_bound = child.alt

        if parent_alt > self.alt:
            if self.upper_bound is None or self.upper_bound > parent_alt:
                self.upper_bound = parent_alt
        elif parent_alt < self.alt:
            if self.lower_bound is None or self.lower_bound < parent_alt:
                self.lower_bound = parent_alt

        # if no finite upper/lower bound was found, -1 represents an infinite bound
        if self.upper_bound is None:
            self.upper_bound = -1
        # we consider that the lower bound cannot be lower than 0 for now
        if self.lower_bound is None:
            if self.alt == 0:
                self.lower_bound = 0
            else:
                self.lower_bound = -1

    # add children node corresponding to the lower bound
    def __add_lower_bound_children(self):
        self.children_lower_bound = []
        if self.lower_bound == -1:
            return
        for child in self.children:
            if child.alt == self.lower_bound:
                self.children_lower_bound.append(child)

    # add children node corresponding to the upper bound
    def __add_upper_bound_children(self):
        self.children_upper_bound = []
        if self.upper_bound == -1:
            return
        for child in self.children:
            if child.alt == self.upper_bound:
                self.children_upper_bound.append(child)

    def __bound_value(self, value):
        if value < self.lower_bound != -1:
            return self.lower_bound
        elif value > self.upper_bound != -1:
            return self.upper_bound
        if self.is_root():
            if value <= self.alt:
                return self.alt
            else:
                return self.upper_bound
        return value

    def get_lower_bound(self):
        self.__compute_bounds()
        return self.lower_bound

    def get_upper_bound(self):
        self.__compute_bounds()
        return self.upper_bound

    def __is_value_strictly_between_bounds(self, value):
        return ((value > self.alt and (value < self.upper_bound or self.upper_bound == -1)) or
                (value < self.alt and (value > self.lower_bound or self.lower_bound == -1)))

    # Returns the parent's altitude. If the node is the root, returns the altitude of the node
    def __get_parent_alt(self):
        if self.parent is None:
            return self.alt
        else:
            return self.parent.alt

    # debugging
    def print_bounding(self):
        self.__compute_bounds()
        print("Bounding:", str(self.lower_bound), "<", str(self.alt), "<", self.upper_bound)

    def __str__(self):
        return str(self.name) + "-" + str(self.alt) + " area: " + str(self.area) + ", pp: " + str(len(self.proper_part))


def compute_children_map(sources, targets):
    children = {}
    i = 0
    for child in sources:
        if children.get(targets[i]) is None:
            children[targets[i]] = [child]
        else:
            children[targets[i]].append(child)
        i += 1
    return children


class TreeOfShapes:
    def __init__(self, image, enable_features=False):
        self.enriched = False
        self.is_3D = len(image.shape) == 3
        self.image = image
        self.nodes = {}
        self.max_alt = 0
        self.__hg_tab_to_obj(image)
        self.area_depth_map = None
        if enable_features:
            self.area_depth_map = []

    def __hg_tab_to_obj(self, image):
        if self.is_3D:
            nb_pixels = image.shape[0] * image.shape[1] * image.shape[2]
            g, altitude = hg.component_tree_tree_of_shapes_image3d(image)
        else:
            nb_pixels = image.shape[0] * image.shape[1]
            g, altitude = hg.component_tree_tree_of_shapes_image2d(image)

        # map of nodes
        self.nb_nodes = len(g.parents()) - nb_pixels
        # node altitudes
        self.altitude = altitude

        sources, targets = g.edge_list()
        children_map = compute_children_map(sources[nb_pixels:], targets[nb_pixels:])
        root = g.root()

        altitude_root = (
            int(bound(0, 255, round(altitude[root] * 255, 0)))) if 0 < altitude[root] < 1 \
            else int(altitude[root])
        self.root = Node(root, altitude_root, None)
        self.nodes[root] = self.root

        # construction of the tree structure from the root
        node_list = [self.root]
        while node_list:
            node = node_list.pop(0)
            children = children_map.get(node.name)
            if node.alt > self.max_alt:
                self.max_alt = node.alt
            if children is not None:
                for child in children:
                    altitude_child = (
                        int(bound(0, 255, round(altitude[child] * 255, 0)))) if 0 < altitude[child] < 1 \
                        else int(altitude[child])
                    child_node = Node(child, altitude_child, node)
                    self.nodes[child_node.name] = child_node
                    node.children.append(child_node)
                    node_list.append(child_node)

        self.compute_proper_parts(nb_pixels, g.parents())
        self.__compute_nodes_areas(self.root)

    def compute_proper_parts(self, nb_pix_im, parents):
        pix_index = 0
        for pixel in parents[:nb_pix_im]:
            self.nodes[pixel].proper_part.append(pix_index)
            pix_index += 1

    def enrich_tos(self):
        nodelist = [self.root]
        while nodelist:
            node = nodelist.pop(0)
            node.enrich_node()
            nodelist += node.children
        self.root.interval_to_parent = (self.max_alt + 1, self.root.interval_to_parent[1])
        self.enriched = True

    # Change the alt of the given node name
    # the new alt is bounded by the neighbourhood of the node
    # Impacts can fusion with parent/children
    def change_alt_of_node(self, node_name, new_alt):
        node = self.nodes.get(node_name)
        if node is not None:
            while node.alt != new_alt:
                node = node.change_node_altitude_in_bounds(new_alt)
            return node
        else:
            print("Node", node_name, "not found")
            return None

    # Consecutive proper part filtering, from starting value to end value with a step.
    def filter_tree_proper_part_bottom_up_consecutive(self, starting_pp_value, step, end_value, to_parent=True):
        while starting_pp_value < end_value:
            self.filter_tree_proper_part_bottom_up(starting_pp_value, to_parent)
            starting_pp_value += step

    # Simplifies the tree of shapes from the leaves to the root, by discarding the nodes that have a proper part smaller
    # than area_value
    # Discarding policy depends on to_parent flag:
    # True: node is merged to parent
    # False: node is merged to its "closest" node (can be parent or child).
    def filter_tree_proper_part_bottom_up(self, proper_part_value, to_parent=False):
        self.__filter_tree_proper_part_bottom_up(proper_part_value, self.root, to_parent)


    def __filter_tree_proper_part_bottom_up(self, proper_part_value, node, to_parent):
        childlist = [] + node.children
        for child in childlist:
            self.__filter_tree_proper_part_bottom_up(proper_part_value, child, to_parent)
        if not node.is_root() and not node.removed:
            if node.get_nb_proper_part() < proper_part_value:
                if not to_parent:
                    self.change_alt_of_node(node.name, node.closest_bound())
                else:
                    self.change_alt_of_node(node.name, node.parent.alt)

    def filter_tree_area_bottom_up_consecutive(self, starting_area_value, step, end_value, to_parent=True):
        while starting_area_value < end_value:
            self.filter_tree_area_bottom_up(starting_area_value, to_parent)
            starting_area_value += step

    def filter_tree_area_bottom_up(self, area_value, to_parent=False):
        self.__filter_tree_area_bottom_up(area_value, self.root, to_parent)

    def __filter_tree_area_bottom_up(self, area_value, node, to_parent=True):
        childlist = [] + node.children
        for child in childlist:
            self.__filter_tree_proper_part_bottom_up(area_value, child, to_parent)
        if not node.is_root() and not node.removed:
            if node.area < area_value:
                if not to_parent:
                    self.change_alt_of_node(node.name, node.closest_bound())
                else:
                    self.change_alt_of_node(node.name, node.parent.alt)

    # Remove nodes w/ area < area_value AND distance to parent < dynamic_value
    def filter_tree_area_dynamic_bottom_up(self, area_value, dynamic_value, absolute_area):
        self.__filter_tree_area_dynamic_bottom_up(area_value, self.root, dynamic_value, absolute_area)

    def __filter_tree_area_dynamic_bottom_up(self, area_value, node, dynamic_value, absolute_area):
        childlist = [] + node.children
        for child in childlist:
            self.__filter_tree_area_dynamic_bottom_up(area_value, child, dynamic_value, absolute_area)
        if not node.is_root() and not node.removed:
            if (node.area < area_value and node.distance_to_parent() < dynamic_value) or node.area < absolute_area:
                self.change_alt_of_node(node.name, node.parent.alt)

    def filter_tree_dynamic_bottom_up(self, dynamic_value, to_parent=True):
        self.__filter_tree_dynamic_bottom_up(self.root, dynamic_value, to_parent)

    def __filter_tree_dynamic_bottom_up(self, node, dynamic_value, to_parent):
        childlist = [] + node.children
        for child in childlist:
            self.__filter_tree_dynamic_bottom_up(child, dynamic_value, to_parent)
        if not node.is_root() and not node.removed:
            if node.distance_to_parent() < dynamic_value:
                if not node.parent.is_root():
                    self.change_alt_of_node(node.name, node.closest_bound())
                    #self.change_alt_of_node(node.name, node.parent.alt)

    def __compute_nodes_areas(self, node):
        childlist = node.children[:]
        for child in childlist:
            self.__compute_nodes_areas(child)
        node.area += len(node.proper_part)
        if not node.is_root():
            node.parent.area += node.area

    # fuses the node to its closer relative(s) if its area (= nb pixels of the proper part) is lesser than area_value
    # going from the root to the leaves
    def filter_tree_area(self, proper_part_value):
        nodelist = [self.root]
        visited = {}
        while nodelist:
            node = nodelist.pop(0)
            while not node.is_root() and node.get_nb_proper_part() < proper_part_value:
                node = self.change_alt_of_node(node.name, node.parent.alt)
            for child in node.children:
                if visited.get(child.name) is None:
                    visited[child.name] = True
                    nodelist.append(child)

    # Assigns to every node in the tree its mean neighbouring value
    def filter_tree_mean(self, weighted=False):
        nodelist = [self.root]
        visited = {}
        while nodelist:
            node = nodelist.pop(0)
            if not node.is_root():
                node = self.change_alt_of_node(node.name, node.get_mean_neighboring_value(weighted, rounding=True))
            for child in node.children:
                if visited.get(child.name) is None:
                    visited[child.name] = True
                    nodelist.append(child)

    # Similar to above but handles rounding differently
    def filter_tree_mean_v2(self):
        nodelist = [self.root]
        visited = {}
        while nodelist:
            node = nodelist.pop(0)
            if not node.is_root():
                # unrounded value
                target_alt = node.get_mean_neighboring_value(rounding=False)
                node = self.change_alt_of_node(node.name, round(target_alt, 0))
                if not node.is_root() and abs(node.parent.alt - target_alt) <= 1:
                    node = self.change_alt_of_node(node.name, node.parent.alt)
            for child in node.children:
                if visited.get(child.name) is None:
                    visited[child.name] = True
                    nodelist.append(child)

    def consecutive_mean_filter(self, nb_it):
        for i in range (0, nb_it):
            self.filter_tree_mean_v2()

    def filter_tree_dynamic_top_down(self, dynamic_value):
        nodelist = [self.root]
        visited = {}
        while nodelist:
            node = nodelist.pop(0)
            if not node.is_root():
                # unrounded value
                if node.distance_to_parent() < dynamic_value:
                    node = self.change_alt_of_node(node.name, node.parent.alt)
            for child in node.children:
                if visited.get(child.name) is None:
                    visited[child.name] = True
                    nodelist.append(child)

    def filter_tree_median(self):
        nodelist = [self.root]
        visited = {}
        while nodelist:
            node = nodelist.pop(0)
            #print("current node:", node)
            if not node.is_root():
                # unrounded value
                target_alt = node.get_median_neighboring_value()
                node = self.change_alt_of_node(node.name, round(target_alt, 0))
                if not node.is_root() and abs(node.parent.alt - target_alt) <= 1:
                    node = self.change_alt_of_node(node.name, node.parent.alt)
            for child in node.children:
                if visited.get(child.name) is None:
                    visited[child.name] = True
                    nodelist.append(child)

    def consecutive_median_filter(self, nb_it):
        for i in range(0, nb_it):
            self.filter_tree_median()

    def filter_tree_to_closest_extrema(self, area_limit=-1):
        sorted_list = self.__get_nodes_sorted_area()
        for node in sorted_list:
            if not node.removed:
                if not node.is_root():
                    if area_limit == -1:
                        if not node.is_leaf():
                            self.change_alt_of_node(node.name, node.get_closest_extrema())
                    else:
                        if node.area < area_limit:
                            self.change_alt_of_node(node.name, node.get_closest_extrema())

    def filter_tree_to_closest_top_down(self):
        nodelist = [self.root]
        visited = {}
        while nodelist:
            node = nodelist.pop(0)
            if not node.is_root() and not node.is_leaf() and node.area < 10000:
                self.change_alt_of_node(node.name, node.get_closest_extrema())
            for child in node.children:
                if visited.get(child.name) is None:
                    visited[child.name] = True
                    nodelist.append(child)


    # Simplify the tree by removing nodes that are further away than a certain value from their parent
    # If pp_value != 0: a criterion is also the proper part of the node
    def filter_tree_distance_to_parent(self, distance_value, pp_value=0):
        treat_area = pp_value != 0
        nodelist = [self.root]
        visited = {}
        while nodelist:
            node = nodelist.pop(0)
            while not node.is_root() and node.distance_to_parent() > distance_value:
                if treat_area and node.get_nb_proper_part() > pp_value:
                    break
                node = self.change_alt_of_node(node.name, node.parent.alt)
            for child in node.children:
                if visited.get(child.name) is None:
                    visited[child.name] = True
                    nodelist.append(child)

    # Filter nodes that have a distance to their closest bound lower than a certain treshold
    def filter_tree_gap(self, gap_value, area=0):
        treat_area = area != 0
        nodelist = [self.root]
        visited = {}
        while nodelist:
            node = nodelist.pop(0)
            # while not node.is_root() and node.distance_to_closest() < gap_value:
            while not node.is_root() and node.distance_to_parent() < gap_value:
                if treat_area and node.get_nb_proper_part() > area:
                    break
                node = self.change_alt_of_node(node.name, node.parent.alt)
            for child in node.children:
                if visited.get(child.name) is None:
                    visited[child.name] = True
                    nodelist.append(child)

    # if the value of a node is below (resp. over or equal to) value, it its altitude is decreased (resp. increased) by the value of increment
    def filter_darker_brighter(self, value, increment):
        nodelist = [self.root]
        visited = {}
        while nodelist:
            node = nodelist.pop(0)
            if not node.is_root():
                if node.alt < value:
                    target = node.alt - increment
                    if target < 0:
                        target = 0
                else:
                    target = node.alt + increment
                    if target > 255:
                        target = 255
                node = self.change_alt_of_node(node.name, target)
            for child in node.children:
                if visited.get(child.name) is None:
                    visited[child.name] = True
                    nodelist.append(child)

    def filter_dark_darker(self, value, increment):
        nodelist = [self.root]
        visited = {}
        while nodelist:
            node = nodelist.pop(0)
            if not node.is_root():
                if node.alt < value:
                    target = node.alt - increment
                    if target < 0:
                        target = 0
                    node = self.change_alt_of_node(node.name, target)
            for child in node.children:
                if visited.get(child.name) is None:
                    visited[child.name] = True
                    nodelist.append(child)

    # Filter the tree by shifting node altitudes randomly within a certain interval
    # Default order: nodes sorted by ascending proper part
    # To parent: random distance will be forced to be towards the parent
    # To children: the random distance won't be towards the parent
    # Random pick: nodes chosen at random
    # reverse_sort: nodes of bigger proper part first
    def filter_random_shifts(self, stop_percent, random_interval, to_parent=False, to_children=False, random_pick=False,
                             reverse_sort=False):
        sorted_list = self.__get_nodes_sorted_proper_part()
        nb_nodes = len(sorted_list)
        target_value = int((stop_percent * nb_nodes) / 100)
        print("nb node is", nb_nodes, "and target value is", target_value)
        while len(sorted_list) > target_value:
            if(random_pick):
                i = random.randint(0, len(sorted_list) - 1)
                node = sorted_list.pop(i)
                while node.removed:
                    i = random.randint(0, len(sorted_list) - 1)
                    node = sorted_list.pop(i)
            else:
                if not reverse_sort:
                    node = sorted_list.pop(0)
                    while node.removed:
                        node = sorted_list.pop(0)
                else:
                    node = sorted_list.pop(len(sorted_list) - 1)
                    while node.removed:
                        node = sorted_list.pop(len(sorted_list) - 1)

            if node.is_root():
                continue

            interval_0 = node.alt - random_interval
            if interval_0 < 0:
                interval_0 = 0

            interval_1 = node.alt + random_interval
            if node.get_lower_bound() == -1:
                interval_0 = node.alt
            if node.get_upper_bound() == -1:
                interval_1 = node.alt

            if to_parent:
                if node.parent is not None:
                    if node.parent.alt > node.alt:
                        interval_0 = node.alt
                    else:
                        interval_1 = node.alt
            elif to_children:
                if len(node.children) > 0 and node.parent is not None:
                    if node.parent.alt > node.alt:
                        interval_1 = node.alt
                    else:
                        interval_0 = node.alt

            v = random.randint(interval_0, interval_1)
            node.change_node_altitude_in_bounds(v)
            if not node.removed:
                for index, n in enumerate(sorted_list):
                    if n.get_nb_proper_part() >= node.get_nb_proper_part():
                        sorted_list.insert(index, node)
                        break

    # Quantize the grayscale in nb_g_values values
    def filter_tree_quantization(self, nb_g_values):
        low_gv = 0
        step_gv = int(255 / nb_g_values)
        target_gv = step_gv
        gv_map = {}
        for i in range(0, 256):
            gv_map[i] = (low_gv, target_gv if target_gv <= 255 else 255)
            if i >= target_gv:
                low_gv += step_gv
                target_gv += step_gv

        sorted_list = self.__get_nodes_sorted_proper_part()
        sorted_list.reverse()
        while sorted_list:
            node = sorted_list.pop(0)
            while node.removed and sorted_list:
                node = sorted_list.pop(0)
            if not sorted_list:
                break
            if node.is_root():
                continue
            alt_range = gv_map[node.alt]
            target_alt = alt_range[0] if (node.alt - alt_range[0]) > (alt_range[1] - node.alt) else alt_range[1]
            self.change_alt_of_node(node.name, target_alt)

    # Consecutive area filtering
    def apply_consecutive_area_filters(self, starting_value, ending_value, increment_value):
        while starting_value < ending_value:
            self.filter_tree_area(starting_value)
            starting_value += increment_value

    def compute_features(self):
        nodelist = [self.root]
        nb_nodes = 0
        avg_node_degree = 0
        nb_leaves = 0
        mean_intensity = 0
        mean_dynamic = 0
        mean_area = -1
        nb_polarity_switches = 0
        nb_small_nodes = -1
        ttos_node_diff = -1
        nb_branching = 0

        while nodelist:
            node = nodelist.pop(0)

            nb_nodes += 1
            avg_node_degree += len(node.children)
            if node.is_leaf():
                nb_leaves += 1
            mean_intensity += node.alt
            if not node.is_root():
                mean_dynamic += abs(node.alt - node.parent.alt)
            if not (node.is_root() or node.parent.is_root()):
                if not (node.alt > node.parent.alt > node.parent.parent.alt
                        or node.alt < node.parent.alt < node.parent.parent.alt):
                    nb_polarity_switches += 1
            if len(node.children) > 1:
                nb_branching += 1
            mean_area += node.area

            nodelist += node.children

        avg_node_degree = round(avg_node_degree / nb_nodes, 5)
        mean_intensity = round(mean_intensity / nb_nodes, 5)
        mean_dynamic = round(mean_dynamic / nb_nodes, 5)
        max_depth = self.max_depth(self.root)
        width = self.max_width()
        mean_area = round(mean_area / nb_nodes, 5)
        area, depth = zip(*self.area_depth_map)
        area_to_depth = np.corrcoef(area, depth)[0,1]
        features = {
            Feature.NB_NODES : nb_nodes,
            Feature.DEPTH : max_depth,
            Feature.AVG_NODE_DEGREE : avg_node_degree,
            Feature.WIDTH : width,
            Feature.NB_LEAVES : nb_leaves,
            Feature.MEAN_INTENSITY : mean_intensity,
            Feature.MEAN_DYNAMIC : mean_dynamic,
            Feature.MEAN_AREA : mean_area,
            Feature.AREA_TO_DEPTH_CORRELATION : area_to_depth,
            Feature.NB_POLARITY_SWITCH : nb_polarity_switches,
            Feature.NB_SMALL_NODES : nb_small_nodes,
            Feature.TTOS_NB_NODE_DIFFERENCE : ttos_node_diff,
            Feature.NB_BRANCHING : nb_branching
        }
        return features

    def max_depth(self, node, depth=0):
        node.depth = depth
        if self.area_depth_map is not None:
            self.area_depth_map.append((node.area, node.depth))
        if not node.children:
            return depth
        return max(self.max_depth(child, depth + 1) for child in node.children)

    def max_width(self):
        curr_nodes = [self.root]
        max_width = 1
        next_nodes = []
        while curr_nodes:
            for node in curr_nodes:
                next_nodes.extend(node.children)
            max_width = max(len(next_nodes), max_width)
            curr_nodes = next_nodes[:]
            next_nodes.clear()
        return max_width

    def reconstruct_image(self):
        im = np.zeros(self.image.shape).astype('uint8')
        nodelist = [self.root]
        while nodelist:
            node = nodelist.pop(0)
            for px in node.proper_part:
                set_px(im, px, node.alt)
            nodelist += node.children
        return im

    def get_node(self, node_name):
        return self.nodes.get(node_name)

    # tree profiling
    def stats_tree(self, verbose = False):
        nb_unary = 0
        nb_close_to_parent = 0
        nb_nodes = 0
        nb_leaves = 0
        nb_close_to_parent_not_leaf = 0
        nb_extrema_towards_parent = 0
        nodelist = [self.root]
        while nodelist:
            node = nodelist.pop(0)
            nb_nodes += 1
            if not node.is_root():
                if node.is_leaf():
                    nb_leaves += 1
                elif len(node.children) == 1:
                    nb_unary += 1
                if not node.is_leaf():
                    extr = node.get_closest_extrema()
                    if node.alt < extr <= node.parent.alt or node.alt > extr >= node.parent.alt:
                        nb_extrema_towards_parent += 1
                if abs(node.alt - node.parent.alt) < 2:
                    nb_close_to_parent += 1
                    if not node.is_leaf():
                        nb_close_to_parent_not_leaf += 1
            nodelist += node.children

        if verbose:
            print("nb nodes =", nb_nodes)
            print("nb leaves =", nb_leaves, str(round(nb_leaves/nb_nodes, 3) * 100), "%")
            print("nb unary non leaf =", nb_unary, str(round(nb_unary/nb_nodes, 3) * 100), "%")
            print("nb close to parent =", nb_close_to_parent, str(round(nb_close_to_parent/nb_nodes, 3) * 100), "%")
            print("nb close to parent not leaf =", nb_close_to_parent_not_leaf, str(round(nb_close_to_parent_not_leaf/nb_nodes, 3) * 100), "%")
            print("nb extrema towards parent =", nb_extrema_towards_parent, str(round(nb_extrema_towards_parent/nb_nodes, 3) * 100), "%")

        stat_map = {'nb_nodes': nb_nodes,
                    'nb_leaves': nb_leaves,
                    'nb_non_leaf_unary': nb_unary,
                    'nb_one_to_parent': nb_close_to_parent,
                    'nb_leaves_pct': round((nb_leaves / nb_nodes) * 100, 3),
                    'nb_non_leaf_unary_pct': round((nb_unary / nb_nodes) * 100, 3),
                    'nb_one_to_parent_pct': round((nb_close_to_parent / nb_nodes) * 100, 3)}
        return stat_map

    def get_nb_nodes(self):
        nodelist = [self.root]
        nb = 0
        while nodelist:
            node = nodelist.pop(0)
            nb += 1
            nodelist += node.children
        return nb

    def __get_nodes_sorted_proper_part(self):
        nodestack = [self.root]
        result = []
        while nodestack:
            node = nodestack.pop(0)
            result.append(node)
            nodestack += node.children
        result.sort(key=lambda x: x.get_nb_proper_part())
        return result

    def __get_nodes_sorted_area(self):
        nodestack = [self.root]
        result = []
        while nodestack:
            node = nodestack.pop(0)
            result.append(node)
            nodestack += node.children
        result.sort(key=lambda x: x.area)
        return result

    def __get_1d_canvas(self):
        return [0] * self.image.shape[0] * self.image.shape[1]

    # debugging
    def print_tree(self):
        nodelist = [self.root]
        while nodelist:
            node = nodelist.pop(0)
            print(node, "children: ", end="")
            for child in node.children:
                print(child, end=", ")
            node.print_bounding()
            nodelist += node.children

    # returns a 1D image s.t. each pixel's value is the node to which this pixel is proper part
    def node_label_image(self):
        arr1d = self.__get_1d_canvas()
        nodelist = [self.root]
        while nodelist:
            node = nodelist.pop(0)
            for px in node.proper_part:
                arr1d[px] = node.name
            nodelist += node.children
        return arr1d
    def get_area_depth_map(self):
        return self.area_depth_map

    def __eq__(self, other):
        if type(self) is not type(other):
            return False

        # Structural check
        label_map = {}
        label_img1 = self.node_label_image()
        label_img2 = other.node_label_image()

        if len(label_img1) != len(label_img2):
            print("false because length")
            return False

        for label1, label2 in zip(label_img1, label_img2):
            lab = label_map.get(label1)
            if lab is None:
                label_map[label1] = label2
            elif lab != label2:
                print("false because", lab, "is not", label2)
                return False

        # Node info check (altitude values)
        im1 = self.reconstruct_image()
        im2 = other.reconstruct_image()

        return im1 == im2
