from helper import *
from treeofshapes import *

def nb_nodes(root):
    nodelist = [root]
    nb = 0
    while nodelist:
        node = nodelist.pop(0)
        nb += 1
        nodelist += node.children
    return nb

def split_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

class TopologicalToS:
    def __init__(self, tos, until='ttos'):
        self.enriched_tos = copy.deepcopy(tos)
        self.enriched_tos.enrich_tos()

        if until == 'tos':
            self.state = 'tos'
            return

        self.gos = copy.deepcopy(self.enriched_tos)
        self.__compute_gos()

        if until == 'gos':
            self.state = 'gos'
            return
        self.ctos = self.enriched_tos
        self.__compute_ctos()
        self.nb_nodes_ctos = nb_nodes(self.ctos.root)
        print('CToS:', self.nb_nodes_ctos, 'nodes.')

        if until == 'ctos':
            self.state = 'ctos'
            return

        self.ttos = self.ctos
        self.__compute_ttos()
        self.nb_nodes_ttos = nb_nodes(self.ttos.root)
        print('TToS:', self.nb_nodes_ttos, 'nodes.')
        self.state = 'ttos'

    def __compute_gos(self):
        nodelist = [self.gos.root]
        while nodelist:
            node = nodelist.pop(0)
            node.compute_gos_relations()
            nodelist += node.children

    def __compute_ctos(self):
        nodelist = [self.ctos.root]
        while nodelist:
            node = nodelist.pop(0)
            node = node.compute_ctos_division(self.__in_gos(node))
            nodelist += node.children

    def __compute_ttos(self):
        nodelist = [self.ttos.root]
        while nodelist:
            node = nodelist.pop(0)
            nodelist += node.children
            if not node.is_root():
                if self.__is_topo_equivalent(node.parent, node):
                    node.compress_in(node.parent)

    def __is_topo_equivalent(self, parent_node, node):
        # Two nodes from different classes cannot be topologically equivalent
        if parent_node.ct_class != node.ct_class:
            return False

        # GoS configuration of both ctos nodes
        parent_gos = self.__in_gos(parent_node)
        node_gos = self.__in_gos(node)

        # In the case the parent node already regroups some nodes, the last regrouped node is the one to consider for
        # the topological equivalence check
        p_gos_node = parent_gos if parent_node.last_compressed is None \
            else self.__in_gos(parent_node.last_compressed)

        # The node must be the ONLY component tree child of the parent node in the GoS
        gos_ct_children = p_gos_node.get_ct_children(parent_node.interval_to_parent[1])
        if len(gos_ct_children) == 1 and gos_ct_children[0].name == node.name:
            # Get the adjacent children of parent node at his omega
            parent_adj_children = p_gos_node.get_adj_children(parent_node.interval_to_parent[1])
            # Get the adjacent children of the node at his alpha
            node_adj_children = node_gos.get_adj_children(node.interval_to_parent[0])

            # Number of adj children must be equal
            if len(parent_adj_children) == len(node_adj_children):
                bijection = False
                checked_map = {}
                # For equivalence, for a given adj child node Cp of the parent, either:
                #   - Cp is also in the list of adj children of node
                #   - The component tree parent of Cp is in the list of adj children of node
                for par_child in parent_adj_children:
                    for node_child in node_adj_children:
                        if par_child.name == node_child.name or node_child == par_child.ct_parent:
                            if checked_map.get(par_child.name) is None:
                                checked_map[par_child.name] = True
                            else:
                                return False
                            bijection = True
                            break
                    if bijection is True:
                        bijection = False
                    else:
                        return False
                return True
        return False

    def reconstruct_img(self, strategy='orig'):
        im = np.zeros(self.enriched_tos.image.shape).astype('uint8')
        nodelist = [self.ttos.root]
        while nodelist:
            node = nodelist.pop(0)
            for alt in node.compressed_proper_part.keys():
                for pixel in node.compressed_proper_part[alt]:
                    set_px(im, pixel, self.compute_alt(node, alt, strategy))
                if node.last_compressed is not None:
                    if node.last_compressed.alt == node.interval_to_parent[1]:
                        for pixel in node.last_compressed.proper_part:
                            set_px(im, pixel, node.interval_to_parent[1])
            nodelist += node.children
        return im

    def compute_alt(self, node, alt, strategy):
        if strategy == 'orig':
            return alt if node.ct_class == 'max' else alt - 1
        if strategy == 'quasi_closing':
            return self.__quasi_closing(node)
        if strategy == 'quasi_opening':
            return self.__quasi_opening(node)
        if strategy == 'higher':
            return self.__to_higher(node)
        if strategy == 'lower':
            return self.__to_lower(node)
        else:
            return 0

    @staticmethod
    def __quasi_closing(node):
        if node.ct_class == 'min':
            return node.parent.interval_to_parent[1] - 1
        else:
            return node.interval_to_parent[1]

    @staticmethod
    def __quasi_opening(node):
        if node.ct_class == 'min':
            return node.interval_to_parent[1] - 1
        else:
            return node.parent.interval_to_parent[1]

    @staticmethod
    def __to_lower(node):
        if node.ct_class == 'min':
            return node.interval_to_parent[1] - 1
        else:
            return node.interval_to_parent[1]

    @staticmethod
    def __to_higher(node):
        if node.ct_class == 'min':
            return node.parent.interval_to_parent[1] - 1
        else:
            return node.parent.interval_to_parent[1]


    def __in_gos(self, node):
        return self.gos.nodes[node.name]