# PPR and DDE module wrapper
# This module provides access to PersonalizedPageRank and DirectedDistanceEncoding
# from the gnn.utils.ppr module

try:
    from gnn.utils.ppr import PersonalizedPageRank, DirectedDistanceEncoding
except ImportError:
    import sys
    import os
    # Add the gnn directory to the path
    gnn_path = os.path.join(os.path.dirname(__file__), 'gnn')
    if gnn_path not in sys.path:
        sys.path.insert(0, gnn_path)
    from utils.ppr import PersonalizedPageRank, DirectedDistanceEncoding

__all__ = ['PersonalizedPageRank', 'DirectedDistanceEncoding']