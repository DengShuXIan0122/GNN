# BMSSP module wrapper
# This module provides access to BMSSPSolver
# from the gnn.utils.bmssp module

try:
    from gnn.utils.bmssp import BMSSPSolver
except ImportError:
    import sys
    import os
    # Add the gnn directory to the path
    gnn_path = os.path.join(os.path.dirname(__file__), 'gnn')
    if gnn_path not in sys.path:
        sys.path.insert(0, gnn_path)
    from utils.bmssp import BMSSPSolver

__all__ = ['BMSSPSolver']