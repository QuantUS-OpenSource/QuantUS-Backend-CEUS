from typing import Tuple
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..time_series_analysis.curves.framework import CurvesAnalysis

class ParamapDrawingBase(ABC):
    """Facilitate parametric map visualizations of ultrasound images.
    """

    def __init__(self):
        # Cmap library
        summer_cmap = plt.get_cmap("summer")
        summer_cmap = [summer_cmap(i)[:3] for i in range(summer_cmap.N)]
        winter_cmap = plt.get_cmap("winter")
        winter_cmap = [winter_cmap(i)[:3] for i in range(winter_cmap.N)]
        autumn_cmap = plt.get_cmap("autumn")  # Fixed typo in "autunm"
        autumn_cmap = [autumn_cmap(i)[:3] for i in range(autumn_cmap.N)]
        spring_cmap = plt.get_cmap("spring")
        spring_cmap = [spring_cmap(i)[:3] for i in range(spring_cmap.N)]
        cool_cmap = plt.get_cmap("cool")
        cool_cmap = [cool_cmap(i)[:3] for i in range(cool_cmap.N)]
        hot_cmap = plt.get_cmap("hot")
        hot_cmap = [hot_cmap(i)[:3] for i in range(hot_cmap.N)]
        bone_cmap = plt.get_cmap("bone")
        bone_cmap = [bone_cmap(i)[:3] for i in range(bone_cmap.N)]
        copper_cmap = plt.get_cmap("copper")
        copper_cmap = [copper_cmap(i)[:3] for i in range(copper_cmap.N)]
        jet_cmap = plt.get_cmap("jet")
        jet_cmap = [jet_cmap(i)[:3] for i in range(jet_cmap.N)]
        self.cmaps = [np.array(plt.get_cmap("viridis").colors), np.array(plt.get_cmap("magma").colors),
                    np.array(plt.get_cmap("plasma").colors), np.array(plt.get_cmap("inferno").colors),
                    np.array(plt.get_cmap("cividis").colors), np.array(summer_cmap),
                    np.array(winter_cmap), np.array(autumn_cmap), np.array(spring_cmap),
                    np.array(cool_cmap), np.array(hot_cmap), np.array(bone_cmap), np.array(copper_cmap),
                    np.array(jet_cmap)]
        self.cmap_names = ["viridis", "magma", "plasma", "inferno", "cividis", "summer", "winter", "autumn",
                        "spring", "cool", "hot", "bone", "copper", "jet"]
        
    @abstractmethod
    def export_visualizations(self):
        """Used to specify which visualizations to export and where.
        """
        pass
