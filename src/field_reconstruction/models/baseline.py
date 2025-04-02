import numpy as np
from scipy.interpolate import griddata

import numpy as np
from scipy.interpolate import griddata

class Interpolator:
    
    def __init__(self, order: int):
        self.order = order
        self.method = self._get_method(order)
        
    def __str__(self):
        if self.order in [1, 2, 3]:
            orders = ["Linear", "Quadratic", "Cubic"]
            return f"{orders[self.order - 1]} interpolator"
        else:
            return f"{self.order}th order interpolator"

    def _get_method(self, order):
        if order == 1:
            return "linear"
        elif order == 3:
            return "cubic"
        elif order == 2:
            return "linear"  # Note: scipy doesn't support quadratic → fallback
        else:
            return "nearest"  # fallback for unsupported orders

    def forward(self, sensor_coords, sensor_values, grid_shape):
        """
        Interpolate sparse sensor values onto a full grid.
        
        Args:
            sensor_coords: list of (row, col) positions (Y, X) of sensors
            sensor_values: list or array of sensor values
            grid_shape: tuple (H, W), the shape of the output grid
        
        Returns:
            grid: interpolated 2D array of shape (H, W)
        """
        H, W = grid_shape
        grid_y, grid_x = np.mgrid[0:H, 0:W]

        # Convert coordinates to (x, y) pairs
        points = np.array([(x, y) for y, x in sensor_coords])
        values = np.array(sensor_values)

        # Interpolate onto full grid
        grid = griddata(points, values, (grid_x, grid_y), method=self.method, fill_value=np.nan)
        return grid 