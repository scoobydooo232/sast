import numpy as np
import torch
import sys
import sys
from pathlib import Path
sys.path.append('../..')

# sys.path.append("workspace/Event-Gen1-ToolBox")
toolbox_path = Path('/workspace/Event-Gen1-ToolBox')
if str(toolbox_path) not in sys.path:
    sys.path.append(str(toolbox_path))

class DatEventReader:
    def __init__(self, file_path: str):
        # Initialize the PSEELoader with the .dat file
        self.video = PSEELoader(file_path)
        
        # DatReader usually reads the header to get dimensions. 
        # If PSEELoader doesn't expose them directly, you might need to access the underlying _file object
        # or hardcode them if known (e.g., Gen1: 304x240, Gen4: 1280x720).
        # Typically available via the underlying reader:
        self.height, self.width = self.video.get_size() 

    def get_event_slice(self, idx_start: int, idx_end: int, convert_2_torch: bool = True):
        # assert self.video.is_open # PSEELoader handles file opening in init
        assert idx_end >= idx_start
        
        # Seek to the starting event index
        self.video.seek_event(idx_start)
        
        # Calculate the number of events to load
        count = idx_end - idx_start
        
        # Load the chunk of events
        # This returns a structured numpy array with fields 't', 'x', 'y', 'p'
        events = self.video.load_n_events(count)
        
        # Extract columns into separate arrays
        # .dat files use microsecond timestamps (int64 usually)
        t_array = events['t'].astype(np.int64) 
        x_array = events['x'].astype(np.int64)
        y_array = events['y'].astype(np.int64)
        p_array = events['p'].astype(np.int64)
        
        # Ensure polarity is 0/1 (Prophesee dat files often use 0/1, but sometimes -1/1 depending on encoding)
        # If needed, clip or shift. Standard dat is usually 0 and 1.
        p_array = np.clip(p_array, a_min=0, a_max=None)
        
        assert np.all(t_array[:-1] <= t_array[1:])
        
        ev_data = dict(
            x=x_array if not convert_2_torch else torch.from_numpy(x_array),
            y=y_array if not convert_2_torch else torch.from_numpy(y_array),
            p=p_array if not convert_2_torch else torch.from_numpy(p_array),
            t=t_array if not convert_2_torch else torch.from_numpy(t_array),
            height=self.height,
            width=self.width,
        )
        return ev_data