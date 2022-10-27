from data import relis3d,semantickitti
from torch.utils.data import DataLoader

def setup_loaders(dataset, path_list, batch_size):
	if(dataset == 'rellis3d'):
		data = relis3d.Relis3D(path_list)
		data_loaded = DataLoader(data, batch_size=batch_size)
		return data_loaded
	elif(dataset == 'semantickitti'):
		data = semantickitti.SemanticKitti(path_list)
		data_loaded = DataLoader(data, batch_size=batch_size)
		return data_loaded