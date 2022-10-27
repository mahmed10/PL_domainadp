from data import relis3d, semantickitti, cityscapes, gta5, ourdataset, umbc
from torch.utils.data import DataLoader, ConcatDataset

def setup_loaders(dataset, path_list, batch_size):
	combined_data = []
	for i in range(len(dataset)):
		if(dataset[i] == 'rellis3d'):
			data = relis3d.Relis3D(path_list[i])
			data_loader = DataLoader(data, drop_last=True, batch_size=batch_size)
			if (len(dataset) == 1):
				return data_loader
			else:
				combined_data.append(data)
				del data_loader
				del data
		if(dataset[i] == 'semantickitti'):
			data = semantickitti.SemanticKitti(path_list[i])
			data_loader = DataLoader(data, drop_last=True, batch_size=batch_size)
			if (len(dataset) == 1):
				return data_loader
			else:
				combined_data.append(data)
				del data_loader
				del data
		if(dataset[i] == 'cityscapes'):
			data = cityscapes.CityScapes(path_list[i])
			data_loader = DataLoader(data, drop_last=True, batch_size=batch_size)
			if (len(dataset) == 1):
				return data_loader
			else:
				combined_data.append(data)
				del data_loader
				del data

		if(dataset[i] == 'gta5'):
			data = gta5.GTA5(path_list[i])
			data_loader = DataLoader(data, drop_last=True, batch_size=batch_size)
			if (len(dataset) == 1):
				return data_loader
			else:
				combined_data.append(data)
				del data_loader
				del data

		if(dataset[i] == 'umbc'):
			data = umbc.UMBC(path_list[i])
			data_loader = DataLoader(data, drop_last=True, batch_size=batch_size)
			if (len(dataset) == 1):
				return data_loader
			else:
				combined_data.append(data)
				del data_loader
				del data

		if(dataset[i] == 'ourdataset'):
			data = ourdataset.Dataset(path_list[i])
			data_loader = DataLoader(data, drop_last=True, batch_size=batch_size)
			if (len(dataset) == 1):
				return data_loader
			else:
				combined_data.append(data)
				del data_loader
				del data

	data_loader = DataLoader(
		ConcatDataset(combined_data),
		batch_size=batch_size, shuffle=True)
	return data_loader