from tqdm import tqdm
from pathlib import Path
import torch, os, sys, argparse
from torch.utils.data import DataLoader

import instafilter
from instafilter.model import ColorNet
from dataset import ColorizedDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_func = torch.optim.AdamW

module_location = Path(instafilter.__file__).resolve().parent


def train_image_pair(
	f_source,
	f_target,
	f_save_model,
	batch_size=2 ** 10,
	n_epochs=30,
	max_learning_rate=0.01,
):

	assert Path(f_source).exists()
	assert Path(f_target).exists()

	data = ColorizedDataset(f_source, f_target, device=device)
	train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

	net = ColorNet()
	criterion = torch.nn.L1Loss()
	optimizer = loss_func(net.parameters())

	scheduler = torch.optim.lr_scheduler.OneCycleLR(
		optimizer,
		max_lr=max_learning_rate,
		steps_per_epoch=len(train_loader),
		epochs=n_epochs,
	)

	net.to(device)
	net.train()

	for epoch in tqdm(range(n_epochs)):
		# monitor training loss
		train_loss = 0.0

		for data, target in train_loader:

			optimizer.zero_grad()

			output = net(data)
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()
			scheduler.step()

			# update running training loss
			train_loss += loss.item() * data.size(0)

		train_loss = train_loss / len(train_loader.dataset)
		print(f"Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f}")

	torch.save(net.state_dict(), f_save_model)

def train_images( d_source, d_target, f_save_model, batch_size=2 ** 10,
	n_epochs=30, max_learning_rate=0.01):
	'''
	Args:
		d_source: directory of the source images
		d_target: directory of the target images
		f_save_model: filepath of the output model
	'''
	assert os.path.isdir(d_source), "d_source is not a valid directory"
	assert os.path.isdir(d_target), "d_target is not a valid directory"
	# make sure all source and target images are aligned
	file_check = [os.path.isfile(os.path.join(d_target, f))
		for f in os.listdir(d_source) if f.endswith(('.jpg','.JPG'))]
	assert all(file_check), "not all files in d_source is matched with all files in d_target"

	data = ColorizedDataset(f_source = d_source, f_target = d_target,
			device=device, debug_mode = False)
	train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

	net = ColorNet()
	criterion = torch.nn.L1Loss()
	optimizer = loss_func(net.parameters())

	scheduler = torch.optim.lr_scheduler.OneCycleLR(
		optimizer,
		max_lr=max_learning_rate,
		steps_per_epoch=len(train_loader),
		epochs=n_epochs,
	)

	net.to(device)
	net.train()

	for epoch in tqdm(range(n_epochs), desc = "ColorNet training"):
		# monitor training loss
		train_loss = 0.0

		for data, target in train_loader:

			optimizer.zero_grad()

			output = net(data)
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()
			scheduler.step()

			# update running training loss
			train_loss += loss.item() * data.size(0)

		train_loss = train_loss / len(train_loader.dataset)
		print(f"Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f}")

	torch.save(net.state_dict(), f_save_model)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Instafilter: simple color NN to create instagram-like filters")
	parser.add_argument("--source_dir", required = False, type = str, default = None,
							help = 'directory of images before transformation')
	parser.add_argument("--target_dir", required = False, type = str, default = None,
							help = 'directory of images AFTER transformation')
	parser.add_argument("--model_name", required = False, type = str, default = None,
							help = 'output filter name (.pt not neccessary)')
	parser.add_argument("--epochs", required = False, type = int, default = 30,
							help = 'number of epochs to train for. (default: 30)')
	parser.add_argument("--bs", required = False, type = int, default = 2**10,
							help = 'batch size. (default: 2**10)')
	args = parser.parse_args()
	model_location = module_location / "models"

	if args.source_dir and args.target_dir:
		f_model = os.path.join(model_location, args.model_name + '.pt') if args.model_name else \
			os.path.join(model_location, os.path.basename(args.target_dir) + '.pt')
		print(f'Training model {os.path.basename(f_model)} with images in {args.source_dir} and {args.target_dir}')
		train_images(d_source = args.source_dir, d_target= args.target_dir,
			f_save_model= f_model, n_epochs = args.epochs, batch_size = args.bs)
	else:
		f_source = "input/Normal.jpg"

		for f_target in tqdm(Path("input").glob("*.jpg"), desc = f"looking for models to train in {Path('input')}"):

			if "Normal.jpg" in str(f_target):
				continue

			f_model = model_location / f_target.name.replace(".jpg", ".pt")

			if f_target.name == f_source:
				continue

			if f_model.exists():
				continue

			print("Training", f_model)

			train_image_pair(f_source, f_target, f_model, n_epochs=args.epochs,
				batch_size = args.bs)
