import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import argparse

print('Helloooo!!')

def setup_device():
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]
        print('Dataset is being created')

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]

def ddp_setup(device):
    print('DDP is being set up...')
    backend = "nccl" if device.type == 'cuda' else "gloo"
    init_process_group(backend=backend, rank=int(os.environ['RANK']), world_size=int(os.environ['WORLD_SIZE']))
    print(f"DDP using {backend} backend")

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
        device,
    ) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.model = DDP(self.model, device_ids=[self.device.index] if self.device.type == 'cuda' else None)

        os.makedirs(os.path.dirname(self.snapshot_path), exist_ok=True)
        if os.path.exists(self.snapshot_path):
            self.load_snapshot()

    def load_snapshot(self):
        snapshot = torch.load(self.snapshot_path, map_location=self.device)
        self.model.module.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[Device{self.device.index if self.device.type == 'cuda' else 'CPU'}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.device)
            targets = targets.to(self.device)
            self._run_batch(source, targets)

    def train(self, max_epochs: int):
        print('Starting the training...')
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if (self.device.type == 'cuda' and self.device.index == 0) or (self.device.type == 'cpu') and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=(device.type == 'cuda'),
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main(save_every: int, total_epochs: int, batch_size: int, snapshot_dir: str = "/tmp/runai-elastic-torch-training"):
    device = setup_device()
    snapshot_path = os.path.join(snapshot_dir, "snapshot.pt")
    ddp_setup(device)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size, device)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path, device)
    trainer.train(total_epochs)
    destroy_process_group()
    print('Training is done, destroyed the process group')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    main(args.save_every, args.total_epochs, args.batch_size)
